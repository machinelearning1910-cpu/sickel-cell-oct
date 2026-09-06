

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise SystemExit("Install segmentation-models-pytorch first.") from exc

SEED = 42
NUM_CLASSES = 8
IMAGE_HEIGHT, IMAGE_WIDTH = 256, 576
FINAL_THRESHOLD = 0.4425
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], np.float32)

# Exact notebook outputs that the supplied paper is intended to report.
# These are used only as reproducibility assertions, never as computed results.
REFERENCE = {
    "segmentation": {
        "test_loss": 0.0814, "ce_loss": 0.0917, "dice_loss": 0.0658,
        "foreground_dice": 0.9489, "foreground_iou": 0.9034,
        "all_class_dice": 0.9548, "pixel_accuracy": 0.9807,
    },
    "validation_lora": {
        "accuracy": 0.906170, "precision": 0.901015, "recall": 0.912596,
        "specificity": 0.899743, "f1": 0.906769, "auroc": 0.964526,
        "average_precision": 0.959189, "brier": 0.074565, "mcc": 0.812406,
    },
    "locked_base": {
        "accuracy": 0.916367, "precision": 0.926335, "recall": 0.904676,
        "specificity": 0.928058, "f1": 0.915378, "auroc": 0.961185,
        "average_precision": 0.967289, "brier": 0.070787, "mcc": 0.832962,
        "tn": 516, "fp": 40, "fn": 53, "tp": 503,
    },
    "locked_lora": {
        "accuracy": 0.916367, "precision": 0.924771, "recall": 0.906475,
        "specificity": 0.926259, "f1": 0.915531, "auroc": 0.960991,
        "average_precision": 0.967114, "brier": 0.069979, "mcc": 0.832897,
        "tn": 515, "fp": 41, "fn": 52, "tp": 504,
    },
    "natural_lora": {
        "accuracy": 0.919501, "precision": 0.848739, "recall": 0.906643,
        "specificity": 0.925435, "f1": 0.876736, "auroc": 0.962993,
        "average_precision": 0.937771, "brier": 0.059704, "mcc": 0.818049,
        "tn": 1117, "fp": 90, "fn": 52, "tp": 505,
    },
    "parameters": {"base": 12_289_154, "lora": 167_936, "final": 12_457_090, "percentage": 1.3481},
}


def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def dump_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(obj, f, indent=2)


def paths(root: Path) -> Dict[str, Path]:
    p={
        "seg_data":root/"01_data/processed/retinal_layer_segmentation",
        "seg_split":root/"01_data/splits/retinal_layer_split_indices.npz",
        "seg_meta":root/"01_data/metadata/retinal_layer_class_weights.npy",
        "seg_ckpt":root/"02_segmentation_pretraining/checkpoints/segmentation_full_best.pt",
        "seg_history":root/"02_segmentation_pretraining/results/segmentation_full_history.csv",
        "csat":root/"01_data/processed/csat_quality_controlled",
        "natural":root/"01_data/processed/csat_natural_test",
        "base_ckpt":root/"03_disease_classification/checkpoints/csat_patch_mil_v4_best.pt",
        "lora_ckpt":root/"03_disease_classification/checkpoints/csat_patch_mil_v4_lora_refined_best.pt",
        "v4_history":root/"03_disease_classification/results/csat_patch_mil_v4_history.csv",
        "lora_history":root/"03_disease_classification/results/csat_patch_mil_v4_lora_history.csv",
        "result":root/"03_disease_classification/results",
        "fig":root/"04_figures",
        "report":root/"03_disease_classification/results/clinical_reports",
    }
    for k in ("result","fig","report"): p[k].mkdir(parents=True,exist_ok=True)
    return p


# =============================================================================
# Models — exactly the trained architectures, without new layers.
# =============================================================================

class CSATPatchMILV4(nn.Module):
    def __init__(self, dropout=(0.40,0.25)):
        super().__init__(); self.encoder=resnet18(weights=None); self.encoder.fc=nn.Identity()
        def projection(cin): return nn.Sequential(nn.Conv2d(cin,128,1,bias=False),nn.BatchNorm2d(128),nn.GELU())
        self.stage8_projection=projection(128); self.stage16_projection=projection(256); self.stage32_projection=projection(512)
        self.feature_fusion=nn.Sequential(nn.Conv2d(384,192,3,padding=1,bias=False),nn.BatchNorm2d(192),nn.GELU(),nn.Conv2d(192,128,3,padding=1,bias=False),nn.BatchNorm2d(128),nn.GELU())
        self.patch_head=nn.Conv2d(128,1,1)
        self.classifier=nn.Sequential(nn.Linear(514,192),nn.LayerNorm(192),nn.GELU(),nn.Dropout(dropout[0]),nn.Linear(192,64),nn.GELU(),nn.Dropout(dropout[1]),nn.Linear(64,1))
    def forward(self,image):
        x=self.encoder.maxpool(self.encoder.relu(self.encoder.bn1(self.encoder.conv1(image))))
        x=self.encoder.layer1(x); stage8=self.encoder.layer2(x); stage16=self.encoder.layer3(stage8); stage32=self.encoder.layer4(stage16)
        f8=self.stage8_projection(stage8); f16=F.interpolate(self.stage16_projection(stage16),size=f8.shape[-2:],mode="bilinear",align_corners=False); f32=F.interpolate(self.stage32_projection(stage32),size=f8.shape[-2:],mode="bilinear",align_corners=False)
        fused=self.feature_fusion(torch.cat([f8,f16,f32],1)); patch=self.patch_head(fused); flat=patch.flatten(1); k=max(1,int(flat.shape[1]*0.10))
        top=torch.topk(flat,k,dim=1).values.mean(1,keepdim=True); mean=flat.mean(1,keepdim=True); glob=F.adaptive_avg_pool2d(stage32,1).flatten(1)
        image=self.classifier(torch.cat([glob,top,mean],1)).squeeze(1); return {"image_logits":image,"patch_logits":patch.squeeze(1)}


class LoRAConv2d(nn.Module):
    def __init__(self,base:nn.Conv2d,rank=8,alpha=16.0):
        super().__init__(); self.base=base; self.rank=rank; self.alpha=alpha; self.scale=alpha/rank
        for p in self.base.parameters(): p.requires_grad=False
        self.lora_A=nn.Conv2d(base.in_channels,rank,1,bias=False)
        self.lora_B=nn.Conv2d(rank,base.out_channels,kernel_size=base.kernel_size,stride=base.stride,padding=base.padding,dilation=base.dilation,bias=False)
        nn.init.kaiming_normal_(self.lora_A.weight); nn.init.zeros_(self.lora_B.weight)
    def forward(self,x): return self.base(x)+self.scale*self.lora_B(self.lora_A(x))


def get_module(model,path):
    cur=model
    for part in path.split("."): cur=cur[int(part)] if part.isdigit() else getattr(cur,part)
    return cur


def set_module(model,path,module):
    parts=path.split("."); cur=model
    for part in parts[:-1]: cur=cur[int(part)] if part.isdigit() else getattr(cur,part)
    last=parts[-1]
    if last.isdigit(): cur[int(last)]=module
    else: setattr(cur,last,module)


def inject_lora(model):
    targets=["encoder.layer4.0.conv1","encoder.layer4.0.conv2","encoder.layer4.0.downsample.0","encoder.layer4.1.conv1","encoder.layer4.1.conv2"]
    for p in model.parameters(): p.requires_grad=False
    for name in targets: set_module(model,name,LoRAConv2d(get_module(model,name),8,16.0))
    return targets


def load_models(p:Dict[str,Path],device):
    base_ck=torch.load(p["base_ckpt"],map_location="cpu",weights_only=False)
    base=CSATPatchMILV4(dropout=(0.40,0.25)); base.load_state_dict(base_ck["ema_state"],strict=True); base.to(device).eval()
    lora_ck=torch.load(p["lora_ckpt"],map_location="cpu",weights_only=False)
    lora=CSATPatchMILV4(dropout=(0.30,0.20)); lora.load_state_dict(base_ck["ema_state"],strict=True); inject_lora(lora); lora.load_state_dict(lora_ck["model_state"],strict=True); lora.to(device).eval()
    selected=float(lora_ck.get("validation_metrics",{}).get("threshold",FINAL_THRESHOLD))
    if abs(selected-FINAL_THRESHOLD)>5e-4:
        raise RuntimeError(f"Final checkpoint threshold {selected:.6f} does not match paper-selected 0.4425")
    return base,lora,base_ck,lora_ck


# =============================================================================
# Datasets/inference
# =============================================================================

class GrayOCTDataset(Dataset):
    def __init__(self,image_path:Path,label_path:Path,manifest_path:Path):
        self.images=np.load(image_path,mmap_mode="r"); self.labels=np.load(label_path,mmap_mode="r"); self.manifest=pd.read_csv(manifest_path)
        if "array_index" in self.manifest: self.manifest=self.manifest.sort_values("array_index").reset_index(drop=True)
        if not (len(self.images)==len(self.labels)==len(self.manifest)): raise RuntimeError("Image/label/manifest length mismatch")
        if "scr_target" in self.manifest and not np.array_equal(self.manifest.scr_target.to_numpy(np.uint8),np.asarray(self.labels)): raise RuntimeError("Manifest labels not aligned")
    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        x=np.asarray(self.images[i],dtype=np.float32).copy()/255.0; x=np.repeat(x[...,None],3,axis=-1); x=(x-IMAGENET_MEAN)/IMAGENET_STD; x=np.transpose(x,(2,0,1))
        return torch.from_numpy(x).float(),torch.tensor(int(self.labels[i])),torch.tensor(i)


def split_dataset(p,split): return GrayOCTDataset(p["csat"]/f"{split}_images_uint8.npy",p["csat"]/f"{split}_labels_uint8.npy",p["csat"]/f"{split}_manifest.csv")
def natural_dataset(p): return GrayOCTDataset(p["natural"]/"natural_test_images_uint8.npy",p["natural"]/"natural_test_labels_uint8.npy",p["natural"]/"natural_test_manifest.csv")


def infer(model,ds,device,batch_size=16,return_patch=False):
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True); probs=[]; targets=[]; patches=[]
    with torch.inference_mode():
        for x,y,_ in tqdm(loader,desc="Inference",leave=False):
            x=x.to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                o1=model(x); o2=model(torch.flip(x,dims=[3])); pr=(torch.sigmoid(o1["image_logits"])+torch.sigmoid(o2["image_logits"]))/2
            probs.extend(pr.cpu().numpy()); targets.extend(y.numpy())
            if return_patch: patches.extend(torch.sigmoid(o1["patch_logits"]).cpu().numpy())
    return np.asarray(targets,int),np.asarray(probs,float),patches


def metrics(y,p,threshold):
    pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {
        "accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),
        "specificity":float(tn/max(tn+fp,1)),"f1":float(f1_score(y,pred,zero_division=0)),"auroc":float(roc_auc_score(y,p)),
        "average_precision":float(average_precision_score(y,p)),"brier":float(brier_score_loss(y,p)),"mcc":float(matthews_corrcoef(y,pred)),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"threshold":float(threshold),"n":int(len(y)),
    }


def reference_check(name:str,actual:Dict[str,float],strict:bool,tol=7e-4):
    if name not in REFERENCE: return
    mismatches=[]
    for key,exp in REFERENCE[name].items():
        if key not in actual: continue
        val=actual[key]
        if isinstance(exp,int): ok=int(val)==exp
        else: ok=abs(float(val)-float(exp))<=tol
        if not ok: mismatches.append((key,val,exp))
    if mismatches:
        msg=f"{name} differs from supplied notebook/paper reference: {mismatches}"
        if strict: raise RuntimeError(msg)
        print("WARNING:",msg)


# =============================================================================
# Segmentation evaluation
# =============================================================================

class SegDataset(Dataset):
    def __init__(self,p,indices):
        self.images=np.load(p["seg_data"]/"preprocessed_images.npy",mmap_mode="r"); self.masks=np.load(p["seg_data"]/"preprocessed_masks.npy",mmap_mode="r"); self.ids=np.asarray(indices,int)
    def __len__(self): return len(self.ids)
    def __getitem__(self,j):
        i=int(self.ids[j]); x=np.asarray(self.images[i],np.float32).copy()/255.0; x=np.stack([x,x,x],0); x=(x-IMAGENET_MEAN[:,None,None])/IMAGENET_STD[:,None,None]
        return torch.from_numpy(x).float(),torch.from_numpy(np.asarray(self.masks[i],np.int64).copy()).long(),torch.tensor(i)


def dice_loss(logits,y):
    pr=torch.softmax(logits,1); oh=F.one_hot(y,NUM_CLASSES).permute(0,3,1,2).float(); inter=(pr*oh).sum((0,2,3)); den=pr.sum((0,2,3))+oh.sum((0,2,3)); return 1-((2*inter+1e-6)/(den+1e-6)).mean()


def evaluate_segmentation(p,device,strict):
    split=np.load(p["seg_split"]); ds=SegDataset(p,split["test_indices"]); loader=DataLoader(ds,batch_size=4,shuffle=False,num_workers=2,pin_memory=True)
    model=smp.Unet(encoder_name="resnet18",encoder_weights=None,in_channels=3,classes=8); ck=torch.load(p["seg_ckpt"],map_location="cpu",weights_only=False); model.load_state_dict(ck["model_state"],strict=True); model.to(device).eval()
    cw=torch.tensor(np.load(p["seg_meta"]),dtype=torch.float32,device=device); cm=np.zeros((8,8),np.int64); sums=np.zeros(4,float); sample_cache=[]
    with torch.inference_mode():
        for x,y,idx in tqdm(loader,desc="Segmentation test"):
            x,y=x.to(device),y.to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                logits=model(x); ce=F.cross_entropy(logits,y,weight=cw); dl=dice_loss(logits,y); total=.60*ce+.40*dl
            pred=logits.argmax(1); b=x.size(0); sums += [total.item()*b,ce.item()*b,dl.item()*b,b]
            yy=y.cpu().numpy().ravel(); pp=pred.cpu().numpy().ravel(); cm += confusion_matrix(yy,pp,labels=list(range(8)))
            if len(sample_cache)<3:
                for k in range(min(b,3-len(sample_cache))): sample_cache.append((int(idx[k]),y[k].cpu().numpy(),pred[k].cpu().numpy()))
    n=max(sums[3],1); rows=[]
    for c in range(8):
        tp=cm[c,c]; fn=cm[c,:].sum()-tp; fp=cm[:,c].sum()-tp; dice=(2*tp)/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0; iou=tp/(tp+fp+fn) if tp+fp+fn else 1.0; precision=tp/(tp+fp) if tp+fp else 1.0; recall=tp/(tp+fn) if tp+fn else 1.0
        rows.append({"class":c,"dice":dice,"iou":iou,"precision":precision,"recall":recall})
    cls=pd.DataFrame(rows); total_pixels=cm.sum(); overall={"test_loss":sums[0]/n,"ce_loss":sums[1]/n,"dice_loss":sums[2]/n,"foreground_dice":float(cls.loc[cls['class']>0,'dice'].mean()),"foreground_iou":float(cls.loc[cls['class']>0,'iou'].mean()),"all_class_dice":float(cls.dice.mean()),"pixel_accuracy":float(np.trace(cm)/total_pixels)}
    reference_check("segmentation",overall,strict)
    outdir=p["result"]/"segmentation_evaluation"; outdir.mkdir(parents=True,exist_ok=True); cls.to_csv(outdir/"classwise_segmentation_metrics.csv",index=False); np.savetxt(outdir/"pixel_confusion_matrix.csv",cm,delimiter=",",fmt="%d"); dump_json(overall,outdir/"overall_segmentation_metrics.json")
    # Class-wise bars
    fig,ax=plt.subplots(figsize=(9,5)); x=np.arange(8); w=.38; ax.bar(x-w/2,cls.dice,w,label="Dice"); ax.bar(x+w/2,cls.iou,w,label="IoU"); ax.set_xticks(x,[f"Class {i}" for i in x]); ax.set_ylim(0,1.05); ax.legend(); ax.set_ylabel("Score"); fig.tight_layout(); fig.savefig(p["fig"]/"segmentation_classwise_dice_iou.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # Row-normalized confusion matrix
    norm=cm/np.maximum(cm.sum(1,keepdims=True),1); fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(norm,vmin=0,vmax=1,cmap="Blues"); ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_xticks(range(8)); ax.set_yticks(range(8));
    for i in range(8):
        for j in range(8): ax.text(j,i,f"{norm[i,j]:.2f}",ha="center",va="center",fontsize=7)
    fig.colorbar(im,ax=ax,label="Proportion"); fig.tight_layout(); fig.savefig(p["fig"]/"segmentation_pixel_confusion_matrix.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # Qualitative examples
    raw=np.load(p["seg_data"]/"preprocessed_images.npy",mmap_mode="r"); fig,ax=plt.subplots(len(sample_cache),3,figsize=(13,3*len(sample_cache))); ax=np.atleast_2d(ax)
    for r,(idx,gt,pred) in enumerate(sample_cache):
        ax[r,0].imshow(raw[idx],cmap="gray"); ax[r,0].set_title(f"Test OCT {idx}"); ax[r,1].imshow(gt,cmap="nipy_spectral",vmin=0,vmax=7); ax[r,1].set_title("Ground truth"); ax[r,2].imshow(pred,cmap="nipy_spectral",vmin=0,vmax=7); ax[r,2].set_title("Prediction")
        for c in range(3): ax[r,c].axis("off")
    fig.tight_layout(); fig.savefig(p["fig"]/"segmentation_test_predictions.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    return overall,cls


# =============================================================================
# Classification, calibration, patient-level and natural-distribution evaluation
# =============================================================================


def save_prediction_frame(ds,y,pred_prob,threshold,path,probability_name):
    frame=ds.manifest.copy(); frame["scr_target"]=y; frame[probability_name]=pred_prob; frame["fixed_threshold"]=threshold; frame["scr_prediction"]=(pred_prob>=threshold).astype(int)
    frame["prediction_status"]=np.select([(y==1)&(frame.scr_prediction==1),(y==0)&(frame.scr_prediction==0),(y==0)&(frame.scr_prediction==1),(y==1)&(frame.scr_prediction==0)],["TP","TN","FP","FN"],default="Unknown")
    frame.to_csv(path,index=False); return frame


def plot_roc_pr(yv,pv,yt,pt,p):
    fprv,tprv,_=roc_curve(yv,pv); fprt,tprt,_=roc_curve(yt,pt); fig,ax=plt.subplots(figsize=(7,6)); ax.plot(fprv,tprv,label=f"Validation AUROC={roc_auc_score(yv,pv):.4f}"); ax.plot(fprt,tprt,label=f"Locked test AUROC={roc_auc_score(yt,pt):.4f}"); ax.plot([0,1],[0,1],'--'); ax.set(xlabel="False-positive rate",ylabel="True-positive rate"); ax.legend(); fig.tight_layout(); fig.savefig(p["fig"]/"final_validation_locked_test_roc.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    prv,rcv,_=precision_recall_curve(yv,pv); prt,rct,_=precision_recall_curve(yt,pt); fig,ax=plt.subplots(figsize=(7,6)); ax.plot(rcv,prv,label=f"Validation AP={average_precision_score(yv,pv):.4f}"); ax.plot(rct,prt,label=f"Locked test AP={average_precision_score(yt,pt):.4f}"); ax.set(xlabel="Recall",ylabel="Precision"); ax.legend(); fig.tight_layout(); fig.savefig(p["fig"]/"final_validation_locked_test_pr.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def plot_confusion(m,name,path):
    cm=np.array([[m["tn"],m["fp"]],[m["fn"],m["tp"]]]); fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap="Blues"); ax.set_xticks([0,1],["Non-SCR","SCR"]); ax.set_yticks([0,1],["Non-SCR","SCR"]); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(name)
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=13)
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig)


def calibration_plot(y,pv,yt,pt,p):
    fig,ax=plt.subplots(figsize=(7,6));
    for yx,px,label in [(yv,pv,"Validation"),(yt,pt,"Locked test")]:
        frac,mean=calibration_curve(yx,px,n_bins=10,strategy="quantile"); ax.plot(mean,frac,marker='o',label=f"{label} (Brier={brier_score_loss(yx,px):.4f})")
    ax.plot([0,1],[0,1],'--'); ax.set(xlabel="Mean predicted probability",ylabel="Observed SCR fraction"); ax.legend(); fig.tight_layout(); fig.savefig(p["fig"]/"final_calibration_curve.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def patient_summary(frame,prob_col):
    rows=[]
    for patient,g in frame.groupby("patient_id"):
        rows.append({"patient_id":patient,"number_of_scans":len(g),"scr_scans":int(g.scr_target.sum()),"non_scr_scans":int(len(g)-g.scr_target.sum()),"mean_scr_probability":float(g[prob_col].mean()),"scan_accuracy":float(accuracy_score(g.scr_target,g.scr_prediction))})
    return pd.DataFrame(rows)


def cluster_bootstrap(frame,prob_col,threshold,n_boot=1000,seed=SEED):
    rng=np.random.default_rng(seed); patients=np.asarray(frame.patient_id.astype(str).unique()); rows=[]
    for _ in tqdm(range(n_boot),desc="Patient-cluster bootstrap",leave=False):
        sampled=rng.choice(patients,size=len(patients),replace=True); parts=[]
        for j,pid in enumerate(sampled):
            g=frame[frame.patient_id.astype(str)==pid].copy(); g["_cluster_draw"]=j; parts.append(g)
        b=pd.concat(parts,ignore_index=True); m=metrics(b.scr_target.to_numpy(int),b[prob_col].to_numpy(float),threshold)
        rows.append({k:m[k] for k in ("accuracy","precision","recall","specificity","f1","auroc","average_precision")})
    boot=pd.DataFrame(rows); summary={}
    for c in boot.columns: summary[c]={"estimate":float(metrics(frame.scr_target.to_numpy(int),frame[prob_col].to_numpy(float),threshold)[c]),"ci95_low":float(boot[c].quantile(.025)),"ci95_high":float(boot[c].quantile(.975))}
    return boot,summary


def evaluate_classification(p,device,strict):
    base,lora,base_ck,lora_ck=load_models(p,device)
    val_ds,test_ds=split_dataset(p,"validation"),split_dataset(p,"test")
    yv,pv_lora,_=infer(lora,val_ds,device); yt,pt_lora,patches=infer(lora,test_ds,device,return_patch=True)
    _,pv_base,_=infer(base,val_ds,device); _,pt_base,_=infer(base,test_ds,device)
    base_threshold=float(base_ck.get("best_threshold", base_ck.get("validation",{}).get("threshold",0.436)))
    mvb=metrics(yv,pv_base,base_threshold); mvl=metrics(yv,pv_lora,FINAL_THRESHOLD); mtb=metrics(yt,pt_base,base_threshold); mtl=metrics(yt,pt_lora,FINAL_THRESHOLD)
    reference_check("validation_lora",mvl,strict); reference_check("locked_base",mtb,strict); reference_check("locked_lora",mtl,strict)
    result_dir=p["result"]; val_frame=save_prediction_frame(val_ds,yv,pv_lora,FINAL_THRESHOLD,result_dir/"csat_patch_mil_v4_lora_validation_predictions.csv","lora_probability"); test_frame=save_prediction_frame(test_ds,yt,pt_lora,FINAL_THRESHOLD,result_dir/"csat_patch_mil_v4_lora_test_predictions.csv","lora_probability")
    save_prediction_frame(test_ds,yt,pt_base,base_threshold,result_dir/"csat_patch_mil_v4_test_predictions.csv","scr_probability")
    table=pd.DataFrame([
        {"split":"Validation","model":"Base V4",**mvb},{"split":"Validation","model":"V4-ConvLoRA",**mvl},
        {"split":"Locked test","model":"Base V4",**mtb},{"split":"Locked test","model":"V4-ConvLoRA",**mtl},
    ])
    table.to_csv(result_dir/"corrected_table9_validation_locked_test.csv",index=False)
    dump_json({"base_validation":mvb,"lora_validation":mvl,"base_locked_test":mtb,"lora_locked_test":mtl},result_dir/"final_classification_metrics.json")
    plot_roc_pr(yv,pv_lora,yt,pt_lora,p); plot_confusion(mtb,"Base V4 — locked test",p["fig"]/"locked_test_confusion_base_v4.png"); plot_confusion(mtl,"V4-ConvLoRA — locked test",p["fig"]/"locked_test_confusion_v4_convlora.png"); calibration_plot(yv,pv_lora,yt,pt_lora,p)
    # Probability distributions
    fig,ax=plt.subplots(figsize=(8,5)); ax.hist(pt_lora[yt==0],bins=30,alpha=.6,label="Non-SCR"); ax.hist(pt_lora[yt==1],bins=30,alpha=.6,label="SCR"); ax.axvline(FINAL_THRESHOLD,linestyle='--',label=f"threshold={FINAL_THRESHOLD:.4f}"); ax.set_xlabel("SCR probability"); ax.set_ylabel("Scans"); ax.legend(); fig.tight_layout(); fig.savefig(p["fig"]/"locked_test_probability_distribution.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # Patient-level final LoRA
    ps=patient_summary(test_frame,"lora_probability"); ps.to_csv(result_dir/"locked_test_patient_summary_final_lora.csv",index=False)
    fig,ax=plt.subplots(figsize=(11,5)); ax.bar(np.arange(len(ps)),ps.scan_accuracy); ax.axhline(mtl["accuracy"],linestyle='--',label="overall scan accuracy"); ax.set_xticks(np.arange(len(ps)),ps.patient_id.astype(str),rotation=75); ax.set_ylim(0,1.05); ax.set_ylabel("Scan-level accuracy"); ax.legend(); fig.tight_layout(); fig.savefig(p["fig"]/"final_lora_patient_scan_accuracy.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # Corrected patient-cluster bootstrap: FINAL LoRA, not base V4.
    boot,ci=cluster_bootstrap(test_frame,"lora_probability",FINAL_THRESHOLD,1000); boot.to_csv(result_dir/"final_lora_patient_cluster_bootstrap.csv",index=False); dump_json(ci,result_dir/"final_lora_patient_cluster_bootstrap_ci.json")
    fig,ax=plt.subplots(figsize=(10,5)); names=list(ci); est=[ci[k]["estimate"] for k in names]; lo=[ci[k]["estimate"]-ci[k]["ci95_low"] for k in names]; hi=[ci[k]["ci95_high"]-ci[k]["estimate"] for k in names]; ax.errorbar(np.arange(len(names)),est,yerr=[lo,hi],fmt='o',capsize=4); ax.set_xticks(np.arange(len(names)),names,rotation=25,ha='right'); ax.set_ylim(0.5,1.02); ax.set_ylabel("Metric with 95% patient-cluster CI"); fig.tight_layout(); fig.savefig(p["fig"]/"final_lora_patient_cluster_bootstrap_ci.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # Parameter analysis
    base_params=sum(x.numel() for x in base.parameters()); final_params=sum(x.numel() for x in lora.parameters()); lora_params=sum(x.numel() for x in lora.parameters() if x.requires_grad)
    param={"base_parameters":base_params,"trainable_lora_parameters":lora_params,"final_model_parameters":final_params,"trainable_percentage":100*lora_params/final_params}
    reference_check("parameters",{"base":base_params,"lora":lora_params,"final":final_params,"percentage":param["trainable_percentage"]},strict,tol=2e-4); dump_json(param,result_dir/"convlora_parameter_analysis.json")
    fig,ax=plt.subplots(figsize=(6,5)); ax.bar(["Frozen V4","Trainable ConvLoRA"],[base_params,lora_params]); ax.set_ylabel("Parameters"); ax.set_title(f"ConvLoRA trainable = {param['trainable_percentage']:.4f}%"); fig.tight_layout(); fig.savefig(p["fig"]/"convlora_parameter_efficiency.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    return {"base":base,"lora":lora,"base_threshold":base_threshold,"metrics":mtl,"val_metrics":mvl,"test_frame":test_frame,"patches":patches,"test_ds":test_ds,"lora_ck":lora_ck}


def evaluate_natural(p,device,lora,strict):
    ds=natural_dataset(p); y,prob,_=infer(lora,ds,device); m=metrics(y,prob,FINAL_THRESHOLD); reference_check("natural_lora",m,strict)
    outdir=p["result"]/"natural_distribution_test"; outdir.mkdir(parents=True,exist_ok=True)
    frame=save_prediction_frame(ds,y,prob,FINAL_THRESHOLD,outdir/"natural_test_predictions.csv","scr_probability")
    dump_json(m,outdir/"natural_test_metrics.json")
    ps=patient_summary(frame,"scr_probability"); ps.to_csv(outdir/"natural_test_patient_summary.csv",index=False)
    # Distribution and performance plots
    fig,ax=plt.subplots(figsize=(6,4)); counts=[int((y==0).sum()),int((y==1).sum())]; ax.bar(["Non-SCR","SCR"],counts); ax.set_ylabel("Scans"); ax.set_title(f"Natural test n={len(y):,}, SCR={100*y.mean():.2f}%"); fig.tight_layout(); fig.savefig(p["fig"]/"natural_test_class_distribution.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    keys=["accuracy","precision","recall","specificity","f1","auroc","average_precision","mcc"]; fig,ax=plt.subplots(figsize=(10,5)); vals=[m[k] for k in keys]; ax.bar(keys,vals); ax.set_ylim(0,1.05); ax.tick_params(axis='x',rotation=30); ax.set_ylabel("Score"); fig.tight_layout(); fig.savefig(p["fig"]/"natural_test_performance.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    return m,frame


# =============================================================================
# Final-model evidence and constrained clinical-style report
# =============================================================================


def strongest_evidence(patch:np.ndarray):
    h,w=patch.shape; idx=np.unravel_index(np.argmax(patch),patch.shape); cy=(idx[0]+0.5)*IMAGE_HEIGHT/h; cx=(idx[1]+0.5)*IMAGE_WIDTH/w
    concentration=float(np.sort(patch.ravel())[-max(1,int(.10*patch.size)):].mean())
    region=("superior" if cy<IMAGE_HEIGHT/2 else "inferior")+"-"+("temporal/left" if cx<IMAGE_WIDTH/2 else "nasal/right")
    return float(cx),float(cy),concentration,region


def evidence_figure(context,p):
    frame=context["test_frame"].reset_index(drop=True); ds=context["test_ds"]; patches=context["patches"]
    statuses=[]
    for s in ("TP","TN","FP","FN"):
        idx=np.where(frame.prediction_status.to_numpy()==s)[0]
        if len(idx): statuses.append((s,int(idx[0])))
    fig,ax=plt.subplots(len(statuses),3,figsize=(14,3.2*len(statuses))); ax=np.atleast_2d(ax)
    raw=ds.images
    for r,(status,i) in enumerate(statuses):
        img=np.asarray(raw[i]); patch=np.asarray(patches[i],np.float32); heat=torch.from_numpy(patch)[None,None]; up=F.interpolate(heat,size=(IMAGE_HEIGHT,IMAGE_WIDTH),mode="bilinear",align_corners=False)[0,0].numpy(); cx,cy,conc,region=strongest_evidence(patch)
        ax[r,0].imshow(img,cmap="gray"); ax[r,0].set_title(f"{status}: raw OCT")
        ax[r,1].imshow(img,cmap="gray"); ax[r,1].imshow(up,cmap="hot",alpha=.45); ax[r,1].set_title("Model-derived relative patch evidence")
        boxw,boxh=IMAGE_WIDTH*.18,IMAGE_HEIGHT*.22; ax[r,2].imshow(img,cmap="gray"); ax[r,2].add_patch(plt.Rectangle((max(0,cx-boxw/2),max(0,cy-boxh/2)),boxw,boxh,fill=False,linewidth=2)); ax[r,2].set_title(f"Strongest evidence: {region}")
        for c in range(3): ax[r,c].axis("off")
    fig.tight_layout(); fig.savefig(p["fig"]/"final_lora_qualitative_patch_evidence.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def deterministic_report(fields:Dict[str,Any]) -> str:
    return (f"The model classifies this structural OCT as {fields['predicted_pattern']} with a scan-level SCR probability of {fields['scan_probability']:.3f} "
            f"and a patient-level mean probability of {fields['patient_probability']:.3f}. The strongest model-derived regional evidence is located in the {fields['evidence_location']} region, "
            f"with relative evidence concentration {fields['evidence_concentration']:.3f}. Scan agreement is {fields['scan_agreement']:.3f}. "
            "These findings are model-derived decision-support outputs and do not replace ophthalmic interpretation.")


def generate_llm_report(fields:Dict[str,Any],enable_llm:bool=True) -> Tuple[str,str]:
    fallback=deterministic_report(fields)
    if not enable_llm: return fallback,"deterministic_fallback"
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model_name="google/flan-t5-small"; tok=AutoTokenizer.from_pretrained(model_name); mdl=AutoModelForSeq2SeqLM.from_pretrained(model_name)
        prompt=("Write one concise clinical-style decision-support paragraph using ONLY the verified fields below. Do not add symptoms, demographics, disease stage, treatment recommendations, diagnoses beyond the predicted OCT pattern, or unsupported facts. Do not change the prediction.\n"
                f"Predicted OCT pattern: {fields['predicted_pattern']}\nScan SCR probability: {fields['scan_probability']:.4f}\nPatient mean SCR probability: {fields['patient_probability']:.4f}\nDecision threshold: {FINAL_THRESHOLD:.4f}\n"
                f"Confidence: {fields['confidence']:.4f}\nScan agreement: {fields['scan_agreement']:.4f}\nEvidence location: {fields['evidence_location']}\nEvidence concentration: {fields['evidence_concentration']:.4f}\nImage quality: {fields['image_quality']}\n"
                "End by stating that this is decision support and not a substitute for ophthalmic review.")
        encoded=tok(prompt,return_tensors="pt",truncation=True,max_length=512); output=mdl.generate(**encoded,max_new_tokens=120,num_beams=4,do_sample=False); text=tok.decode(output[0],skip_special_tokens=True).strip()
        if len(text)<30: return fallback,"deterministic_fallback"
        return text,"flan_t5_small"
    except Exception as exc:
        print(f"FLAN-T5-small unavailable; using deterministic fallback: {exc}")
        return fallback,"deterministic_fallback"


def final_report(context,p,enable_llm):
    frame=context["test_frame"].reset_index(drop=True); patches=context["patches"]; ds=context["test_ds"]
    # Select a correct positive representative when available, otherwise highest-confidence scan.
    tp=np.where(frame.prediction_status.to_numpy()=="TP")[0]; i=int(tp[np.argmax(frame.loc[tp,"lora_probability"].to_numpy())]) if len(tp) else int(np.argmax(np.abs(frame.lora_probability.to_numpy()-.5)))
    row=frame.iloc[i]; patient=str(row.patient_id); group=frame[frame.patient_id.astype(str)==patient]; prob=float(row.lora_probability); patient_prob=float(group.lora_probability.mean()); pred=int(prob>=FINAL_THRESHOLD); agreement=float((group.scr_prediction==pred).mean()); cx,cy,conc,region=strongest_evidence(np.asarray(patches[i]))
    sharp=float(row.get("sharpness",np.nan)); contrast=float(row.get("contrast",np.nan)); image_quality="quality-controlled / acceptable" if np.isfinite(sharp) and np.isfinite(contrast) else "quality-controlled"
    fields={"patient_id":patient,"scan_index":i,"predicted_pattern":"SCR-associated OCT pattern" if pred else "Non-SCR OCT pattern","scan_probability":prob,"patient_probability":patient_prob,"confidence":max(prob,1-prob),"scan_agreement":agreement,"evidence_location":region,"evidence_x":cx,"evidence_y":cy,"evidence_concentration":conc,"image_quality":image_quality,"fixed_threshold":FINAL_THRESHOLD,"model":"V4-ConvLoRA"}
    text,source=generate_llm_report(fields,enable_llm); fields["report_source"]=source; fields["generated_report"]=text; dump_json(fields,p["report"]/"representative_final_lora_report.json")
    # Report figure uses FINAL ConvLoRA output/evidence.
    img=np.asarray(ds.images[i]); patch=np.asarray(patches[i],np.float32); up=F.interpolate(torch.from_numpy(patch)[None,None],size=(IMAGE_HEIGHT,IMAGE_WIDTH),mode="bilinear",align_corners=False)[0,0].numpy()
    fig=plt.figure(figsize=(12,7)); gs=fig.add_gridspec(2,2,height_ratios=[2,1]); a1=fig.add_subplot(gs[0,0]); a2=fig.add_subplot(gs[0,1]); a3=fig.add_subplot(gs[1,:]); a1.imshow(img,cmap="gray"); a1.set_title("Raw OCT"); a1.axis("off"); a2.imshow(img,cmap="gray"); a2.imshow(up,cmap="hot",alpha=.45); a2.set_title(f"Final V4-ConvLoRA evidence | p={prob:.3f}"); a2.axis("off"); a3.axis("off"); a3.text(0,1,text,va="top",wrap=True,fontsize=10); fig.tight_layout(); fig.savefig(p["fig"]/"final_lora_clinical_style_report.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    return fields


# =============================================================================
# Training-history figures and consolidated outputs
# =============================================================================


def plot_histories(p):
    if p["seg_history"].exists():
        h=pd.read_csv(p["seg_history"]); fig,ax=plt.subplots(figsize=(8,5)); ax.plot(h.epoch,h.train_loss,label="Training loss"); ax.plot(h.epoch,h.validation_loss,label="Validation loss"); ax.axvline(int(h.loc[h.validation_foreground_dice.idxmax(),"epoch"]),linestyle='--',label="selected"); ax.set(xlabel="Epoch",ylabel="Loss"); ax.legend(); fig.tight_layout(); fig.savefig(p["fig"]/"segmentation_training_validation_loss.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    if p["v4_history"].exists():
        h=pd.read_csv(p["v4_history"]); fig,ax=plt.subplots(figsize=(9,5)); ax.plot(h.epoch,h.train_total_loss,label="Training total loss"); ax.plot(h.epoch,h.validation_f1,label="Validation F1"); ax.plot(h.epoch,h.validation_auroc,label="Validation AUROC"); selected=int(h.loc[h.selection_score.idxmax(),"epoch"]); ax.axvline(selected,linestyle='--',label=f"selected epoch {selected}"); ax.set_xlabel("Epoch"); ax.legend(); fig.tight_layout(); fig.savefig(p["fig"]/"v4_training_validation_overview.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    if p["lora_history"].exists():
        h=pd.read_csv(p["lora_history"]); fig,ax=plt.subplots(figsize=(8,5)); ax.plot(h.epoch,h.validation_f1,marker='o',label="Validation F1"); ax.plot(h.epoch,h.validation_auroc,marker='o',label="Validation AUROC"); ax.set_xlabel("ConvLoRA epoch"); ax.legend(); fig.tight_layout(); fig.savefig(p["fig"]/"convlora_validation_history.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def consolidated_table(seg,context,natural,p):
    rows=[
        {"evaluation":"Segmentation test","metric":"Foreground Dice","value":seg["foreground_dice"]},
        {"evaluation":"Segmentation test","metric":"Foreground IoU","value":seg["foreground_iou"]},
    ]
    for label,m in (("Validation final",context["val_metrics"]),("Balanced locked test final",context["metrics"]),("Natural-distribution test final",natural)):
        for key in ("accuracy","precision","recall","specificity","f1","auroc","average_precision","brier","mcc"): rows.append({"evaluation":label,"metric":key,"value":m[key]})
    pd.DataFrame(rows).to_csv(p["result"]/"paper_aligned_final_results.csv",index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root",type=Path,default=Path("/content/drive/MyDrive/Sickle_Cell_OCT_Project"))
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-llm",action="store_true",help="Use deterministic constrained report instead of loading FLAN-T5-small.")
    parser.add_argument("--no-strict-paper-checks",action="store_true",help="Do not fail if reproduced metrics differ from the supplied notebook reference.")
    args=parser.parse_args(); seed_everything(); root=args.root.expanduser().resolve(); p=paths(root); device=torch.device(args.device); strict=not args.no_strict_paper_checks
    print(f"Root: {root}\nDevice: {device}\nFinal fixed threshold: {FINAL_THRESHOLD}")
    plot_histories(p)
    seg,_=evaluate_segmentation(p,device,strict)
    context=evaluate_classification(p,device,strict)
    natural,_=evaluate_natural(p,device,context["lora"],strict)
    evidence_figure(context,p)
    report=final_report(context,p,enable_llm=not args.skip_llm)
    consolidated_table(seg,context,natural,p)
    print("\n"+"="*80)
    print("FINAL PAPER-ALIGNED EVALUATION COMPLETE")
    print("="*80)
    print(f"Segmentation foreground Dice : {seg['foreground_dice']:.4f}")
    print(f"Locked-test accuracy         : {context['metrics']['accuracy']:.4f}")
    print(f"Locked-test F1               : {context['metrics']['f1']:.4f}")
    print(f"Locked-test AUROC            : {context['metrics']['auroc']:.4f}")
    print(f"Locked-test AP               : {context['metrics']['average_precision']:.4f}")
    print(f"Natural-test accuracy        : {natural['accuracy']:.4f}")
    print(f"Natural-test AUROC           : {natural['auroc']:.4f}")
    print(f"Report model                 : {report['model']} (threshold={FINAL_THRESHOLD:.4f})")
    print("Ablation study               : intentionally excluded")
    print("="*80)


if __name__=="__main__":
    main()
