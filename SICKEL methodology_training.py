

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from tqdm.auto import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise SystemExit("Install segmentation-models-pytorch first, e.g. pip install segmentation-models-pytorch") from exc

SEED = 42
NUM_CLASSES = 8
IMAGE_HEIGHT, IMAGE_WIDTH = 256, 576
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(obj, f, indent=2)


def project_paths(root: Path) -> Dict[str, Path]:
    p = {
        "seg_data": root / "01_data/processed/retinal_layer_segmentation",
        "splits": root / "01_data/splits",
        "meta": root / "01_data/metadata",
        "csat": root / "01_data/processed/csat_quality_controlled",
        "boxes": root / "01_data/processed/csat_box_targets",
        "seg_ckpt": root / "02_segmentation_pretraining/checkpoints",
        "seg_result": root / "02_segmentation_pretraining/results",
        "cls_ckpt": root / "03_disease_classification/checkpoints",
        "cls_result": root / "03_disease_classification/results",
        "fig": root / "04_figures",
        "config": root / "00_config",
    }
    for key in ("seg_ckpt","seg_result","cls_ckpt","cls_result","fig","config"): p[key].mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# Stage 1 — anatomy-guided retinal-layer segmentation
# =============================================================================

class RetinalLayerDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray, indices: Sequence[int]):
        self.images, self.masks = images, masks
        self.indices = np.asarray(indices, dtype=int)
        self.mean = IMAGENET_MEAN[:, None, None]
        self.std = IMAGENET_STD[:, None, None]

    def __len__(self) -> int: return len(self.indices)

    def __getitem__(self, idx: int):
        i = int(self.indices[idx])
        x = np.asarray(self.images[i], dtype=np.float32).copy() / 255.0
        x = np.stack([x, x, x], axis=0)
        x = (x - self.mean) / self.std
        y = np.asarray(self.masks[i], dtype=np.int64).copy()
        return torch.from_numpy(x).float(), torch.from_numpy(y).long(), torch.tensor(i)


def segmentation_loaders(paths: Dict[str, Path], batch_size: int = 4, workers: int = 2):
    images = np.load(paths["seg_data"] / "preprocessed_images.npy", mmap_mode="r")
    masks = np.load(paths["seg_data"] / "preprocessed_masks.npy", mmap_mode="r")
    split = np.load(paths["splits"] / "retinal_layer_split_indices.npz")
    datasets = {
        "train": RetinalLayerDataset(images, masks, split["train_indices"]),
        "validation": RetinalLayerDataset(images, masks, split["validation_indices"]),
        "test": RetinalLayerDataset(images, masks, split["test_indices"]),
    }
    g = torch.Generator().manual_seed(SEED)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, generator=g),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
    }
    return loaders


def build_segmentation_model() -> nn.Module:
    # Canonical trained architecture. No untrained/extra attention module is added.
    return smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES)


def multiclass_dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Multiclass Dice used by the actual training code, including encoded class 0."""
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(targets, num_classes=num_classes).permute(0,3,1,2).float()
    dims = (0,2,3)
    intersection = (probs * onehot).sum(dims)
    denominator = probs.sum(dims) + onehot.sum(dims)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor):
    ce = F.cross_entropy(logits, targets, weight=class_weights)
    dice = multiclass_dice_loss(logits, targets)
    return 0.60 * ce + 0.40 * dice, ce, dice


def seg_metric_state() -> Dict[str, Any]:
    return {"inter": np.zeros(NUM_CLASSES), "pred": np.zeros(NUM_CLASSES), "target": np.zeros(NUM_CLASSES), "correct": 0, "pixels": 0}


def seg_update(state: Dict[str, Any], pred: torch.Tensor, target: torch.Tensor) -> None:
    state["correct"] += int((pred == target).sum().item()); state["pixels"] += int(target.numel())
    for c in range(NUM_CLASSES):
        p, t = pred == c, target == c
        state["inter"][c] += (p & t).sum().item(); state["pred"][c] += p.sum().item(); state["target"][c] += t.sum().item()


def seg_metrics(state: Dict[str, Any]) -> Dict[str, float]:
    dice = (2*state["inter"]+1e-6)/(state["pred"]+state["target"]+1e-6)
    union = state["pred"]+state["target"]-state["inter"]
    iou = (state["inter"]+1e-6)/(union+1e-6)
    return {"foreground_dice": float(dice[1:].mean()), "foreground_iou": float(iou[1:].mean()), "pixel_accuracy": float(state["correct"]/max(state["pixels"],1))}


def run_seg_epoch(model, loader, optimizer, scaler, class_weights, device, training: bool):
    model.train(training); totals = {"loss":0.0,"ce":0.0,"dice":0.0,"n":0}; state = seg_metric_state()
    for x,y,_ in tqdm(loader, desc="Seg train" if training else "Seg validation", leave=False):
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
            logits = model(x); total, ce, dice = segmentation_loss(logits,y,class_weights)
        if training:
            scaler.scale(total).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
        b = x.size(0); totals["loss"] += total.item()*b; totals["ce"] += ce.item()*b; totals["dice"] += dice.item()*b; totals["n"] += b
        seg_update(state, logits.argmax(1), y)
    m = seg_metrics(state); n=max(totals["n"],1)
    return {"loss":totals["loss"]/n,"ce_loss":totals["ce"]/n,"dice_loss":totals["dice"]/n,**m}


def train_segmentation(paths: Dict[str, Path], device: torch.device) -> Path:
    loaders = segmentation_loaders(paths, batch_size=4)
    model = build_segmentation_model().to(device)
    weights = torch.tensor(np.load(paths["meta"] / "retinal_layer_class_weights.npy"), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    best = -np.inf; best_epoch = 0; patience = 0; history=[]
    best_path = paths["seg_ckpt"] / "segmentation_full_best.pt"
    last_path = paths["seg_ckpt"] / "segmentation_full_last.pt"
    encoder_path = paths["seg_ckpt"] / "retinal_layer_encoder.pt"
    for epoch in range(30):
        tr = run_seg_epoch(model, loaders["train"], optimizer, scaler, weights, device, True)
        va = run_seg_epoch(model, loaders["validation"], optimizer, scaler, weights, device, False)
        scheduler.step(va["foreground_dice"])
        row={"epoch":epoch+1,**{f"train_{k}":v for k,v in tr.items()},**{f"validation_{k}":v for k,v in va.items()},"lr":optimizer.param_groups[0]["lr"]}; history.append(row)
        state={"epoch":epoch+1,"model_state":model.state_dict(),"encoder_state":model.encoder.state_dict(),"optimizer_state":optimizer.state_dict(),"history":history,"validation":va}
        torch.save(state,last_path)
        if va["foreground_dice"] > best:
            best=va["foreground_dice"]; best_epoch=epoch+1; patience=0; torch.save(state,best_path)
        else: patience+=1
        print(f"Seg epoch {epoch+1:02d}/30 | train={tr['loss']:.4f} | val Dice={va['foreground_dice']:.4f} | best={best:.4f}")
        if patience >= 8: break
    checkpoint=torch.load(best_path,map_location="cpu",weights_only=False)
    torch.save({"encoder_state":checkpoint["encoder_state"],"best_epoch":checkpoint["epoch"],"source_checkpoint":str(best_path)},encoder_path)
    pd.DataFrame(history).to_csv(paths["seg_result"] / "segmentation_full_history.csv",index=False)
    dump_json({"max_epochs":30,"selected_epoch":best_epoch,"batch_size":4,"initial_lr":3e-4,"optimizer":"AdamW","loss":"0.60 weighted CE + 0.40 multiclass Dice","best_validation_foreground_dice":best}, paths["seg_result"] / "segmentation_full_summary.json")
    return encoder_path


# =============================================================================
# Stages 2–3 — anatomy transfer + coordinate-guided global-local Patch-MIL
# =============================================================================

class CSATPatchMILDataset(Dataset):
    def __init__(self, paths: Dict[str, Path], split: str, training: bool=False, patient_weight_mode: str="sqrt"):
        self.images=np.load(paths["csat"] / f"{split}_images_uint8.npy",mmap_mode="r")
        self.labels=np.load(paths["csat"] / f"{split}_labels_uint8.npy",mmap_mode="r")
        self.manifest=pd.read_csv(paths["csat"] / f"{split}_manifest.csv").sort_values("array_index").reset_index(drop=True)
        box_df=pd.read_csv(paths["boxes"] / f"{split}_box_coordinates.csv")
        box_df=box_df[box_df.label_id==1].copy()
        self.box_lookup={int(i):g[["x1","y1","x2","y2"]].to_numpy(np.float32) for i,g in box_df.groupby("array_index")}
        counts=self.manifest.patient_id.astype(str).value_counts()
        if patient_weight_mode=="inverse": raw=np.asarray([1.0/counts[str(p)] for p in self.manifest.patient_id],np.float32)
        else: raw=np.asarray([1.0/np.sqrt(counts[str(p)]) for p in self.manifest.patient_id],np.float32)
        self.patient_weights=raw/raw.mean(); self.training=training
        if not np.array_equal(self.manifest.scr_target.to_numpy(np.uint8),np.asarray(self.labels)): raise RuntimeError(f"{split} labels/manifests misaligned")

    def __len__(self): return len(self.labels)

    def __getitem__(self,index):
        image=np.asarray(self.images[index],dtype=np.float32).copy(); target=float(self.labels[index]); array_index=int(self.manifest.loc[index,"array_index"])
        boxes=self.box_lookup.get(array_index,np.empty((0,4),np.float32)).copy()
        if self.training and random.random()<0.5:
            image=np.ascontiguousarray(image[:,::-1])
            if len(boxes):
                x1,x2=boxes[:,0].copy(),boxes[:,2].copy(); boxes[:,0]=IMAGE_WIDTH-1-x2; boxes[:,2]=IMAGE_WIDTH-1-x1
        image=image/255.0; image=np.repeat(image[...,None],3,axis=-1); image=(image-IMAGENET_MEAN)/IMAGENET_STD; image=np.transpose(image,(2,0,1))
        return {"image":torch.from_numpy(image).float(),"target":torch.tensor(target,dtype=torch.float32),"boxes":torch.from_numpy(boxes).float(),
                "patient_weight":torch.tensor(self.patient_weights[index],dtype=torch.float32),"array_index":torch.tensor(array_index),
                "patient_id":str(self.manifest.loc[index,"patient_id"])}


def mil_collate(batch):
    return {"image":torch.stack([b["image"] for b in batch]),"target":torch.stack([b["target"] for b in batch]),"boxes":[b["boxes"] for b in batch],
            "patient_weight":torch.stack([b["patient_weight"] for b in batch]),"array_index":torch.stack([b["array_index"] for b in batch]),"patient_id":[b["patient_id"] for b in batch]}


def csat_loaders(paths: Dict[str, Path], batch_size: int=12, weight_mode: str="sqrt"):
    datasets={s:CSATPatchMILDataset(paths,s,training=(s=="train"),patient_weight_mode=weight_mode) for s in ("train","validation","test")}
    g=torch.Generator().manual_seed(SEED)
    return {"train":DataLoader(datasets["train"],batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True,collate_fn=mil_collate,generator=g),
            "validation":DataLoader(datasets["validation"],batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True,collate_fn=mil_collate),
            "test":DataLoader(datasets["test"],batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True,collate_fn=mil_collate)}


class CSATPatchMILV4(nn.Module):
    """ResNet-18 V4 classifier with the same parameter/state names as the notebook.

    The segmentation-trained SMP ResNet-18 encoder state is state-dict compatible
    with torchvision ResNet-18; the original notebook used this exact compatibility
    when recreating the final ConvLoRA model.
    """
    def __init__(self, encoder_checkpoint: Path | None=None, dropout=(0.40, 0.25)):
        super().__init__()
        self.encoder=resnet18(weights=None); self.encoder.fc=nn.Identity()
        if encoder_checkpoint is not None:
            data=torch.load(encoder_checkpoint,map_location="cpu",weights_only=False); self.encoder.load_state_dict(data["encoder_state"],strict=True)
        for p in self.encoder.parameters(): p.requires_grad=False
        def projection(cin): return nn.Sequential(nn.Conv2d(cin,128,1,bias=False),nn.BatchNorm2d(128),nn.GELU())
        self.stage8_projection=projection(128); self.stage16_projection=projection(256); self.stage32_projection=projection(512)
        self.feature_fusion=nn.Sequential(nn.Conv2d(384,192,3,padding=1,bias=False),nn.BatchNorm2d(192),nn.GELU(),nn.Conv2d(192,128,3,padding=1,bias=False),nn.BatchNorm2d(128),nn.GELU())
        self.patch_head=nn.Conv2d(128,1,1)
        self.classifier=nn.Sequential(nn.Linear(514,192),nn.LayerNorm(192),nn.GELU(),nn.Dropout(dropout[0]),nn.Linear(192,64),nn.GELU(),nn.Dropout(dropout[1]),nn.Linear(64,1))

    def train(self, mode=True):
        super().train(mode)
        # The notebook keeps the transferred encoder in eval mode so frozen BN
        # statistics are preserved; trainable layer-4 convolutions still receive gradients.
        self.encoder.eval()
        return self

    def forward(self,image):
        x=self.encoder.maxpool(self.encoder.relu(self.encoder.bn1(self.encoder.conv1(image))))
        x=self.encoder.layer1(x); stage8=self.encoder.layer2(x); stage16=self.encoder.layer3(stage8); stage32=self.encoder.layer4(stage16)
        f8=self.stage8_projection(stage8); f16=F.interpolate(self.stage16_projection(stage16),size=f8.shape[-2:],mode="bilinear",align_corners=False); f32=F.interpolate(self.stage32_projection(stage32),size=f8.shape[-2:],mode="bilinear",align_corners=False)
        fused=self.feature_fusion(torch.cat([f8,f16,f32],dim=1)); patch_logits=self.patch_head(fused); flat=patch_logits.flatten(1); top_k=max(1,int(flat.shape[1]*0.10))
        top=torch.topk(flat,top_k,dim=1).values.mean(dim=1,keepdim=True); mean=flat.mean(dim=1,keepdim=True); glob=F.adaptive_avg_pool2d(stage32,1).flatten(1)
        image_logits=self.classifier(torch.cat([glob,top,mean],dim=1)).squeeze(1)
        return {"image_logits":image_logits,"patch_logits":patch_logits.squeeze(1)}

def coordinate_guided_patch_loss(patch_logits, boxes, targets, inner_ratio=0.70, outer_expansion=0.10):
    b,h,w=patch_logits.shape
    yc=(torch.arange(h,device=patch_logits.device,dtype=torch.float32)+0.5)*IMAGE_HEIGHT/h
    xc=(torch.arange(w,device=patch_logits.device,dtype=torch.float32)+0.5)*IMAGE_WIDTH/w
    gy,gx=torch.meshgrid(yc,xc,indexing="ij"); losses=[]
    for i in range(b):
        logits=patch_logits[i]; target=targets[i]; sb=boxes[i].to(patch_logits.device,dtype=torch.float32)
        if target>0.5 and len(sb)>0:
            pos=torch.zeros((h,w),dtype=torch.bool,device=patch_logits.device); exclusion=torch.zeros_like(pos)
            for box in sb:
                x1,y1,x2,y2=box; bw=(x2-x1).clamp_min(1.0); bh=(y2-y1).clamp_min(1.0); cx=(x1+x2)/2; cy=(y1+y2)/2
                iw,ih=bw*inner_ratio,bh*inner_ratio
                pos |= (gx>=cx-iw/2)&(gx<=cx+iw/2)&(gy>=cy-ih/2)&(gy<=cy+ih/2)
                ex,ey=bw*outer_expansion,bh*outer_expansion
                ex1=torch.clamp(x1-ex,0,IMAGE_WIDTH-1); ex2=torch.clamp(x2+ex,0,IMAGE_WIDTH-1); ey1=torch.clamp(y1-ey,0,IMAGE_HEIGHT-1); ey2=torch.clamp(y2+ey,0,IMAGE_HEIGHT-1)
                exclusion |= (gx>=ex1)&(gx<=ex2)&(gy>=ey1)&(gy<=ey2)
            p=logits[pos]; n=logits[~exclusion]
            if p.numel()==0: continue
            pk=max(1,math.ceil(0.25*p.numel())); sp=torch.topk(p,k=pk).values; pos_loss=F.binary_cross_entropy_with_logits(sp,torch.ones_like(sp))
            if n.numel():
                nk=max(1,math.ceil(0.10*n.numel())); sn=torch.topk(n,k=nk).values; neg_loss=F.binary_cross_entropy_with_logits(sn,torch.zeros_like(sn)); ranking=F.relu(0.75-sp.mean()+sn.mean())
            else: neg_loss=logits.sum()*0; ranking=logits.sum()*0
            loss=pos_loss+0.50*neg_loss+0.25*ranking
        else:
            flat=logits.flatten(); nk=max(1,math.ceil(0.10*flat.numel())); hard=torch.topk(flat,k=nk).values; loss=F.binary_cross_entropy_with_logits(hard,torch.zeros_like(hard))
        losses.append(loss)
    return torch.stack(losses).mean() if losses else patch_logits.sum()*0


def local_weight_for_epoch(epoch: int) -> float:
    return 0.05 if epoch<2 else (0.15 if epoch<5 else 0.25)


def v4_loss(outputs,targets,boxes,patient_weights,local_weight: float):
    smoothed=targets*(1.0-0.03)+0.5*0.03
    cls=(F.binary_cross_entropy_with_logits(outputs["image_logits"],smoothed,reduction="none")*patient_weights).mean()
    regional=coordinate_guided_patch_loss(outputs["patch_logits"],boxes,targets)
    return cls+local_weight*regional,cls,regional


def best_threshold(y: np.ndarray,p: np.ndarray,low=0.10,high=0.90,steps=321):
    best_t,best_f=0.5,-1.0
    for t in np.linspace(low,high,steps):
        f=f1_score(y,p>=t,zero_division=0)
        if f>best_f: best_f,best_t=f,float(t)
    return best_t,best_f


def v4_validate(model,loader,device,tta: bool=False):
    model.eval(); ys=[]; ps=[]
    with torch.inference_mode():
        for batch in loader:
            x=batch["image"].to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                p1=torch.sigmoid(model(x)["image_logits"])
                if tta:
                    p2=torch.sigmoid(model(torch.flip(x,dims=[3]))["image_logits"]); p=(p1+p2)/2
                else:
                    p=p1
            ys.extend(batch["target"].numpy()); ps.extend(p.cpu().numpy())
    y,p=np.asarray(ys,int),np.asarray(ps,float); threshold,f1=best_threshold(y,p); auc=roc_auc_score(y,p); ap=average_precision_score(y,p)
    return {"threshold":threshold,"f1":float(f1),"auroc":float(auc),"average_precision":float(ap),"targets":y,"probabilities":p}


def configure_v4_optimizer(model: CSATPatchMILV4, phase: str):
    for p in model.encoder.parameters(): p.requires_grad=False
    head_params=[p for n,p in model.named_parameters() if not n.startswith("encoder.")]
    if phase=="head_only":
        opt=torch.optim.AdamW(head_params,lr=2e-4,weight_decay=2e-4)
    else:
        for p in model.encoder.layer4.parameters(): p.requires_grad=True
        opt=torch.optim.AdamW([{"params":head_params,"lr":6e-5},{"params":model.encoder.layer4.parameters(),"lr":8e-6}],weight_decay=2e-4)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=3,min_lr=1e-6)
    return opt,sch


def update_ema(ema: Dict[str,torch.Tensor],state: Dict[str,torch.Tensor],decay=0.995):
    with torch.no_grad():
        for k,v in state.items():
            if k not in ema: ema[k]=v.detach().clone()
            elif torch.is_floating_point(v): ema[k].mul_(decay).add_(v.detach(),alpha=1-decay)
            else: ema[k].copy_(v)


def train_v4(paths: Dict[str, Path],device: torch.device,encoder_checkpoint: Path|None=None) -> Path:
    encoder_checkpoint=encoder_checkpoint or (paths["seg_ckpt"] / "retinal_layer_encoder.pt")
    loaders=csat_loaders(paths,batch_size=12,weight_mode="sqrt")
    model=CSATPatchMILV4(encoder_checkpoint).to(device); phase="head_only"; optimizer,scheduler=configure_v4_optimizer(model,phase)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda"); ema={k:v.detach().clone() for k,v in model.state_dict().items()}; history=[]
    best_score=-np.inf; best_path=paths["cls_ckpt"] / "csat_patch_mil_v4_best.pt"; last_path=paths["cls_ckpt"] / "csat_patch_mil_v4_last.pt"; patience=0
    for epoch in range(35):
        new_phase="head_only" if epoch<5 else "finetune"
        if new_phase!=phase: phase=new_phase; optimizer,scheduler=configure_v4_optimizer(model,phase)
        model.train(); sums=np.zeros(4,float)
        for batch in tqdm(loaders["train"],desc=f"V4 epoch {epoch+1}",leave=False):
            x=batch["image"].to(device); y=batch["target"].to(device); pw=batch["patient_weight"].to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                out=model(x); total,cls,reg=v4_loss(out,y,batch["boxes"],pw,local_weight_for_epoch(epoch))
            scaler.scale(total).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); scaler.step(optimizer); scaler.update(); update_ema(ema,model.state_dict(),0.995)
            sums += [total.item()*len(y),cls.item()*len(y),reg.item()*len(y),len(y)]
        # Validate EMA state, then restore current state.
        current=copy.deepcopy(model.state_dict()); model.load_state_dict(ema,strict=True); val=v4_validate(model,loaders["validation"],device,tta=False); model.load_state_dict(current,strict=True)
        score=0.65*val["f1"]+0.35*val["auroc"]; scheduler.step(score)
        row={"epoch":epoch+1,"phase":phase,"train_total_loss":sums[0]/sums[3],"train_classification_loss":sums[1]/sums[3],"train_regional_loss":sums[2]/sums[3],"local_weight":local_weight_for_epoch(epoch),"validation_f1":val["f1"],"validation_auroc":val["auroc"],"validation_average_precision":val["average_precision"],"validation_threshold":val["threshold"],"selection_score":score}; history.append(row)
        state={"epoch":epoch+1,"model_state":model.state_dict(),"ema_state":copy.deepcopy(ema),"optimizer_state":optimizer.state_dict(),"history":history,"validation":{k:v for k,v in val.items() if k not in ("targets","probabilities")}}
        torch.save(state,last_path)
        if score>best_score:
            best_score=score; patience=0; torch.save(state,best_path)
        else: patience+=1
        print(f"V4 epoch {epoch+1:02d}/35 | phase={phase} | val F1={val['f1']:.4f} | AUROC={val['auroc']:.4f} | score={score:.4f}")
        if patience>=9: break
    pd.DataFrame(history).to_csv(paths["cls_result"] / "csat_patch_mil_v4_history.csv",index=False)
    best=torch.load(best_path,map_location="cpu",weights_only=False)
    dump_json({"architecture":"Transferred ResNet-18 + multiscale global-local coordinate-guided Patch-MIL","batch_size":12,"maximum_epochs":35,"selected_epoch":int(best["epoch"]),"initial_learning_rate":2e-4,"selection":"0.65 validation F1 + 0.35 validation AUROC","regional_supervision":"central 70% of genuine SCR boxes; boundary/nearby ignored; strongest 25% positive patches; hard 10% background","test_time_augmentation":"mean(original,horizontal reflection)"}, paths["cls_result"] / "csat_patch_mil_v4_summary.json")
    return best_path


# =============================================================================
# Stage 4 — ConvLoRA refinement
# =============================================================================

class LoRAConv2d(nn.Module):
    def __init__(self,base: nn.Conv2d,rank: int=8,alpha: float=16.0):
        super().__init__(); self.base=base; self.rank=rank; self.alpha=alpha; self.scale=alpha/rank
        for p in self.base.parameters(): p.requires_grad=False
        self.lora_A=nn.Conv2d(base.in_channels,rank,kernel_size=1,bias=False)
        self.lora_B=nn.Conv2d(rank,base.out_channels,kernel_size=base.kernel_size,stride=base.stride,padding=base.padding,dilation=base.dilation,bias=False)
        nn.init.kaiming_normal_(self.lora_A.weight); nn.init.zeros_(self.lora_B.weight)
    def forward(self,x): return self.base(x)+self.scale*self.lora_B(self.lora_A(x))


def get_module(model: nn.Module,path: str):
    cur=model
    for part in path.split("."): cur=cur[int(part)] if part.isdigit() else getattr(cur,part)
    return cur


def set_module(model: nn.Module,path: str,module: nn.Module):
    parts=path.split("."); parent=model
    for part in parts[:-1]: parent=parent[int(part)] if part.isdigit() else getattr(parent,part)
    last=parts[-1]
    if last.isdigit(): parent[int(last)]=module
    else: setattr(parent,last,module)


def inject_convlora(model: CSATPatchMILV4,rank=8,alpha=16.0) -> List[str]:
    targets=["encoder.layer4.0.conv1","encoder.layer4.0.conv2","encoder.layer4.0.downsample.0","encoder.layer4.1.conv1","encoder.layer4.1.conv2"]
    for p in model.parameters(): p.requires_grad=False
    for name in targets:
        base=get_module(model,name)
        if not isinstance(base,nn.Conv2d): raise TypeError(f"{name} is not Conv2d")
        set_module(model,name,LoRAConv2d(base,rank,alpha))
    return targets


def load_v4_for_lora(paths: Dict[str, Path],device: torch.device):
    base_path=paths["cls_ckpt"] / "csat_patch_mil_v4_best.pt"; checkpoint=torch.load(base_path,map_location="cpu",weights_only=False)
    model=CSATPatchMILV4(None, dropout=(0.30,0.20)); model.load_state_dict(checkpoint["ema_state"],strict=True); model.eval()
    # Check zero-initialized LoRA parity.
    reference=copy.deepcopy(model).eval(); targets=inject_convlora(model,rank=8,alpha=16.0); model.eval()
    x=torch.randn(2,3,IMAGE_HEIGHT,IMAGE_WIDTH)
    with torch.inference_mode(): before=reference(x)["image_logits"]; after=model(x)["image_logits"]
    diff=float((before-after).abs().max())
    if diff>1e-6: raise RuntimeError(f"Zero-init ConvLoRA changed V4 output before training: max diff={diff}")
    model.to(device); reference.to(device)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); total=sum(p.numel() for p in model.parameters())
    if trainable!=167_936: raise RuntimeError(f"Expected 167,936 LoRA parameters; found {trainable:,}")
    return model,reference,checkpoint,targets,total,trainable


def plain_split_loader(paths: Dict[str, Path],split: str,batch_size=16,training=False,weight_mode="inverse"):
    ds=CSATPatchMILDataset(paths,split,training=training,patient_weight_mode=weight_mode)
    return DataLoader(ds,batch_size=batch_size,shuffle=training,num_workers=2,pin_memory=True,collate_fn=mil_collate,generator=torch.Generator().manual_seed(SEED) if training else None)


def lora_validation(model,loader,device):
    model.eval(); ys=[]; ps=[]
    with torch.inference_mode():
        for batch in loader:
            x=batch["image"].to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                p=(torch.sigmoid(model(x)["image_logits"])+torch.sigmoid(model(torch.flip(x,[3]))["image_logits"]))/2
            ys.extend(batch["target"].numpy()); ps.extend(p.cpu().numpy())
    y,p=np.asarray(ys,int),np.asarray(ps,float); t,f=best_threshold(y,p,0.05,0.95,361)
    return {"threshold":t,"f1":float(f),"auroc":float(roc_auc_score(y,p)),"average_precision":float(average_precision_score(y,p))}


def train_convlora(paths: Dict[str, Path],device: torch.device) -> Path:
    model,teacher,base_checkpoint,targets,total_params,trainable_params=load_v4_for_lora(paths,device)
    teacher.eval(); [p.requires_grad_(False) for p in teacher.parameters()]
    train_loader=plain_split_loader(paths,"train",16,True,"inverse"); val_loader=plain_split_loader(paths,"validation",16,False,"inverse")
    baseline=lora_validation(model,val_loader,device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5,eta_min=1e-6); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best_score=-np.inf; history=[]; patience=0; best_path=paths["cls_ckpt"] / "csat_patch_mil_v4_lora_refined_best.pt"
    def b_norm(): return float(sum((m.lora_B.weight.detach()**2).sum().item() for m in model.modules() if isinstance(m,LoRAConv2d))**0.5)
    for epoch in range(5):
        # Important: keep frozen BN and dropout deterministic during LoRA-only adaptation.
        model.eval(); running=0.0; n=0
        for batch in tqdm(train_loader,desc=f"ConvLoRA epoch {epoch+1}",leave=False):
            x=batch["image"].to(device); y=batch["target"].to(device); pw=batch["patient_weight"].to(device); optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): tout=teacher(x)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                sout=model(x)
                pos=(len(y)-y.sum())/torch.clamp(y.sum(),min=1.0)
                bce=F.binary_cross_entropy_with_logits(sout["image_logits"],y,reduction="none",pos_weight=pos)
                cls=(bce*pw).mean(); logit_pres=F.smooth_l1_loss(sout["image_logits"],tout["image_logits"]); patch_pres=F.smooth_l1_loss(sout["patch_logits"],tout["patch_logits"])
                loss=0.45*cls+0.40*logit_pres+0.15*patch_pres
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); scaler.step(optimizer); scaler.update(); running+=loss.item()*len(y); n+=len(y)
        scheduler.step(); val=lora_validation(model,val_loader,device); norm=b_norm()
        preserved=(val["f1"]>=baseline["f1"]-0.002 and val["auroc"]>=baseline["auroc"]-0.002 and norm>0)
        score=val["f1"]+0.25*val["auroc"] if preserved else -np.inf
        row={"epoch":epoch+1,"train_loss":running/max(n,1),"validation_f1":val["f1"],"validation_auroc":val["auroc"],"validation_average_precision":val["average_precision"],"threshold":val["threshold"],"lora_b_norm":norm,"performance_preserved":bool(preserved),"lr":optimizer.param_groups[0]["lr"]}; history.append(row)
        print(f"LoRA epoch {epoch+1}/5 | val F1={val['f1']:.4f} | AUROC={val['auroc']:.4f} | threshold={val['threshold']:.4f} | preserved={preserved}")
        if score>best_score:
            best_score=score; patience=0; torch.save({"epoch":epoch+1,"model_state":model.state_dict(),"validation_metrics":val,"history":history,"rank":8,"alpha":16,"target_modules":targets,"trainable_parameters":trainable_params,"total_parameters":total_params,"base_checkpoint":str(paths["cls_ckpt"] / "csat_patch_mil_v4_best.pt")},best_path)
        else: patience+=1
        if patience>=3: break
    pd.DataFrame(history).to_csv(paths["cls_result"] / "csat_patch_mil_v4_lora_history.csv",index=False)
    best=torch.load(best_path,map_location="cpu",weights_only=False)
    dump_json({"rank":8,"alpha":16,"target_convolutions":targets,"trainable_parameters":trainable_params,"final_model_parameters":total_params,"trainable_percentage":100*trainable_params/total_params,"maximum_epochs":5,"selected_epoch":best["epoch"],"selected_threshold":best["validation_metrics"]["threshold"],"loss":"0.45 patient-weighted BCE + 0.40 SmoothL1(image logits) + 0.15 SmoothL1(patch evidence)","optimizer":"AdamW","initial_lr":2e-5,"scheduler":"CosineAnnealingLR"}, paths["cls_result"] / "csat_patch_mil_v4_lora_training_summary.json")
    return best_path


def verify_canonical_counts(paths: Dict[str, Path]) -> None:
    # These checks are metadata/model-structure checks only; performance is evaluated in file 03.
    seg=np.load(paths["seg_data"] / "preprocessed_images.npy",mmap_mode="r")
    if seg.shape!=(220,224,512): raise RuntimeError(f"Unexpected segmentation tensor {seg.shape}")
    expected={"train":3758,"validation":778,"test":1112}
    for split,n in expected.items():
        labels=np.load(paths["csat"] / f"{split}_labels_uint8.npy")
        if len(labels)!=n: raise RuntimeError(f"{split}: expected {n} scans, found {len(labels)}")


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root",type=Path,default=Path("/content/drive/MyDrive/Sickle_Cell_OCT_Project"))
    parser.add_argument("--stage",choices=["segmentation","v4","lora","all"],default="all")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args(); seed_everything(); root=args.root.expanduser().resolve(); paths=project_paths(root); device=torch.device(args.device)
    verify_canonical_counts(paths)
    print(f"Device: {device} | root: {root}")
    if args.stage in ("segmentation","all"): train_segmentation(paths,device)
    if args.stage in ("v4","all"): train_v4(paths,device)
    if args.stage in ("lora","all"): train_convlora(paths,device)
    print("\nMethodology training pipeline complete. Run 03_experiments_evaluation.py next.")


if __name__=="__main__":
    main()
