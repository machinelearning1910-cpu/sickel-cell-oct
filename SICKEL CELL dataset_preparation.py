


from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

SEED = 42
SEG_HANDLE = "smasifulislamsaky/retinal-layer-segmentation-dataset"
CSAT_GDRIVE_ID = "1GSaanysnf2dYD6pqomuUOGOHfxyfO54W"
CSAT_OFFICIAL_REPO = "https://github.com/VimsLab/CSAT"
SEG_EXPECTED = {"images": 220, "height": 216, "width": 500, "classes": 8}
CSAT_EXPECTED = {
    "records": 12623,
    "patients": 102,
    "split_patients": {"train": 71, "validation": 15, "test": 16},
    "split_records": {"train": 9177, "validation": 1679, "test": 1767},
    "balanced_final": {"train": 3758, "validation": 778, "test": 1112},
    "natural_final": 1764,
    "natural_scr": 557,
    "natural_non_scr": 1207,
}


def install_dependencies() -> None:
    packages = [
        "kagglehub",
        "gdown",
        "opencv-python-headless",
        "scikit-learn",
        "matplotlib",
        "tqdm",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *packages])


def imports():
    global cv2, plt, torch, tqdm, StratifiedShuffleSplit, kagglehub, gdown
    import cv2  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import torch  # type: ignore
    from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore
    from tqdm.auto import tqdm  # type: ignore
    import kagglehub  # type: ignore
    import gdown  # type: ignore


def ensure_dirs(root: Path) -> Dict[str, Path]:
    paths = {
        "raw_seg": root / "01_data/raw/retinal_layer_segmentation",
        "raw_csat": root / "01_data/raw/csat_dataset/original_zip",
        "csat_audit": root / "01_data/raw/csat_dataset/metadata/full_audit",
        "proc_seg": root / "01_data/processed/retinal_layer_segmentation",
        "proc_csat": root / "01_data/processed/csat_quality_controlled",
        "box_targets": root / "01_data/processed/csat_box_targets",
        "natural": root / "01_data/processed/csat_natural_test",
        "splits": root / "01_data/splits",
        "meta": root / "01_data/metadata",
        "fig": root / "04_figures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def download_datasets(paths: Dict[str, Path], skip_existing: bool = True) -> None:
    seg_image = paths["raw_seg"] / "resized_images.npy"
    seg_mask = paths["raw_seg"] / "resized_labeledimages.npy"
    if not (skip_existing and seg_image.exists() and seg_mask.exists()):
        print("\n[1/2] Downloading retinal-layer segmentation dataset from Kaggle...")
        src = Path(kagglehub.dataset_download(SEG_HANDLE))
        shutil.copytree(src, paths["raw_seg"], dirs_exist_ok=True)
    else:
        print("\n[1/2] Retinal-layer dataset already present; skipping download.")

    zip_path = paths["raw_csat"] / "pickle.zip"
    if not (skip_existing and zip_path.exists()):
        print("[2/2] Downloading official CSAT pickle.zip from the VimsLab release...")
        gdown.download(id=CSAT_GDRIVE_ID, output=str(zip_path), quiet=False, fuzzy=True)
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            raise RuntimeError("CSAT download failed or produced an empty file.")
    else:
        print("[2/2] CSAT pickle.zip already present; skipping download.")


def find_seg_arrays(raw_seg: Path) -> Tuple[Path, Path]:
    image_path = raw_seg / "resized_images.npy"
    mask_path = raw_seg / "resized_labeledimages.npy"
    if image_path.exists() and mask_path.exists():
        return image_path, mask_path
    npys = list(raw_seg.rglob("*.npy"))
    images = [p for p in npys if "image" in p.name.lower() and "label" not in p.name.lower() and "mask" not in p.name.lower()]
    masks = [p for p in npys if "label" in p.name.lower() or "mask" in p.name.lower()]
    if not images or not masks:
        raise FileNotFoundError("Could not locate retinal-layer image/mask NumPy arrays.")
    return images[0], masks[0]


def audit_and_prepare_segmentation(paths: Dict[str, Path], strict: bool) -> None:
    image_path, mask_path = find_seg_arrays(paths["raw_seg"])
    raw_images = np.load(image_path, mmap_mode="r")
    raw_masks = np.load(mask_path, mmap_mode="r")
    if raw_images.shape != raw_masks.shape:
        raise ValueError(f"Segmentation image/mask shape mismatch: {raw_images.shape} vs {raw_masks.shape}")

    classes = sorted(np.unique(raw_masks).astype(int).tolist())
    audit = {
        "dataset_handle": SEG_HANDLE,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "image_shape": list(raw_images.shape),
        "mask_shape": list(raw_masks.shape),
        "image_dtype": str(raw_images.dtype),
        "mask_dtype": str(raw_masks.dtype),
        "classes": classes,
        "image_min": float(np.min(raw_images)),
        "image_max": float(np.max(raw_images)),
    }
    json_dump(audit, paths["meta"] / "retinal_layer_dataset_audit.json")

    if strict:
        assert tuple(raw_images.shape) == (220, 216, 500), raw_images.shape
        assert classes == list(range(8)), classes

    # Paper/notebook preprocessing: clip -> uint8 -> very mild 3x3 Gaussian -> zero-pad.
    n, h, w = raw_images.shape
    target_h, target_w = 224, 512
    pt = (target_h - h) // 2
    pb = target_h - h - pt
    pl = (target_w - w) // 2
    pr = target_w - w - pl
    out_image = paths["proc_seg"] / "preprocessed_images.npy"
    out_mask = paths["proc_seg"] / "preprocessed_masks.npy"
    images_mm = np.lib.format.open_memmap(out_image, mode="w+", dtype=np.uint8, shape=(n, target_h, target_w))
    masks_mm = np.lib.format.open_memmap(out_mask, mode="w+", dtype=np.uint8, shape=(n, target_h, target_w))

    for i in tqdm(range(n), desc="Retinal-layer preprocessing"):
        img = np.clip(np.asarray(raw_images[i], dtype=np.float32), 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0.5, 0.5)
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
        msk = np.rint(np.asarray(raw_masks[i])).astype(np.uint8)
        msk = cv2.copyMakeBorder(msk, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
        images_mm[i], masks_mm[i] = img, msk
    images_mm.flush(); masks_mm.flush(); del images_mm, masks_mm

    proc_masks = np.load(out_mask, mmap_mode="r")
    restored = proc_masks[:, pt:pt+h, pl:pl+w]
    changed = int(np.sum(restored.astype(np.int16) != np.rint(raw_masks).astype(np.int16)))
    if changed:
        raise RuntimeError(f"Mask labels changed inside the original field ({changed} pixels).")

    # Fixed 70/15/15 split exactly as the notebook.
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(n)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    train_idx = np.sort(shuffled[:n_train])
    val_idx = np.sort(shuffled[n_train:n_train+n_val])
    test_idx = np.sort(shuffled[n_train+n_val:])
    np.savez(paths["splits"] / "retinal_layer_split_indices.npz",
             train_indices=train_idx, validation_indices=val_idx, test_indices=test_idx)
    pd.DataFrame(
        [(int(i), split) for split, ids in (("train", train_idx), ("validation", val_idx), ("test", test_idx)) for i in ids],
        columns=["sample_index", "split"],
    ).to_csv(paths["splits"] / "retinal_layer_splits.csv", index=False)

    # Training-only class weights: inverse sqrt frequency -> mean 1 -> clip 0.25..4.
    counts = np.zeros(8, dtype=np.int64)
    for i in train_idx:
        counts += np.bincount(np.asarray(proc_masks[i], dtype=np.uint8).ravel(), minlength=8)
    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(freq + 1e-8)
    weights = np.clip(weights / weights.mean(), 0.25, 4.0).astype(np.float32)
    np.save(paths["meta"] / "retinal_layer_class_weights.npy", weights)

    json_dump({
        "random_seed": SEED,
        "original_shape": list(raw_images.shape),
        "processed_shape": [n, target_h, target_w],
        "padding": {"top": pt, "bottom": pb, "left": pl, "right": pr},
        "preprocessing": ["clip 0-255", "uint8", "Gaussian 3x3 sigma 0.5 (image only)", "zero-pad to 224x512"],
        "mask_interpolation": False,
        "mask_changed_pixels": changed,
        "split_counts": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "class_weights": weights.tolist(),
    }, paths["meta"] / "retinal_layer_preparation_summary.json")

    # Samples: original / processed / mask.
    proc_images = np.load(out_image, mmap_mode="r")
    sample_ids = [0, 73, 146, 219] if n >= 220 else np.linspace(0, n-1, min(4, n), dtype=int).tolist()
    fig, ax = plt.subplots(len(sample_ids), 3, figsize=(14, 3 * len(sample_ids)))
    ax = np.atleast_2d(ax)
    for r, i in enumerate(sample_ids):
        ax[r, 0].imshow(raw_images[i], cmap="gray", vmin=0, vmax=255); ax[r, 0].set_title(f"Original OCT {i}")
        ax[r, 1].imshow(proc_images[i], cmap="gray", vmin=0, vmax=255); ax[r, 1].set_title("Conservative input")
        ax[r, 2].imshow(proc_masks[i], cmap="nipy_spectral", vmin=0, vmax=7); ax[r, 2].set_title("8-class mask")
        for c in range(3): ax[r, c].axis("off")
    fig.tight_layout(); fig.savefig(paths["fig"] / "retinal_layer_dataset_samples.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    if strict:
        assert (len(train_idx), len(val_idx), len(test_idx)) == (154, 33, 33)
    print(f"Segmentation dataset ready: {n} images; split {len(train_idx)}/{len(val_idx)}/{len(test_idx)}.")


def parse_csat_name(name: str) -> Tuple[str, str, str]:
    """Replicate the official CSAT naming convention.

    Official CSAT utility defines eye id as filename through the first L/R and
    patient id as the token before the first underscore. We use patient id (not
    eye id) for leakage-free splitting.
    """
    stem = Path(str(name)).stem
    patient_id = stem.split("_")[0].strip()
    eye = ""
    lr_pos = stem.find("L")
    if lr_pos < 0:
        lr_pos = stem.find("R")
    if lr_pos >= 0:
        eye = stem[lr_pos]
    scan_number = ""
    if lr_pos >= 0:
        suffix = stem[lr_pos + 1:].lstrip("_")
        scan_number = suffix.split("_")[0] if suffix else ""
    return patient_id, eye, scan_number


def prepare_labels(value: Any) -> np.ndarray:
    arr = to_numpy(value).reshape(-1)
    if arr.size == 0:
        return np.empty(0, dtype=int)
    return np.rint(arr).astype(int)


def prepare_boxes(value: Any) -> np.ndarray:
    arr = to_numpy(value)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    arr = np.squeeze(arr).astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return np.empty((0, 4), dtype=np.float32)
    return arr[:, :4]


def audit_csat_zip(paths: Dict[str, Path], strict: bool) -> pd.DataFrame:
    zip_path = paths["raw_csat"] / "pickle.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    rows: List[Dict[str, Any]] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [m for m in archive.namelist() if m.lower().endswith((".pkl", ".pickle"))]
        for member in tqdm(members, desc="Auditing CSAT pickle records"):
            row: Dict[str, Any] = {"archive_member": member}
            try:
                with archive.open(member) as f:
                    record = pickle.load(f)
                if not isinstance(record, dict):
                    raise TypeError("record is not a dictionary")
                image_raw = to_numpy(record.get("img"))
                image_nan = bool(np.isnan(image_raw).any()) if np.issubdtype(image_raw.dtype, np.number) else False
                image_inf = bool(np.isinf(image_raw).any()) if np.issubdtype(image_raw.dtype, np.number) else False
                stored_name = str(record.get("name", Path(member).stem))
                patient_id, eye, scan_number = parse_csat_name(stored_name)
                labels = prepare_labels(record.get("label", []))
                boxes = prepare_boxes(record.get("box", []))
                invalid_label = bool(np.any(~np.isin(labels, [0, 1, 2])))
                n_pairs = min(len(labels), len(boxes))
                invalid_boxes = 0
                for b in boxes[:n_pairs]:
                    if (not np.isfinite(b).all()) or b[2] <= 0 or b[3] <= 0:
                        invalid_boxes += 1
                c = Counter(labels.tolist())
                if c.get(1, 0) > 0:
                    category = "SCR"
                elif c.get(0, 0) > 0:
                    category = "Fovea-only"
                else:
                    category = "Negative"
                row.update({
                    "stored_name": stored_name,
                    "patient_id": patient_id,
                    "eye": eye,
                    "scan_number": scan_number,
                    "image_category": category,
                    "labels": json.dumps(labels.tolist()),
                    "label_0_fovea_count": int(c.get(0, 0)),
                    "label_1_scr_count": int(c.get(1, 0)),
                    "label_2_negative_count": int(c.get(2, 0)),
                    "boxes": json.dumps(boxes.tolist()),
                    "invalid_object_box_count": int(invalid_boxes),
                    "image_shape": json.dumps(list(image_raw.shape)),
                    "image_dtype": str(image_raw.dtype),
                    "image_has_nan": image_nan,
                    "image_has_inf": image_inf,
                    "has_invalid_label": invalid_label,
                    "read_error": "",
                })
            except Exception as exc:
                row.update({"stored_name": "", "patient_id": "", "eye": "", "scan_number": "",
                            "image_category": "", "labels": "[]", "label_0_fovea_count": 0,
                            "label_1_scr_count": 0, "label_2_negative_count": 0, "boxes": "[]",
                            "invalid_object_box_count": 0, "image_shape": "[]", "image_dtype": "",
                            "image_has_nan": False, "image_has_inf": False, "has_invalid_label": True,
                            "read_error": str(exc)})
            rows.append(row)
    audit = pd.DataFrame(rows)
    audit_path = paths["csat_audit"] / "csat_record_audit.csv"
    audit.to_csv(audit_path, index=False)
    valid = audit[(audit["read_error"] == "") & (~audit["image_has_nan"].astype(bool)) & (~audit["image_has_inf"].astype(bool)) & (~audit["has_invalid_label"].astype(bool))].copy()
    if strict:
        assert len(valid) == CSAT_EXPECTED["records"], f"Expected 12,623 valid records; found {len(valid)}"
        assert valid["patient_id"].nunique() == CSAT_EXPECTED["patients"], f"Expected 102 patients; found {valid['patient_id'].nunique()}"
    json_dump({
        "official_repository": CSAT_OFFICIAL_REPO,
        "google_drive_file_id": CSAT_GDRIVE_ID,
        "archive": str(zip_path),
        "records_audited": len(audit),
        "valid_records": len(valid),
        "unique_patients": int(valid["patient_id"].nunique()),
        "annotation_classes": {"0": "Fovea", "1": "SCR", "2": "Negative"},
    }, paths["meta"] / "csat_dataset_audit_summary.json")
    return audit


def make_patient_splits(audit: pd.DataFrame, paths: Dict[str, Path], strict: bool) -> pd.DataFrame:
    records = audit.copy()
    records["patient_id"] = records["patient_id"].astype(str).str.strip()
    valid = records[(records["read_error"] == "") & (records["patient_id"] != "") & (records["patient_id"].str.lower() != "nan") &
                    (~records["image_has_nan"].astype(bool)) & (~records["image_has_inf"].astype(bool)) & (~records["has_invalid_label"].astype(bool))].copy()
    valid["scr_target"] = (valid["label_1_scr_count"] > 0).astype(np.uint8)
    valid["target_name"] = valid["scr_target"].map({0: "Non-SCR", 1: "SCR"})
    patients = valid.groupby("patient_id").agg(total_images=("archive_member", "count"), scr_images=("scr_target", "sum"), eyes=("eye", "nunique")).reset_index()
    patients["patient_has_scr"] = (patients["scr_images"] > 0).astype(np.uint8)

    pids = patients["patient_id"].to_numpy(); labels = patients["patient_has_scr"].to_numpy()
    first = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr_pos, tmp_pos = next(first.split(pids, labels))
    tmp_ids, tmp_labels = pids[tmp_pos], labels[tmp_pos]
    second = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    va_pos, te_pos = next(second.split(tmp_ids, tmp_labels))
    split_map = {str(p): "train" for p in pids[tr_pos]}
    split_map.update({str(p): "validation" for p in tmp_ids[va_pos]})
    split_map.update({str(p): "test" for p in tmp_ids[te_pos]})
    valid["split"] = valid["patient_id"].map(split_map)
    patients["split"] = patients["patient_id"].map(split_map)
    if valid["split"].isna().any():
        raise RuntimeError("Some records were not assigned to a patient split.")
    sets = {s: set(valid.loc[valid["split"] == s, "patient_id"]) for s in ("train", "validation", "test")}
    assert sets["train"].isdisjoint(sets["validation"]) and sets["train"].isdisjoint(sets["test"]) and sets["validation"].isdisjoint(sets["test"])
    valid.to_csv(paths["splits"] / "csat_record_splits.csv", index=False)
    patients.to_csv(paths["splits"] / "csat_patient_splits.csv", index=False)
    for s in ("validation", "test"):
        valid[valid["split"] == s].to_csv(paths["splits"] / f"csat_{s}_full.csv", index=False)

    if strict:
        for s, n in CSAT_EXPECTED["split_patients"].items():
            assert patients[patients["split"] == s]["patient_id"].nunique() == n
        for s, n in CSAT_EXPECTED["split_records"].items():
            assert len(valid[valid["split"] == s]) == n, (s, len(valid[valid["split"] == s]))
    return valid


def balanced_subset(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    pos = df[df.scr_target == 1].copy(); neg = df[df.scr_target == 0].copy()
    n = min(len(pos), len(neg))
    if "image_category" in neg.columns:
        counts = neg["image_category"].value_counts(); props = counts / counts.sum(); raw = props * n
        quotas = np.floor(raw).astype(int); remainder = n - int(quotas.sum())
        for cat in (raw - quotas).sort_values(ascending=False).index[:remainder]: quotas.loc[cat] += 1
        parts = [neg[neg.image_category == cat].sample(n=min(int(q), len(neg[neg.image_category == cat])), random_state=seed) for cat, q in quotas.items() if q > 0]
        sampled = pd.concat(parts, ignore_index=True)
        if len(sampled) < n:
            missing = neg[~neg.archive_member.isin(sampled.archive_member)].sample(n=n-len(sampled), random_state=seed)
            sampled = pd.concat([sampled, missing], ignore_index=True)
    else:
        sampled = neg.sample(n=n, random_state=seed)
    return pd.concat([pos.sample(n=n, random_state=seed), sampled.sample(n=n, random_state=seed)], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def create_balanced_manifests(records: pd.DataFrame, paths: Dict[str, Path], strict: bool) -> Dict[str, pd.DataFrame]:
    out = {}
    names = {"train": "csat_balanced_train.csv", "validation": "csat_validation_balanced.csv", "test": "csat_test_balanced.csv"}
    for split in names:
        b = balanced_subset(records[records.split == split].copy())
        b.to_csv(paths["splits"] / names[split], index=False)
        out[split] = b
    # Full manifests are already required for the natural distribution protocol.
    records[records.split == "validation"].to_csv(paths["splits"] / "csat_validation_full.csv", index=False)
    records[records.split == "test"].to_csv(paths["splits"] / "csat_test_full.csv", index=False)
    if strict:
        # Before strict image QC, notebook counts are 3780 / 782 / 1116.
        assert len(out["train"]) == 3780
        assert len(out["validation"]) == 782
        assert len(out["test"]) == 1116
    return out


def model_image(value: Any, height: int = 256, width: int = 576, edge_columns: int = 12) -> np.ndarray:
    image = np.squeeze(to_numpy(value))
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if image.size and image.max() <= 1.0: image *= 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3: image = np.median(image[..., :3], axis=-1).astype(np.uint8)
    if image.shape != (height, width):
        raise ValueError(f"Unexpected CSAT image shape {image.shape}; expected {(height, width)}. No resizing is permitted.")
    image = image.copy(); image[:, :edge_columns] = 0; image[:, -edge_columns:] = 0
    return image


def quality_metrics(image: np.ndarray, edge_columns: int = 12) -> Dict[str, float]:
    h, w = image.shape
    roi = image[int(h * 0.20):int(h * 0.96), edge_columns:w-edge_columns]
    p1, p99 = np.percentile(roi, [1, 99])
    return {
        "sharpness": float(cv2.Laplacian(roi, cv2.CV_64F).var()),
        "contrast": float(np.std(roi)),
        "dynamic_range": float(p99 - p1),
        "dark_fraction": float(np.mean(roi <= 2)),
        "bright_fraction": float(np.mean(roi >= 253)),
    }


def derive_qc_thresholds(audit: pd.DataFrame) -> Dict[str, float]:
    return {
        "near_black_dynamic_range": 15.0,
        "near_black_contrast": 4.0,
        "near_black_dark_fraction": 0.97,
        "sharpness_1st_percentile": float(audit.sharpness.quantile(0.01)),
        "contrast_2nd_percentile": float(audit.contrast.quantile(0.02)),
        "dynamic_range_2nd_percentile": float(audit.dynamic_range.quantile(0.02)),
    }


def qc_pass(m: Dict[str, float], t: Dict[str, float]) -> Tuple[bool, str]:
    near_black = m["dynamic_range"] < t["near_black_dynamic_range"] or m["contrast"] < t["near_black_contrast"] or m["dark_fraction"] > t["near_black_dark_fraction"]
    severe = m["sharpness"] <= t["sharpness_1st_percentile"] and m["contrast"] <= t["contrast_2nd_percentile"] and m["dynamic_range"] <= t["dynamic_range_2nd_percentile"]
    reasons = (["near_black_or_empty"] if near_black else []) + (["severe_information_loss"] if severe else [])
    return not (near_black or severe), ";".join(reasons)


def audit_balanced_quality(balanced: Dict[str, pd.DataFrame], paths: Dict[str, Path]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    zip_path = paths["raw_csat"] / "pickle.zip"
    selected = pd.concat([df.assign(evaluation_split=s) for s, df in balanced.items()], ignore_index=True).drop_duplicates("archive_member")
    rows = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = set(archive.namelist())
        for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Auditing CSAT image quality"):
            entry = {"archive_member": row.archive_member, "patient_id": row.patient_id, "scr_target": int(row.scr_target), "evaluation_split": row.evaluation_split}
            try:
                if row.archive_member not in members: raise FileNotFoundError(row.archive_member)
                with archive.open(row.archive_member) as f: record = pickle.load(f)
                image = model_image(record["img"])
                entry.update(quality_metrics(image)); entry["read_error"] = ""
            except Exception as exc:
                entry.update({k: np.nan for k in ("sharpness", "contrast", "dynamic_range", "dark_fraction", "bright_fraction")}); entry["read_error"] = str(exc)
            rows.append(entry)
    audit = pd.DataFrame(rows)
    valid = audit[audit.read_error.eq("")].copy()
    thresholds = derive_qc_thresholds(valid)
    audit.to_csv(paths["meta"] / "csat_image_quality_audit.csv", index=False)
    json_dump({"thresholds": thresholds, "source_records": len(selected)}, paths["meta"] / "csat_quality_controlled_manifest_configuration.json")
    return audit, thresholds


def build_qc_arrays(balanced: Dict[str, pd.DataFrame], audit: pd.DataFrame, thresholds: Dict[str, float], paths: Dict[str, Path], strict: bool) -> Dict[str, pd.DataFrame]:
    zip_path = paths["raw_csat"] / "pickle.zip"
    quality = audit.set_index("archive_member")
    accepted_manifests: Dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(zip_path, "r") as archive:
        for split, manifest in balanced.items():
            accepted_rows, images, labels = [], [], []
            for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"QC/preparing {split}"):
                qrow = quality.loc[row.archive_member]
                if qrow.read_error:
                    continue
                m = {k: float(qrow[k]) for k in ("sharpness", "contrast", "dynamic_range", "dark_fraction", "bright_fraction")}
                ok, reason = qc_pass(m, thresholds)
                if not ok: continue
                with archive.open(row.archive_member) as f: record = pickle.load(f)
                img = model_image(record["img"])
                new = row.to_dict(); new.update(m); new["quality_control_passed"] = True; new["array_index"] = len(images); new["exclusion_reason"] = reason
                accepted_rows.append(new); images.append(img); labels.append(int(row.scr_target))
            if not images: raise RuntimeError(f"No {split} images passed QC")
            arr = np.stack(images).astype(np.uint8); lab = np.asarray(labels, dtype=np.uint8)
            np.save(paths["proc_csat"] / f"{split}_images_uint8.npy", arr)
            np.save(paths["proc_csat"] / f"{split}_labels_uint8.npy", lab)
            out = pd.DataFrame(accepted_rows).sort_values("array_index").reset_index(drop=True)
            out.to_csv(paths["proc_csat"] / f"{split}_manifest.csv", index=False)
            accepted_manifests[split] = out
    if strict:
        for split, expected in CSAT_EXPECTED["balanced_final"].items():
            assert len(accepted_manifests[split]) == expected, (split, len(accepted_manifests[split]), expected)
        for split, expected in (("train", (1879,1879)), ("validation", (389,389)), ("test", (556,556))):
            labels = np.load(paths["proc_csat"] / f"{split}_labels_uint8.npy")
            assert (int(labels.sum()), int(len(labels)-labels.sum())) == expected
    return accepted_manifests


def convert_box_to_pixels(box: Sequence[float], width: int = 576, height: int = 256) -> Tuple[int, int, int, int] | None:
    b = np.asarray(box, dtype=np.float32)
    if b.size < 4 or not np.isfinite(b[:4]).all(): return None
    cx, cy, bw, bh = map(float, b[:4])
    if bw <= 0 or bh <= 0: return None
    if np.max(np.abs(b[:4])) <= 1.5:
        cx *= width; bw *= width; cy *= height; bh *= height
    x1 = max(0, min(width-1, int(round(cx-bw/2)))); x2 = max(0, min(width-1, int(round(cx+bw/2))))
    y1 = max(0, min(height-1, int(round(cy-bh/2)))); y2 = max(0, min(height-1, int(round(cy+bh/2))))
    if x2 <= x1 or y2 <= y1 or (x2-x1+1) < 2 or (y2-y1+1) < 2: return None
    return x1, y1, x2, y2


def extract_box_targets(qc_manifests: Dict[str, pd.DataFrame], paths: Dict[str, Path]) -> None:
    zip_path = paths["raw_csat"] / "pickle.zip"
    summary = {}
    with zipfile.ZipFile(zip_path, "r") as archive:
        for split, manifest in qc_manifests.items():
            coord_rows, supervision = [], []
            invalid = 0
            for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"Extracting {split} boxes"):
                with archive.open(row.archive_member) as f: record = pickle.load(f)
                boxes = prepare_boxes(record.get("box", [])); labels = prepare_labels(record.get("label", []))
                n_pairs = min(len(boxes), len(labels)); fovea = scr = 0
                for j in range(n_pairs):
                    label = int(labels[j])
                    if label not in (0, 1): continue
                    px = convert_box_to_pixels(boxes[j])
                    if px is None: invalid += 1; continue
                    x1,y1,x2,y2 = px
                    coord_rows.append({"array_index": int(row.array_index), "archive_member": row.archive_member, "patient_id": row.patient_id,
                                       "label_id": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                       "box_width": x2-x1+1, "box_height": y2-y1+1, "box_area": (x2-x1+1)*(y2-y1+1)})
                    if label == 0: fovea += 1
                    if label == 1: scr += 1
                supervision.append({"array_index": int(row.array_index), "archive_member": row.archive_member, "patient_id": row.patient_id,
                                    "scr_target": int(row.scr_target), "fovea_box_count": fovea, "scr_box_count": scr,
                                    "scr_box_available": bool(scr > 0)})
            coords = pd.DataFrame(coord_rows); sup = pd.DataFrame(supervision)
            coords.to_csv(paths["box_targets"] / f"{split}_box_coordinates.csv", index=False)
            sup.to_csv(paths["box_targets"] / f"{split}_box_supervision.csv", index=False)
            summary[split] = {"valid_boxes": len(coords), "scr_boxes": int((coords.label_id==1).sum()) if len(coords) else 0,
                              "fovea_boxes": int((coords.label_id==0).sum()) if len(coords) else 0, "invalid_boxes": invalid}
    json_dump(summary, paths["meta"] / "csat_box_target_configuration.json")


def prepare_natural_test(records: pd.DataFrame, thresholds: Dict[str, float], paths: Dict[str, Path], strict: bool) -> pd.DataFrame:
    manifest = records[records.split == "test"].copy().reset_index(drop=True)
    zip_path = paths["raw_csat"] / "pickle.zip"
    accepted, images, labels, audit_rows = [], [], [], []
    with zipfile.ZipFile(zip_path, "r") as archive:
        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Preparing natural-distribution test"):
            entry = {"archive_member": row.archive_member, "patient_id": row.patient_id, "scr_target": int(row.scr_target)}
            try:
                with archive.open(row.archive_member) as f: record = pickle.load(f)
                img = model_image(record["img"]); m = quality_metrics(img); ok, reason = qc_pass(m, thresholds)
                entry.update(m); entry.update({"accepted": ok, "exclusion_reason": reason, "read_error": ""})
                if ok:
                    new = row.to_dict(); new.update(m); new["array_index"] = len(images); new["quality_control_passed"] = True
                    accepted.append(new); images.append(img); labels.append(int(row.scr_target))
            except Exception as exc:
                entry.update({"accepted": False, "exclusion_reason": "read_error", "read_error": str(exc)})
            audit_rows.append(entry)
    if not images: raise RuntimeError("No natural-test scans passed QC")
    np.save(paths["natural"] / "natural_test_images_uint8.npy", np.stack(images).astype(np.uint8))
    np.save(paths["natural"] / "natural_test_labels_uint8.npy", np.asarray(labels, dtype=np.uint8))
    out = pd.DataFrame(accepted).sort_values("array_index").reset_index(drop=True)
    out.to_csv(paths["natural"] / "natural_test_manifest.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(paths["natural"] / "natural_test_quality_audit.csv", index=False)
    lab = np.asarray(labels, dtype=np.uint8)
    summary = {"original_records": len(manifest), "quality_controlled_records": len(out), "excluded": len(manifest)-len(out),
               "patients": int(out.patient_id.nunique()), "scr_images": int(lab.sum()), "non_scr_images": int(len(lab)-lab.sum()),
               "scr_prevalence": float(lab.mean()), "class_balancing_applied": False, "model_retraining": False, "thresholds": thresholds}
    json_dump(summary, paths["natural"] / "natural_test_preparation_summary.json")
    if strict:
        assert len(out) == CSAT_EXPECTED["natural_final"], len(out)
        assert int(lab.sum()) == CSAT_EXPECTED["natural_scr"]
        assert int(len(lab)-lab.sum()) == CSAT_EXPECTED["natural_non_scr"]
        assert out.patient_id.nunique() == 16
    return out


def save_csat_samples(paths: Dict[str, Path]) -> None:
    image_path = paths["proc_csat"] / "train_images_uint8.npy"
    manifest_path = paths["proc_csat"] / "train_manifest.csv"
    box_path = paths["box_targets"] / "train_box_coordinates.csv"
    if not (image_path.exists() and manifest_path.exists()): return
    images = np.load(image_path, mmap_mode="r"); manifest = pd.read_csv(manifest_path)
    boxes = pd.read_csv(box_path) if box_path.exists() else pd.DataFrame()
    pos = manifest[manifest.scr_target == 1].head(2).array_index.tolist(); neg = manifest[manifest.scr_target == 0].head(2).array_index.tolist()
    ids = (pos + neg)[:4]
    fig, ax = plt.subplots(len(ids), 3, figsize=(15, 3*len(ids)))
    ax = np.atleast_2d(ax)
    for r, idx in enumerate(ids):
        img = np.asarray(images[int(idx)])
        ax[r,0].imshow(img,cmap="gray"); ax[r,0].set_title(f"Raw-preserving input {idx}")
        ax[r,1].imshow(img,cmap="gray"); ax[r,1].set_title("Genuine coordinate boxes")
        if len(boxes):
            for _, b in boxes[boxes.array_index == idx].iterrows():
                rect=plt.Rectangle((b.x1,b.y1),b.x2-b.x1,b.y2-b.y1,fill=False,linewidth=1.2); ax[r,1].add_patch(rect)
        ax[r,2].imshow(np.fliplr(img),cmap="gray"); ax[r,2].set_title("Training horizontal reflection")
        for c in range(3): ax[r,c].axis("off")
    fig.tight_layout(); fig.savefig(paths["fig"] / "csat_preprocessing_and_samples.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def verify_dataset_summary(paths: Dict[str, Path]) -> None:
    seg = np.load(paths["proc_seg"] / "preprocessed_images.npy", mmap_mode="r")
    print("\n" + "="*78)
    print("DATASET PREPARATION COMPLETE")
    print("="*78)
    print(f"Retinal-layer data : {seg.shape}")
    for split in ("train","validation","test"):
        lab = np.load(paths["proc_csat"] / f"{split}_labels_uint8.npy")
        man = pd.read_csv(paths["proc_csat"] / f"{split}_manifest.csv")
        print(f"CSAT {split:10s}: {len(lab):4d} scans | SCR={int(lab.sum()):4d} | Non-SCR={int(len(lab)-lab.sum()):4d} | patients={man.patient_id.nunique():2d}")
    natural = np.load(paths["natural"] / "natural_test_labels_uint8.npy")
    print(f"Natural test     : {len(natural)} scans | SCR={int(natural.sum())} | Non-SCR={int(len(natural)-natural.sum())}")
    print("Patient leakage   : prevented by patient-level split before balancing/QC")
    print("="*78)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/Sickle_Cell_OCT_Project"))
    parser.add_argument("--install-deps", action="store_true", help="Install Python dependencies first.")
    parser.add_argument("--skip-download", action="store_true", help="Use datasets already present under --root.")
    parser.add_argument("--no-strict-paper-checks", action="store_true", help="Do not assert the exact paper/notebook counts.")
    args = parser.parse_args()
    if args.install_deps: install_dependencies()
    imports()
    root = args.root.expanduser().resolve(); paths = ensure_dirs(root); strict = not args.no_strict_paper_checks
    print(f"Project root: {root}")
    if not args.skip_download: download_datasets(paths)
    audit_and_prepare_segmentation(paths, strict)
    audit = audit_csat_zip(paths, strict)
    records = make_patient_splits(audit, paths, strict)
    balanced = create_balanced_manifests(records, paths, strict)
    quality_audit, thresholds = audit_balanced_quality(balanced, paths)
    qc_manifests = build_qc_arrays(balanced, quality_audit, thresholds, paths, strict)
    extract_box_targets(qc_manifests, paths)
    prepare_natural_test(records, thresholds, paths, strict)
    save_csat_samples(paths)
    verify_dataset_summary(paths)


if __name__ == "__main__":
    main()
