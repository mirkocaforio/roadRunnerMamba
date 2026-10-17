#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export per-sample softmax probabilities of the RSCDMamba model for the stacking ensemble.

Run this in the MAMBA conda env, from the /home/caforio/rscd_mamba directory:

    python export_probs_mamba.py \
        --data_path ../datahome/RSCD_dataset_1million_reorganized \
        --resume core/processing/output/<run>/rscd_mamba/checkpoints/best_model.pt \
        --out_dir /home/mcaforio/datahome/stacking_data/preds

It writes mamba_val.npz / mamba_test.npz with the SAME schema as the JITrans export
(export_probs.py in the JIT project): probs (N,C), rel_paths (N,), targets (N,), classes (C,).
The rel_paths key ("<class_folder>/<file_name>") lets stacking.py align the two models
per image regardless of DataLoader ordering.

NOTE: the RSCDMamba constructor args below must match the loaded checkpoint. Defaults follow
core/processing/inference.py. If they differ, pass them via CLI or set them here.
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make `core` importable regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.model.net import RSCDMamba
from core.data.dataloader import build_dataset


def get_args():
    p = argparse.ArgumentParser("Export RSCDMamba softmax probabilities for stacking")
    # Data
    p.add_argument("--data_path", required=True, type=str)
    p.add_argument("--data_set", default="RSCD", type=str)
    p.add_argument("--img_size", default=224, type=int)
    p.add_argument("--eval_crop_ratio", default=0.875, type=float)
    # Model / checkpoint
    p.add_argument("--resume", required=True, type=str,
                   help="Path to the trained RSCDMamba checkpoint (best_model.pt).")
    p.add_argument("--num_classes", default=6, type=int)
    p.add_argument("--depths", default="2,4,8,4", type=str,
                   help="Comma-separated stage depths (must match the checkpoint).")
    p.add_argument("--dims", default="64", type=str,
                   help="Either a single int (base dim) or comma-separated per-stage dims, "
                        "matching the checkpoint instantiation.")
    p.add_argument("--drop_path_rate", default=0.3, type=float)
    # Runtime
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--num_workers", default=2, type=int)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out_dir", default="/home/mcaforio/JIT/stacking/preds", type=str)
    return p.parse_args()


def parse_dims(s):
    parts = [int(x) for x in s.split(",") if x.strip() != ""]
    return parts[0] if len(parts) == 1 else parts


def rel_paths_from_dataset(dataset):
    """Return ['<class_folder>/<file_name>', ...] in dataset.samples order."""
    rels = []
    for path, _ in dataset.samples:
        cls = os.path.basename(os.path.dirname(path))
        fname = os.path.basename(path)
        rels.append(f"{cls}/{fname}")
    return rels


def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        return {k[len("module."):] if k.startswith("module.") else k: v
                for k, v in state_dict.items()}
    return state_dict


@torch.no_grad()
def export_split(model, dataset, device, args, out_path):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    model.eval()
    all_probs, all_targets = [], []
    for images, targets in tqdm(loader, desc=os.path.basename(out_path), leave=False):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.numpy())

    probs = np.concatenate(all_probs, axis=0).astype(np.float32)
    targets = np.concatenate(all_targets, axis=0).astype(np.int64)
    rel_paths = np.array(rel_paths_from_dataset(dataset), dtype=object)
    classes = np.array(list(dataset.classes), dtype=object)

    assert len(probs) == len(targets) == len(rel_paths), "Length mismatch in export."

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, probs=probs, rel_paths=rel_paths,
                        targets=targets, classes=classes)
    print(f"[export_mamba] wrote {out_path}  (N={len(probs)}, classes={list(classes)})")


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export_mamba] device: {device}")

    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    val_dataset, nb_classes = build_dataset(is_train=False, is_test=False, args=args)
    test_dataset, _ = build_dataset(is_train=False, is_test=True, args=args)
    print(f"[export_mamba] classes ({nb_classes}): {val_dataset.classes}")
    print(f"[export_mamba] val: {len(val_dataset)} | test: {len(test_dataset)}")

    model = RSCDMamba(
        img_size=args.img_size,
        num_classes=args.num_classes,
        depths=parse_dims(args.depths),
        dims=parse_dims(args.dims),
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    state = strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[export_mamba] WARNING load_state_dict: "
              f"{len(missing)} missing, {len(unexpected)} unexpected keys. "
              f"Check that --depths/--dims match the checkpoint.")
        if missing:
            print(f"  missing[:5]={missing[:5]}")
        if unexpected:
            print(f"  unexpected[:5]={unexpected[:5]}")

    export_split(model, val_dataset, device, args, os.path.join(args.out_dir, "mamba_val.npz"))
    export_split(model, test_dataset, device, args, os.path.join(args.out_dir, "mamba_test.npz"))


if __name__ == "__main__":
    main()
