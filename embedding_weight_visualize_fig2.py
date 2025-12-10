#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# -------------------------
# Optional: cuML UMAP (GPU) → fallback to umap-learn (CPU)
# -------------------------
USE_CUML = False
try:
    from cuml.manifold import UMAP as cuUMAP  # RAPIDS
    USE_CUML = True
except Exception:
    from umap import UMAP as cpuUMAP          # umap-learn
    USE_CUML = False

parser = argparse.ArgumentParser(description="UMAP viz of embedding weights (Before/After TSP).")
parser.add_argument("--norm", default="cos_sim", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--cache_dir", default="/extdata1/donghwan/huggingface", type=str)
parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf", type=str)
parser.add_argument("--start", default=9962, type=int, help="inclusive token id (Before panel)")
parser.add_argument("--end", default=9970, type=int, help="exclusive token id (Before panel)")
parser.add_argument("--idx_start", default=18927, type=int, help="inclusive new-index (After panel)")
parser.add_argument("--idx_end", default=18935, type=int, help="exclusive new-index (After panel)")
parser.add_argument("--single_umap", action="store_true",
                    help="Fit UMAP once on original weights and just reorder coords for TSP (faster, slightly different result).")
args = parser.parse_args()

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# -------------------------
# Load sorted indices (TSP order)
# -------------------------
with open(f"llama2-7b-hf_sorted_idx_{args.norm}.json", "r") as f:
    sorted_idx = json.load(f)

old_id_to_new_id = {old_id: new_id for new_id, old_id in enumerate(sorted_idx)}
new_id_to_old_id = {new_id: old_id for new_id, old_id in enumerate(sorted_idx)}
new_order = [new_id_to_old_id[i] for i in range(len(sorted_idx))]  # permutation

# -------------------------
# Model / Tokenizer
# -------------------------
tokenizer = LlamaTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
model = LlamaForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir=args.cache_dir,
)
model.eval()

# -------------------------
# Windows (Before / After)
# -------------------------
start = args.start - 1  # keep the original off-by-one convention
end = args.end
idx_start = args.idx_start - 1
idx_end = args.idx_end

# -------------------------
# Embedding weights → numpy float32 (UMAP is float32-friendly)
# -------------------------
emb_w = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float32, copy=False)

# -------------------------
# UMAP helpers (GPU if available)
# -------------------------
def fit_umap(x, seed):
    if USE_CUML:
        reducer = cuUMAP(random_state=seed)
        # cuML expects float32 on device; host array is fine (it copies under the hood)
        return reducer.fit_transform(x)
    else:
        reducer = cpuUMAP(random_state=seed)
        return reducer.fit_transform(x)

def normalize_xy(xy):
    norms = np.linalg.norm(xy, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return xy / norms

# -------------------------
# Tokens (Before)
# -------------------------
tokens_left = tokenizer.convert_ids_to_tokens(list(range(start, end)))
print(f"Tokens (Before) from {start} to {end-1}:")
for idx, tok in enumerate(tokens_left, start=start):
    print(f"{idx}: {tok}")

# -------------------------
# UMAP fit(s)
# -------------------------
if args.single_umap:
    # Fit once on original weights; "After" is just a permutation of coordinates
    xy_left = fit_umap(emb_w, args.seed)               # (V, 2)
    xy_right = xy_left[new_order]                      # reorder
else:
    # Fit twice to preserve the original visual result behavior
    xy_left = fit_umap(emb_w, args.seed)
    emb_w_sorted = emb_w[new_order]
    xy_right = fit_umap(emb_w_sorted, args.seed)

xy_left = normalize_xy(xy_left).astype(np.float32, copy=False)
xy_right = normalize_xy(xy_right).astype(np.float32, copy=False)

# -------------------------
# Tokens (After) — map new indices back to old ids
# -------------------------
tokens_right_ids = [new_id_to_old_id[num] for num in range(idx_start, idx_end)]
tokens_right = tokenizer.convert_ids_to_tokens(tokens_right_ids)
print(f"Tokens (After) from {idx_start} to {idx_end-1}:")
for rel, tok in enumerate(tokens_right, start=idx_start):
    print(f"{rel}: {tok}")

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(wspace=0.3)

# ----- Left (Before) -----
n_left = end - start
colors_left = [(1 - i / max(1, n_left - 1), 0, i / max(1, n_left - 1)) for i in range(n_left)]
markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'p', 'H']

for i in range(n_left):
    idx0 = start + i
    token_label = tokens_left[i].replace("_", r"\_")
    ax[0].scatter(
        xy_left[idx0:idx0+1, 0],
        xy_left[idx0:idx0+1, 1],
        color=colors_left[i],
        s=150,
        marker=markers[i % len(markers)],
        edgecolors="none",
        linewidths=0.8,
        alpha=0.8,
        label=f"{token_label}",
    )

# 연결선/레이블
x_left = xy_left[start:end, 0]
y_left = xy_left[start:end, 1]
ax[0].plot(x_left, y_left, color=(0, 1, 0), linewidth=1)

ax[0].set_xlabel(r"$x$-axis", fontsize=18)
ax[0].set_ylabel(r"$y$-axis", fontsize=18)
ax[0].set_title("Without TSP", fontsize=28)
ax[0].legend(markerscale=1.5, fontsize=13.5, ncol=3, loc="upper left")
ax[0].tick_params(axis="both", labelsize=16)

for i in range(n_left):
    xv = xy_left[start + i, 0]
    yv = xy_left[start + i, 1]
    ax[0].text(xv, yv + 0.003, f"{tokens_left[i]}", fontsize=22, color="black",
               ha="center", va="bottom")

# ----- Right (After) -----
n_right = idx_end - idx_start
colors_right = [(1 - i / max(1, n_right - 1), 0, i / max(1, n_right - 1)) for i in range(n_right)]

for i in range(n_right):
    idx0 = idx_start + i
    token_label = tokens_right[i].replace("_", r"\_")
    ax[1].scatter(
        xy_right[idx0:idx0+1, 0],
        xy_right[idx0:idx0+1, 1],
        color=colors_right[i],
        s=150,
        marker=markers[i % len(markers)],
        edgecolors="none",
        linewidths=0.8,
        alpha=0.8,
        label=f"{token_label}",
    )

x_right = xy_right[idx_start:idx_end, 0]
y_right = xy_right[idx_start:idx_end, 1]
ax[1].plot(x_right, y_right, color=(0, 1, 0), linewidth=1)

ax[1].set_xlabel(r"$x$-axis", fontsize=18)
ax[1].set_title("With TSP", fontsize=28)
ax[1].legend(markerscale=1.5, fontsize=13.5, ncol=3, loc="upper left")
ax[1].tick_params(axis="both", labelsize=16)

for i in range(n_right):
    xv = xy_right[idx_start + i, 0]
    yv = xy_right[idx_start + i, 1]
    ax[1].text(xv, yv + 0.003, f"{tokens_right[i]}", fontsize=22, color="black",
               ha="center", va="bottom")

# ----- 통일된 축 범위 -----
all_x = np.concatenate([xy_left[start:end, 0], xy_right[idx_start:idx_end, 0]])
all_y = np.concatenate([xy_left[start:end, 1], xy_right[idx_start:idx_end, 1]])
x_min, x_max = np.min(all_x), np.max(all_x)
y_min, y_max = np.min(all_y), np.max(all_y)
for axes in ax:
    axes.set_xlim(x_min - 0.085, x_max + 0.05)
    axes.set_ylim(y_min - 0.05, y_max + 0.012)

# 연결선 가운데 번호
for i in range(max(0, n_left - 1)):
    ax[0].text((x_left[i] + x_left[i+1]) / 2,
               (y_left[i] + y_left[i+1]) / 2,
               str(i + 1), color="black", fontsize=14, ha="center", va="center")
for i in range(max(0, n_right - 1)):
    ax[1].text((x_right[i] + x_right[i+1]) / 2,
               (y_right[i] + y_right[i+1]) / 2,
               str(i + 1), color="black", fontsize=14, ha="center", va="center")

plt.tight_layout()
out_name = f"step0_embedding_weights_umap_{start+1}-{end}_{idx_start+1}-{idx_end}"
if args.single_umap:
    out_name += "_singlefit"
plt.savefig(out_name + ".png", dpi=200)

# -------------------------
# Print token → 2D coords (for both panels)
# -------------------------
print("\n=== Coordinates (Before / original indices) ===")
for i in range(n_left):
    tid = start + i
    print(f"[{tid:>6}] {tokens_left[i]:<20s} -> ({xy_left[tid,0]:+.6f}, {xy_left[tid,1]:+.6f})")

print("\n=== Coordinates (After / TSP order mapped back to old ids) ===")
for i in range(n_right):
    tid = idx_start + i
    print(f"[{tid:>6}] {tokens_right[i]:<20s} -> ({xy_right[tid,0]:+.6f}, {xy_right[tid,1]:+.6f})")
