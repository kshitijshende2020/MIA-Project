# -*- coding: utf-8 -*-
"""
Full CVI Pipeline on OIMHS Dataset (Inference + Ground Truth Evaluation)
========================================================================
Reference Paper:
  "Automatic Choroid Vascularity Index Calculation in Optical Coherence
   Tomography Images with Low-Contrast Sclerochoroidal Junction Using Deep Learning"
   (Arian et al., Photonics 2023)

Key Features:
  - Evaluates pre-trained U-Net on OIMHS benchmark dataset
  - Computes pixel-level Segmentation Metrics: Dice Similarity (DSC) & IoU
  - Calculates Choroid Vascularity Index (CVI) via Niblack adaptive thresholding
  - Compares CVI_predicted vs. CVI_ground_truth (Correlation, MAE)
  - Exports comprehensive CSV summary, scatter plots, and multi-panel figures

USAGE (Google Colab / Local):
  1. pip install tensorflow tf_keras opencv-python-headless scikit-image matplotlib -q
  2. Ensure 'images/' and 'masks/' directories are present
  3. python oimhs_cvi_pipeline.py
"""

import os
import cv2
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import measure
from skimage.filters import threshold_niblack

# ── Keras 2 / 3 Compatibility Layer ──────────────────────────────────────────
import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")

try:
    import tf_keras as keras
    print("Using tf_keras compatibility layer")
except ImportError:
    keras = tf.keras
    print("Using standard tf.keras")

from keras.preprocessing.image import img_to_array
from keras.layers import (
    Input, Conv2D, Conv2DTranspose,
    MaxPooling2D, BatchNormalization, Activation, Dropout, concatenate
)
from keras.models import Model

# ════════════════════════════════════════════════════════════════════════════
# ── CONFIGURATION & PATHS ────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# Input image dimensions (OIMHS processed images are 512x512)
im_height    = 512
im_width     = 512
im_size      = 256    # U-Net internal training/inference resolution
number_class = 3      # 0: Background/Retina, 1: Choroid, 2: Sclera/Sub-choroid

# Search candidate directories for images and masks
candidate_bases = [
    '.',
    './OIMHS_dataset_clean',
    './OIMHS_dataset_clean/proc',
    '/content',
    '/content/OIMHS_dataset_clean',
    '/content/OIMHS_dataset_clean/proc',
    r"F:\Medical Image Analysis\MIA Project",
    r"F:\Medical Image Analysis\MIA Project\OIMHS_dataset_clean",
    r"F:\Medical Image Analysis\MIA Project\OIMHS_dataset_clean\proc"
]

path_img = None
path_mask = None

for base in candidate_bases:
    p_img = os.path.join(base, 'images')
    p_mask = os.path.join(base, 'masks')
    if os.path.isdir(p_img) and os.path.isdir(p_mask):
        path_img = p_img
        path_mask = p_mask
        break

if path_img is None:
    # Deep search if nested differently
    for root, dirs, _ in os.walk('.'):
        if 'images' in dirs and 'masks' in dirs:
            path_img = os.path.join(root, 'images')
            path_mask = os.path.join(root, 'masks')
            break

output_dir = './results'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

# Weight file selection (Raster or EDI)
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
model_name = 'saved model for raster data.h5'
model_path = None

model_search_dirs = [
    script_dir,
    '.',
    '/content',
    '/content/oimhs_data/OIMHS_Complete_Colab',
    r"F:\Medical Image Analysis\MIA Project"
]

for d in model_search_dirs:
    cand_raster = os.path.join(d, 'saved model for raster data.h5')
    if os.path.exists(cand_raster):
        model_path = cand_raster
        break
    cand_edi = os.path.join(d, 'saved model for EDI data.h5')
    if os.path.exists(cand_edi):
        model_path = cand_edi
        break

if model_path is None or not os.path.exists(model_path):
    # Deep search for any .h5 file
    for root, _, files in os.walk('/content'):
        for f in files:
            if f.endswith('.h5'):
                model_path = os.path.join(root, f)
                break
        if model_path: break

if model_path is None:
    model_path = 'saved model for raster data.h5'

print(f"Images Path : {path_img}")
print(f"Masks Path  : {path_mask}")
print(f"Model Path  : {model_path}")
print(f"Output Path : {output_dir}")

# ════════════════════════════════════════════════════════════════════════════
# ── MATCH IMAGE AND MASK FILES ────────────────══════════════════════════════
# ════════════════════════════════════════════════════════════════════════════

all_images = sorted([f for f in os.listdir(path_img) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
image_list = []

for fname in all_images:
    mask_file = os.path.join(path_mask, fname)
    if os.path.exists(mask_file):
        image_list.append(fname)

num_img = len(image_list)
print(f"\nMatched {num_img} image-mask pairs successfully.")
if num_img == 0:
    raise FileNotFoundError(f"No matching image-mask pairs found between {path_img} and {path_mask}")

# ════════════════════════════════════════════════════════════════════════════
# ── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def conv2d_block(inp, n_filters, kernel_size=3, batchnorm=True):
    """Matches the paper's saved .h5 weights (1 Conv + 1 BatchNorm per block)."""
    x = Conv2D(n_filters, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(inp)
    if batchnorm: 
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def get_unet(input_img, num_classes=3, n_filters=32, dropout=0.05, batchnorm=True):
    # Contracting path
    c1 = conv2d_block(input_img, n_filters * 1, batchnorm=batchnorm)
    p1 = Dropout(dropout)(MaxPooling2D((2, 2))(c1))

    c2 = conv2d_block(p1, n_filters * 2, batchnorm=batchnorm)
    p2 = Dropout(dropout)(MaxPooling2D((2, 2))(c2))

    c3 = conv2d_block(p2, n_filters * 4, batchnorm=batchnorm)
    p3 = Dropout(dropout)(MaxPooling2D((2, 2))(c3))

    c4 = conv2d_block(p3, n_filters * 8, batchnorm=batchnorm)
    p4 = Dropout(dropout)(MaxPooling2D((2, 2))(c4))

    # Bottleneck
    c5 = conv2d_block(p4, n_filters * 16, batchnorm=batchnorm)

    # Expanding path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = Dropout(dropout)(concatenate([u6, c4]))
    c6 = conv2d_block(u6, n_filters * 8, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = Dropout(dropout)(concatenate([u7, c3]))
    c7 = conv2d_block(u7, n_filters * 4, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = Dropout(dropout)(concatenate([u8, c2]))
    c8 = conv2d_block(u8, n_filters * 2, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = Dropout(dropout)(concatenate([u9, c1]))
    c9 = conv2d_block(u9, n_filters * 1, batchnorm=batchnorm)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    return Model(inputs=[input_img], outputs=[outputs])

input_tensor = Input((im_size, im_size, 1))
model = get_unet(input_tensor, number_class, n_filters=32, dropout=0.05, batchnorm=True)
model.load_weights(model_path, by_name=False)
print("U-Net pre-trained weights loaded successfully.")

# ════════════════════════════════════════════════════════════════════════════
# ── METRIC CALCULATION FUNCTIONS ─────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def compute_dice_and_iou(gt_mask, pred_mask):
    """Calculates Dice Similarity Coefficient and IoU for binary masks."""
    gt_bin = (gt_mask > 0).astype(bool)
    pred_bin = (pred_mask > 0).astype(bool)

    intersection = np.logical_and(gt_bin, pred_bin).sum()
    total_area = gt_bin.sum() + pred_bin.sum()
    union = np.logical_or(gt_bin, pred_bin).sum()

    dice = (2.0 * intersection) / (total_area) if total_area > 0 else 1.0
    iou = (intersection) / (union) if union > 0 else 1.0
    return float(dice), float(iou)

def compute_cvi_niblack(denoised_img, choroid_mask, window_size=15, k=0.001):
    """Computes Choroid Vascularity Index using local Niblack binarization."""
    if np.sum(choroid_mask) == 0:
        return 0.0, np.zeros_like(denoised_img)

    # Local Niblack adaptive threshold
    thresh = threshold_niblack(denoised_img, window_size=window_size, k=k)
    binary_stroma = denoised_img > (thresh * 1.03)

    # Vessels (luminal area) are the darker non-stromal pixels inside the choroid
    luminal_pixels = np.logical_and(~binary_stroma, choroid_mask > 0)
    total_choroid_pixels = np.sum(choroid_mask > 0)

    cvi = float(np.sum(luminal_pixels)) / float(total_choroid_pixels)
    return cvi, luminal_pixels.astype(np.uint8)

# ════════════════════════════════════════════════════════════════════════════
# ── BATCH INFERENCE & EVALUATION LOOP ────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

BATCH_SIZE = 32
results = []
SAVE_VIS_EVERY = 50

print(f"\nProcessing {num_img} images in batches of {BATCH_SIZE}...")

for start_idx in range(0, num_img, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, num_img)
    batch_fnames = image_list[start_idx:end_idx]
    current_bsize = len(batch_fnames)

    # Pre-allocate batch tensors
    batch_x = np.zeros((current_bsize, im_size, im_size, 1), dtype=np.float32)
    batch_orig_imgs = []
    batch_denoised_imgs = []
    batch_gt_masks = []

    # Load and preprocess batch
    for b, fname in enumerate(batch_fnames):
        img_raw = cv2.imread(os.path.join(path_img, fname), cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(os.path.join(path_mask, fname), cv2.IMREAD_GRAYSCALE)

        if img_raw.shape != (im_height, im_width):
            img_raw = cv2.resize(img_raw, (im_width, im_height))
        if mask_raw.shape != (im_height, im_width):
            mask_raw = cv2.resize(mask_raw, (im_width, im_height), interpolation=cv2.INTER_NEAREST)

        denoised = cv2.fastNlMeansDenoising(img_raw, None, 10)
        norm_denoised = denoised.astype(np.float32) / 255.0

        # Resize for U-Net
        img_resized = resize(img_raw, (im_size, im_size), mode='constant', preserve_range=True) / 255.0
        batch_x[b, :, :, 0] = img_resized

        batch_orig_imgs.append(img_raw)
        batch_denoised_imgs.append(norm_denoised)
        batch_gt_masks.append(mask_raw)

    # Run U-Net prediction
    batch_pred = model.predict(batch_x, verbose=0)
    batch_pred_choroid = (batch_pred[:, :, :, 1] > 0.5).astype(np.uint8)

    # Post-process & Evaluate each sample
    for b in range(current_bsize):
        idx = start_idx + b
        fname = batch_fnames[b]
        gt_mask = (batch_gt_masks[b] > 127).astype(np.uint8)
        norm_denoised = batch_denoised_imgs[b]
        orig_img = batch_orig_imgs[b]

        # Post-processing: keep largest connected component
        pred_small = batch_pred_choroid[b]
        lbl = measure.label(pred_small)
        regions = measure.regionprops(lbl)
        if len(regions) > 0:
            largest = max(regions, key=lambda r: r.area)
            cleaned_small = (lbl == largest.label).astype(np.uint8)
        else:
            cleaned_small = pred_small

        # Upscale prediction back to original 512x512
        pred_choroid = np.round(resize(cleaned_small, (im_height, im_width), mode='constant', preserve_range=True)).astype(np.uint8)

        # 1. Segmentation Accuracy
        dice, iou = compute_dice_and_iou(gt_mask, pred_choroid)

        # 2. CVI from Ground Truth Mask
        cvi_gt, gt_luminal = compute_cvi_niblack(norm_denoised, gt_mask)

        # 3. CVI from Predicted Mask
        cvi_pred, pred_luminal = compute_cvi_niblack(norm_denoised, pred_choroid)

        results.append({
            'name': fname,
            'dice': dice,
            'iou': iou,
            'cvi_gt': cvi_gt,
            'cvi_pred': cvi_pred,
            'cvi_diff': abs(cvi_gt - cvi_pred)
        })

        # Save sample visual figures
        if idx % SAVE_VIS_EVERY == 0 or idx == 0:
            fig, ax = plt.subplots(2, 2, figsize=(12, 10))
            ax[0, 0].imshow(orig_img, cmap='gray')
            ax[0, 0].set_title(f"Original B-Scan: {fname}")
            ax[0, 0].axis('off')

            ax[0, 1].imshow(orig_img, cmap='gray')
            ax[0, 1].contour(gt_mask, colors='lime', linewidths=1.5, levels=[0.5])
            ax[0, 1].contour(pred_choroid, colors='red', linewidths=1.5, levels=[0.5])
            ax[0, 1].set_title(f"Boundaries: GT (Green) vs Pred (Red)\nDice = {dice:.3f} | IoU = {iou:.3f}")
            ax[0, 1].axis('off')

            ax[1, 0].imshow(gt_luminal, cmap='Blues')
            ax[1, 0].set_title(f"GT Luminal Vessels (CVI = {cvi_gt:.4f})")
            ax[1, 0].axis('off')

            ax[1, 1].imshow(pred_luminal, cmap='Reds')
            ax[1, 1].set_title(f"Pred Luminal Vessels (CVI = {cvi_pred:.4f})")
            ax[1, 1].axis('off')

            plt.tight_layout()
            vis_file = os.path.join(output_dir, 'visualizations', f"eval_{idx:04d}_{fname}")
            plt.savefig(vis_file, dpi=120)
            plt.close()

    if end_idx % 200 == 0 or end_idx == num_img:
        current_dices = [r['dice'] for r in results]
        print(f"  Processed [{end_idx:4d}/{num_img}] | Running Mean Dice: {np.mean(current_dices):.4f}")

# ════════════════════════════════════════════════════════════════════════════
# ── EXPORT SUMMARY METRICS & PLOTS ───────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

csv_file = os.path.join(output_dir, 'cvi_full_evaluation.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'dice', 'iou', 'cvi_gt', 'cvi_pred', 'cvi_diff'])
    writer.writeheader()
    writer.writerows(results)

dices     = np.array([r['dice'] for r in results])
ious      = np.array([r['iou'] for r in results])
cvis_gt   = np.array([r['cvi_gt'] for r in results])
cvis_pred = np.array([r['cvi_pred'] for r in results])
cvi_diffs = np.array([r['cvi_diff'] for r in results])

# Correlation between GT CVI and Predicted CVI
valid_mask = (cvis_gt > 0) & (cvis_pred > 0)
if np.sum(valid_mask) > 1:
    corr = np.corrcoef(cvis_gt[valid_mask], cvis_pred[valid_mask])[0, 1]
else:
    corr = 0.0

print("\n" + "="*65)
print("              OIMHS EXPERIMENTAL EVALUATION RESULTS              ")
print("="*65)
print(f" Total Evaluated B-scans     : {num_img}")
print(f" Mean Dice Similarity (DSC)  : {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
print(f" Median Dice                 : {np.median(dices):.4f}")
print(f" Mean IoU (Jaccard Index)    : {np.mean(ious):.4f} +/- {np.std(ious):.4f}")
print(f" Mean Ground-Truth CVI       : {np.mean(cvis_gt):.4f} +/- {np.std(cvis_gt):.4f}")
print(f" Mean Predicted CVI          : {np.mean(cvis_pred):.4f} +/- {np.std(cvis_pred):.4f}")
print(f" CVI Mean Absolute Error     : {np.mean(cvi_diffs):.4f}")
print(f" Pearson Correlation (r)     : {corr:.4f}")
print(f" Full metrics exported to    : {csv_file}")
print("="*65)

# 1. Dice Distribution Histogram
plt.figure(figsize=(8, 5))
plt.hist(dices, bins=35, color='royalblue', edgecolor='black', alpha=0.8)
plt.axvline(np.mean(dices), color='red', linestyle='--', linewidth=2, label=f"Mean Dice = {np.mean(dices):.3f}")
plt.xlabel("Dice Similarity Coefficient (DSC)")
plt.ylabel("Number of B-Scans")
plt.title(f"Choroid Segmentation Dice Distribution on OIMHS (n={num_img})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dice_score_distribution.png'), dpi=150)
plt.close()

# 2. Scatter Plot: CVI Ground Truth vs. CVI Predicted
plt.figure(figsize=(7, 7))
plt.scatter(cvis_gt[valid_mask], cvis_pred[valid_mask], alpha=0.35, color='darkgreen', s=18)
min_val = min(np.min(cvis_gt[valid_mask]), np.min(cvis_pred[valid_mask]))
max_val = max(np.max(cvis_gt[valid_mask]), np.max(cvis_pred[valid_mask]))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement (y = x)')
plt.xlabel("Ground Truth CVI (Manual Choroid Mask)")
plt.ylabel("Predicted CVI (U-Net Choroid Mask)")
plt.title(f"CVI Concordance on OIMHS Dataset\nPearson r = {corr:.3f} | MAE = {np.mean(cvi_diffs):.4f}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cvi_gt_vs_pred_scatter.png'), dpi=150)
plt.close()

print("\nPlots saved successfully in:", output_dir)
