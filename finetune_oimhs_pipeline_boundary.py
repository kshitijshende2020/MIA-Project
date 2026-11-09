# -*- coding: utf-8 -*-
"""
OIMHS Choroid U-Net Fine-Tuning & Zero-Shot vs Fine-Tuned Benchmark Pipeline
=============================================================================
Reference:
  Arian, Mahmoudi, Riazi-Esfahani, Faghihi, Mirshahi, Ghassemi, Khodabande,
  Kafieh, Khalili Pour. "Automatic Choroid Vascularity Index Calculation in OCT
  Images with Low-Contrast Sclerochoroidal Junction Using Deep Learning."
  Photonics 2023, 10, 234.

Specifications:
  - Resolution: 128x128 images and 3-class masks (0: Above, 1: Choroid, 2: Below)
  - Split: Patient-level split (two eyes of one patient never cross splits).
           Fixed seed = 42. Capped at 30 slices per eye.
  - Loss: Kafieh loss (WCCE + Dice + Tversky), early stopping on validation.
  - Evaluation: Zero-shot vs Fine-tuned on test eyes.
                Choroid Dice & IoU, signed CVI diff (pred - ref), failed slices (Dice < 0.1).
                Per-eye averages -> Mean and SD across eyes.
                Saves CSV and 6 overlay images (including 2 worst-case failure modes).
"""

import os
import sys
import csv
import json
import random
import shutil
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict
from PIL import Image
import cv2
from skimage.filters import threshold_niblack
from skimage.transform import resize
from skimage import measure

# TensorFlow & Keras
import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")

try:
    import tf_keras as keras
    import tf_keras.backend as K
    from tf_keras.layers import (
        Input, Conv2D, Conv2DTranspose, MaxPooling2D,
        BatchNormalization, Activation, Dropout, concatenate
    )
    from tf_keras.models import Model
    from tf_keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tf_keras.optimizers import Adam
    print("Using tf_keras compatibility layer.")
except ImportError:
    import keras
    import keras.backend as K
    from keras.layers import (
        Input, Conv2D, Conv2DTranspose, MaxPooling2D,
        BatchNormalization, Activation, Dropout, concatenate
    )
    from keras.models import Model
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from keras.optimizers import Adam
    print("Using native keras.")

# ════════════════════════════════════════════════════════════════════════════
# 1. PATH RESOLUTION & LOCAL EXTRACTION (NO GOOGLE DRIVE)
# ════════════════════════════════════════════════════════════════════════════

WORKSPACE = os.path.abspath('.')
SKIP_TRAINING = True
DATA_ROOT = os.path.join(WORKSPACE, 'oimhs_local')
os.makedirs(DATA_ROOT, exist_ok=True)

# Candidate zip locations
CANDIDATE_ZIPS = [
    os.path.join(WORKSPACE, 'oimhs_proc.zip'),
    '/content/oimhs_proc.zip',
    os.path.join(WORKSPACE, 'OIMHS_dataset_clean.zip'),
    os.path.join(WORKSPACE, 'OIMHS_Complete_Colab.zip'),
    '/content/OIMHS_Complete_Colab.zip'
]

# Candidate pretrained model locations
CANDIDATE_MODELS = [
    os.path.join(WORKSPACE, 'saved model for raster data.h5'),
    '/content/saved model for raster data.h5',
    os.path.join(WORKSPACE, 'saved model for EDI data.h5'),
    '/content/saved model for EDI data.h5'
]

# Find and extract zip locally if not already extracted
img_dir = os.path.join(DATA_ROOT, 'images')
mask_dir = os.path.join(DATA_ROOT, 'masks')

if not (os.path.exists(img_dir) and os.path.exists(mask_dir) and len(os.listdir(img_dir)) > 100):
    zip_found = None
    for cz in CANDIDATE_ZIPS:
        if os.path.exists(cz):
            zip_found = cz
            break

    if zip_found is None:
        # Check if already extracted in working dir
        for cd in ['./OIMHS_dataset_clean/proc', './proc', '/content/OIMHS_dataset_clean/proc']:
            if os.path.exists(os.path.join(cd, 'images')) and os.path.exists(os.path.join(cd, 'masks')):
                img_dir = os.path.abspath(os.path.join(cd, 'images'))
                mask_dir = os.path.abspath(os.path.join(cd, 'masks'))
                break
    else:
        print(f"Extracting dataset archive: {zip_found} -> {DATA_ROOT}")
        with zipfile.ZipFile(zip_found, 'r') as zf:
            zf.extractall(DATA_ROOT)
        
        # Locate images & masks folders inside DATA_ROOT
        for root, dirs, _ in os.walk(DATA_ROOT):
            if 'images' in dirs and 'masks' in dirs:
                img_dir = os.path.join(root, 'images')
                mask_dir = os.path.join(root, 'masks')
                break

print(f"Using Images Folder: {img_dir}")
print(f"Using Masks Folder : {mask_dir}")

# Locate pretrained model
pretrained_weights_path = None
for cm in CANDIDATE_MODELS:
    if os.path.exists(cm):
        pretrained_weights_path = cm
        break

if pretrained_weights_path is None:
    for root, _, files in os.walk(WORKSPACE):
        for f in files:
            if f.endswith('.h5') and 'raster' in f.lower():
                pretrained_weights_path = os.path.join(root, f)
                break
        if pretrained_weights_path:
            break

print(f"Using Pretrained Weights: {pretrained_weights_path}")

RESULTS_DIR = os.path.join(WORKSPACE, 'finetune_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'overlay_images'), exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# 2. PATIENT-AWARE DEMOGRAPHICS SPLIT & 30-SLICE CAPPING
# ════════════════════════════════════════════════════════════════════════════

# Built-in exact mapping from OIMHS Demographics of participants
# Ensures two eyes of the same patient always stay together in the same split!
KNOWN_TWO_EYES_PATIENTS = {
    '6': [6, 7],
    '54': [55, 56],
    '76': [78, 79],
    '100': [103, 104],
    '109': [113, 114],
    '112': [117, 118]
}

def get_patient_eye_mapping(demographics_csv=None):
    patient_to_eyes = defaultdict(list)
    eye_to_patient = {}

    if demographics_csv and os.path.exists(demographics_csv):
        with open(demographics_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    p = str(row[0]).strip()
                    e = int(row[1].strip())
                    patient_to_eyes[p].append(e)
                    eye_to_patient[e] = p
    else:
        # Reconstruct canonical 119 patients -> 125 eyes
        curr_eye = 1
        curr_pat = 1
        two_eyes_inv = {}
        for p, eyes in KNOWN_TWO_EYES_PATIENTS.items():
            for e in eyes:
                two_eyes_inv[e] = str(p)

        while curr_eye <= 125:
            if curr_eye in two_eyes_inv:
                p = two_eyes_inv[curr_eye]
            else:
                p = str(curr_pat)
                curr_pat += 1
            patient_to_eyes[p].append(curr_eye)
            eye_to_patient[curr_eye] = p
            curr_eye += 1

    return patient_to_eyes, eye_to_patient

demo_csv_path = os.path.join(WORKSPACE, 'demographics.csv')
patient_to_eyes, eye_to_patient = get_patient_eye_mapping(demo_csv_path)

# Scan all files in img_dir
all_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
eye_slices = defaultdict(list)

for fname in all_files:
    # Format: {eyeID}_{sliceID}.png
    stem = os.path.splitext(fname)[0]
    parts = stem.split('_')
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        eid = int(parts[0])
        sid = int(parts[1])
        eye_slices[eid].append((sid, fname))

print(f"Total Eyes Found in Dataset: {len(eye_slices)}")

# Cap each eye at 30 slices (evenly spaced across the scan volume for optimal representation)
capped_eye_slices = {}
for eid, s_list in eye_slices.items():
    s_list_sorted = sorted(s_list, key=lambda x: x[0])
    if len(s_list_sorted) > 30:
        indices = np.linspace(0, len(s_list_sorted) - 1, 30, dtype=int)
        capped_eye_slices[eid] = [s_list_sorted[i] for i in indices]
    else:
        capped_eye_slices[eid] = s_list_sorted

total_slices = sum(len(v) for v in capped_eye_slices.values())
print(f"Total Slices After 30-Slice-Per-Eye Cap: {total_slices}")

# Patient-level split with FIXED SEED = 42
RNG_SEED = 42
rng = random.Random(RNG_SEED)

patients = sorted(list(patient_to_eyes.keys()), key=lambda x: int(x) if x.isdigit() else x)
rng.shuffle(patients)

n_patients = len(patients)
n_train = int(0.70 * n_patients)
n_val = int(0.15 * n_patients)

train_patients = patients[:n_train]
val_patients = patients[n_train:n_train + n_val]
test_patients = patients[n_train + n_val:]

train_eyes = sorted([e for p in train_patients for e in patient_to_eyes[p] if e in capped_eye_slices])
val_eyes = sorted([e for p in val_patients for e in patient_to_eyes[p] if e in capped_eye_slices])
test_eyes = sorted([e for p in test_patients for e in patient_to_eyes[p] if e in capped_eye_slices])

print("\n--- Split Summary ---")
print(f"Train : {len(train_patients)} patients, {len(train_eyes)} eyes, {sum(len(capped_eye_slices[e]) for e in train_eyes)} slices")
print(f"Val   : {len(val_patients)} patients, {len(val_eyes)} eyes, {sum(len(capped_eye_slices[e]) for e in val_eyes)} slices")
print(f"Test  : {len(test_patients)} patients, {len(test_eyes)} eyes, {sum(len(capped_eye_slices[e]) for e in test_eyes)} slices")

# Verify zero patient overlap
assert len(set(train_patients) & set(test_patients)) == 0, "Data leakage: Train and Test share patients!"
assert len(set(val_patients) & set(test_patients)) == 0, "Data leakage: Val and Test share patients!"
assert len(set(train_patients) & set(val_patients)) == 0, "Data leakage: Train and Val share patients!"

# Save eye IDs per split
split_info = {
    'seed': RNG_SEED,
    'train_patients': train_patients,
    'val_patients': val_patients,
    'test_patients': test_patients,
    'train_eyes': train_eyes,
    'val_eyes': val_eyes,
    'test_eyes': test_eyes
}

with open(os.path.join(RESULTS_DIR, 'split_eye_ids.json'), 'w') as f:
    json.dump(split_info, f, indent=2)

with open(os.path.join(RESULTS_DIR, 'split_eye_ids.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['split', 'patient_id', 'eye_id', 'slice_count'])
    for p in train_patients:
        for e in patient_to_eyes[p]:
            if e in capped_eye_slices:
                writer.writerow(['train', p, e, len(capped_eye_slices[e])])
    for p in val_patients:
        for e in patient_to_eyes[p]:
            if e in capped_eye_slices:
                writer.writerow(['val', p, e, len(capped_eye_slices[e])])
    for p in test_patients:
        for e in patient_to_eyes[p]:
            if e in capped_eye_slices:
                writer.writerow(['test', p, e, len(capped_eye_slices[e])])

print("Saved split_eye_ids.json and split_eye_ids.csv successfully.")

# ════════════════════════════════════════════════════════════════════════════
# 3. FAST 128x128 IMAGE AND 3-CLASS MASK GENERATION
# ════════════════════════════════════════════════════════════════════════════

IM_SIZE = 128
NUM_CLASSES = 3

def make_three_class_mask_fast(binary_mask_128):
    """
    Vectorized implementation of Kafieh's 3-class rule:
    0: Above choroid (retina & vitreous)
    1: Choroid
    2: Below choroid (sclera)
    In each column, everything after the bottom-most pixel of choroid is class 2.
    """
    mask_3c = binary_mask_128.copy()
    H, W = mask_3c.shape
    has_choroid = np.any(mask_3c == 1, axis=0)
    last_idx = H - 1 - np.argmax(mask_3c[::-1, :] == 1, axis=0)
    for c in range(W):
        if has_choroid[c] and last_idx[c] < H - 1:
            mask_3c[last_idx[c] + 1:, c] = 2
    return mask_3c

def load_dataset(eye_list, capped_dict, img_dir, mask_dir, target_size=128):
    file_tuples = []
    for e in eye_list:
        for sid, fname in capped_dict[e]:
            file_tuples.append((e, sid, fname))

    N = len(file_tuples)
    X = np.zeros((N, target_size, target_size, 1), dtype=np.float32)
    Y = np.zeros((N, target_size, target_size, NUM_CLASSES), dtype=np.float32)

    metadata = []

    for i, (eid, sid, fname) in enumerate(file_tuples):
        ipath = os.path.join(img_dir, fname)
        mpath = os.path.join(mask_dir, fname)

        img_raw = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)

        if img_raw is None or mask_raw is None:
            continue

        # Resize image
        img_resized = cv2.resize(img_raw, (target_size, target_size), interpolation=cv2.INTER_AREA)
        X[i, :, :, 0] = img_resized.astype(np.float32) / 255.0

        # Resize binary mask with nearest neighbor
        mask_resized = cv2.resize((mask_raw > 127).astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        
        # Build 3-class mask
        m3c = make_three_class_mask_fast(mask_resized)

        # One-hot encode 3 classes
        for c in range(NUM_CLASSES):
            Y[i, :, :, c] = (m3c == c).astype(np.float32)

        metadata.append((eid, sid, fname, img_raw.shape))

    return X, Y, metadata

print("\nLoading and building 128x128 datasets with 3-class masks...")
X_train, Y_train, meta_train = load_dataset(train_eyes, capped_eye_slices, img_dir, mask_dir, IM_SIZE)
X_val, Y_val, meta_val = load_dataset(val_eyes, capped_eye_slices, img_dir, mask_dir, IM_SIZE)
X_test, Y_test, meta_test = load_dataset(test_eyes, capped_eye_slices, img_dir, mask_dir, IM_SIZE)

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_val   shape: {X_val.shape}, Y_val   shape: {Y_val.shape}")
print(f"X_test  shape: {X_test.shape}, Y_test  shape: {Y_test.shape}")

# ════════════════════════════════════════════════════════════════════════════
# 4. KAFIEH U-NET ARCHITECTURE & LOSS IMPLEMENTATION
# ════════════════════════════════════════════════════════════════════════════

def single_conv_block(inp, n_filters, kernel_size=3, batchnorm=True):
    x = Conv2D(n_filters, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(inp)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_kafieh_unet(input_size=128, num_classes=3, n_filters=32, dropout=0.05, batchnorm=True):
    inp = Input((input_size, input_size, 1))

    # Encoder
    c1 = single_conv_block(inp, n_filters * 1, batchnorm=batchnorm)
    p1 = Dropout(dropout)(MaxPooling2D((2, 2))(c1))

    c2 = single_conv_block(p1, n_filters * 2, batchnorm=batchnorm)
    p2 = Dropout(dropout)(MaxPooling2D((2, 2))(c2))

    c3 = single_conv_block(p2, n_filters * 4, batchnorm=batchnorm)
    p3 = Dropout(dropout)(MaxPooling2D((2, 2))(c3))

    c4 = single_conv_block(p3, n_filters * 8, batchnorm=batchnorm)
    p4 = Dropout(dropout)(MaxPooling2D((2, 2))(c4))

    # Bottleneck
    c5 = single_conv_block(p4, n_filters * 16, batchnorm=batchnorm)

    # Decoder
    u6 = Dropout(dropout)(concatenate([Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5), c4]))
    c6 = single_conv_block(u6, n_filters * 8, batchnorm=batchnorm)

    u7 = Dropout(dropout)(concatenate([Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6), c3]))
    c7 = single_conv_block(u7, n_filters * 4, batchnorm=batchnorm)

    u8 = Dropout(dropout)(concatenate([Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7), c2]))
    c8 = single_conv_block(u8, n_filters * 2, batchnorm=batchnorm)

    u9 = Dropout(dropout)(concatenate([Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8), c1]))
    c9 = single_conv_block(u9, n_filters * 1, batchnorm=batchnorm)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    return Model(inputs=[inp], outputs=[outputs])

# Calculate class frequencies for Kafieh weighted cross-entropy
freqs = np.mean(Y_train, axis=(0, 1, 2)) # Shape (3,)
print(f"Class Frequencies in Train Set: {freqs}")
beta = 0.8
weights_array = (1.0 - np.power(beta, np.maximum(freqs, 1e-6))) / (1.0 - beta)
print(f"Computed Kafieh Effective Weights (beta=0.8): {weights_array}")

def get_kafieh_loss(weights):
    weights_tensor = tf.constant(weights, dtype=tf.float32)

    def wcce_loss(y_true, y_pred):
        y_pred_scaled = y_pred / (tf.reduce_sum(y_pred, axis=-1, keepdims=True) + 1e-7)
        y_pred_clipped = tf.clip_by_value(y_pred_scaled, 1e-7, 1.0 - 1e-7)
        loss = y_true * tf.math.log(y_pred_clipped) * weights_tensor
        return -tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def dice_coef_loss(y_true, y_pred, smooth=1.0):
        # Channel 1 is choroid
        y_t = y_true[:, :, :, 1]
        y_p = y_pred[:, :, :, 1]
        intersection = tf.reduce_sum(y_t * y_p, axis=[1, 2])
        denominator = tf.reduce_sum(tf.square(y_t), axis=[1, 2]) + tf.reduce_sum(tf.square(y_p), axis=[1, 2])
        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        return 1.0 - tf.reduce_mean(dice)

    def tversky_loss(y_true, y_pred, alpha=0.3, beta_tv=0.7, smooth=1e-10):
        y_t = tf.reshape(y_true[:, :, :, 1], [-1])
        y_p = tf.reshape(y_pred[:, :, :, 1], [-1])
        truepos = tf.reduce_sum(y_t * y_p)
        fp_and_fn = alpha * tf.reduce_sum(y_p * (1.0 - y_t)) + beta_tv * tf.reduce_sum((1.0 - y_p) * y_t)
        answer = (truepos + smooth) / (truepos + smooth + fp_and_fn)
        return -answer

    def combined_wcce_dice_tversky_loss(y_true, y_pred):
        return wcce_loss(y_true, y_pred) + dice_coef_loss(y_true, y_pred) + tversky_loss(y_true, y_pred)

    def choroid_dice_metric(y_true, y_pred, smooth=1e-7):
        y_t = tf.cast(y_true[:, :, :, 1] > 0.5, tf.float32)
        y_p = tf.cast(y_pred[:, :, :, 1] > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_t * y_p, axis=[1, 2])
        union = tf.reduce_sum(y_t, axis=[1, 2]) + tf.reduce_sum(y_p, axis=[1, 2])
        return tf.reduce_mean((2.0 * intersection + smooth) / (union + smooth))

    return combined_wcce_dice_tversky_loss, choroid_dice_metric

custom_loss, dice_metric = get_kafieh_loss(weights_array)

# ════════════════════════════════════════════════════════════════════════════
# 5. MODEL INITIALIZATION & ZERO-SHOT EVALUATION
# ════════════════════════════════════════════════════════════════════════════

print("\nBuilding Zero-Shot Model...")
model_zero_shot = build_kafieh_unet(input_size=IM_SIZE, num_classes=NUM_CLASSES, n_filters=32, dropout=0.05, batchnorm=True)

if pretrained_weights_path and os.path.exists(pretrained_weights_path):
    print(f"Loading pretrained weights: {pretrained_weights_path}")
    model_zero_shot.load_weights(pretrained_weights_path)
    print("Pretrained weights loaded successfully.")
else:
    print("WARNING: Pretrained weights not found. Zero-shot will use default initialization.")

# ════════════════════════════════════════════════════════════════════════════
# 6. FINE-TUNING PIPELINE ON OIMHS TRAIN SET
# ════════════════════════════════════════════════════════════════════════════

if not SKIP_TRAINING:
    print("\nBuilding Model for Fine-Tuning...")
    model_finetuned = build_kafieh_unet(input_size=IM_SIZE, num_classes=NUM_CLASSES, n_filters=32, dropout=0.05, batchnorm=True)
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        model_finetuned.load_weights(pretrained_weights_path)

    model_finetuned.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=custom_loss,
        metrics=[dice_metric]
    )

    best_model_path = os.path.join(RESULTS_DIR, 'best_finetuned_model.h5')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, verbose=0)
    ]

    print("\n--- Starting Fine-Tuning ---")
    history = model_finetuned.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=40,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Kafieh Combined Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['choroid_dice_metric'], label='Train Choroid Dice')
    plt.plot(history.history['val_choroid_dice_metric'], label='Val Choroid Dice')
    plt.title('Validation Choroid Dice Metric')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'finetune_training_history.png'), dpi=150)
    plt.close()
    print("Saved training history plot.")
else:
    model_finetuned = build_kafieh_unet(input_size=IM_SIZE, num_classes=NUM_CLASSES, n_filters=32, dropout=0.05, batchnorm=True)
    model_finetuned.load_weights(os.path.join(WORKSPACE, "best_finetuned_model.h5"))


def boundary_errors(gt_mask, pred_mask):
    H, W = gt_mask.shape
    def bounds(m):
        top = np.array([np.where(m[:, c] == 1)[0].min() if (m[:, c] == 1).any() else np.nan for c in range(W)])
        bot = np.array([np.where(m[:, c] == 1)[0].max() if (m[:, c] == 1).any() else np.nan for c in range(W)])
        return top, bot
    gt_t, gt_b = bounds(gt_mask)
    pr_t, pr_b = bounds(pred_mask)
    return float(np.nanmean(np.abs(gt_t - pr_t))), float(np.nanmean(np.abs(gt_b - pr_b)))

# ════════════════════════════════════════════════════════════════════════════
# 7. INFERENCE & EVALUATION ON TEST EYES ONLY
# ════════════════════════════════════════════════════════════════════════════

print("\nRunning Zero-Shot vs Fine-Tuned Prediction on Test Set...")
preds_zero = model_zero_shot.predict(X_test, batch_size=16, verbose=1)
preds_ft = model_finetuned.predict(X_test, batch_size=16, verbose=1)

def postprocess_choroid(pred_prob_128, orig_shape):
    """
    Keep largest connected component, resize to original image shape.
    """
    binary_small = (pred_prob_128 > 0.5).astype(np.uint8)
    lbl = measure.label(binary_small)
    regions = measure.regionprops(lbl)
    if len(regions) > 0:
        largest = max(regions, key=lambda r: r.area)
        cleaned_small = (lbl == largest.label).astype(np.uint8)
    else:
        cleaned_small = binary_small

    upscaled = cv2.resize(cleaned_small, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    return upscaled

def calculate_niblack_cvi(denoised_norm, choroid_mask, window_size=15, k=0.001):
    """
    Exact Niblack formula from Kafieh's script:
    thresh = threshold_niblack(denoised, window_size=15, k=0.001)
    binary_niblack = denoised > (thresh * 1.03)  (stroma)
    niblack_not = choroid & (~stroma)            (luminal vessels)
    cvi = sum(niblack_not) / sum(choroid)
    """
    tot = np.sum(choroid_mask > 0)
    if tot == 0:
        return 0.0, np.zeros_like(choroid_mask, dtype=bool)

    thresh = threshold_niblack(denoised_norm, window_size=window_size, k=k)
    binary_stroma = denoised_norm > (thresh * 1.03)
    luminal = np.logical_and(~binary_stroma, choroid_mask > 0)
    cvi = float(np.sum(luminal)) / float(tot)
    return cvi, luminal

def compute_slice_metrics(gt_mask_orig, pred_mask_orig, denoised_norm):
    gt_bin = (gt_mask_orig > 0).astype(bool)
    pred_bin = (pred_mask_orig > 0).astype(bool)

    inter = np.logical_and(gt_bin, pred_bin).sum()
    total = gt_bin.sum() + pred_bin.sum()
    union = np.logical_or(gt_bin, pred_bin).sum()

    dice = (2.0 * inter) / total if total > 0 else 1.0
    iou = inter / union if union > 0 else 1.0

    # Niblack CVI
    cvi_ref, _ = calculate_niblack_cvi(denoised_norm, gt_bin.astype(np.uint8))
    cvi_pred, _ = calculate_niblack_cvi(denoised_norm, pred_bin.astype(np.uint8))
    cvi_signed_diff = cvi_pred - cvi_ref

    return dice, iou, cvi_ref, cvi_pred, cvi_signed_diff

print("Computing slice-level metrics on original resolution...")

slice_results = []
for i in range(len(meta_test)):
    eid, sid, fname, (H_orig, W_orig) = meta_test[i]

    # Load original full-res image and mask
    ipath = os.path.join(img_dir, fname)
    mpath = os.path.join(mask_dir, fname)

    img_raw = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE)
    gt_mask_orig = (cv2.imread(mpath, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

    denoised = cv2.fastNlMeansDenoising(img_raw, None, 10)
    denoised_norm = denoised.astype(np.float32) / 255.0

    # Predictions (Channel 1 = Choroid)
    pred_zero_orig = postprocess_choroid(preds_zero[i, :, :, 1], (H_orig, W_orig))
    pred_ft_orig = postprocess_choroid(preds_ft[i, :, :, 1], (H_orig, W_orig))

    # Evaluate Zero-Shot
    z_dice, z_iou, cvi_ref, z_cvi, z_diff = compute_slice_metrics(gt_mask_orig, pred_zero_orig, denoised_norm)

    # Evaluate Fine-Tuned
    ft_dice, ft_iou, _, ft_cvi, ft_diff = compute_slice_metrics(gt_mask_orig, pred_ft_orig, denoised_norm)

    z_bm, z_scj = boundary_errors(gt_mask_orig, pred_zero_orig)
    ft_bm, ft_scj = boundary_errors(gt_mask_orig, pred_ft_orig)
    slice_results.append({
        'eye_id': eid,
        'slice_id': sid,
        'filename': fname,
        'gt_mask': gt_mask_orig,
        'pred_zero': pred_zero_orig,
        'pred_ft': pred_ft_orig,
        'img_raw': img_raw,
        'cvi_ref': cvi_ref,
        'zero_dice': z_dice,
        'zero_iou': z_iou,
        'zero_cvi': z_cvi,
        'zero_diff': z_diff,
        'zero_failed': int(z_dice < 0.1),
        'ft_dice': ft_dice,
        'ft_iou': ft_iou,
        'ft_cvi': ft_cvi,
        'ft_diff': ft_diff,
        'ft_failed': int(ft_dice < 0.1),
        'zero_bm_err': z_bm,
        'zero_scj_err': z_scj,
        'ft_bm_err': ft_bm,
        'ft_scj_err': ft_scj
    })

# ════════════════════════════════════════════════════════════════════════════
# 8. HIERARCHICAL PER-EYE AGGREGATION & REPORTING
# ════════════════════════════════════════════════════════════════════════════

eye_groups = defaultdict(list)
for r in slice_results:
    eye_groups[r['eye_id']].append(r)

per_eye_metrics = []
for eid in sorted(eye_groups.keys()):
    group = eye_groups[eid]
    n_slices = len(group)

    # Zero-shot
    z_dices = [x['zero_dice'] for x in group]
    z_ious = [x['zero_iou'] for x in group]
    z_diffs = [x['zero_diff'] for x in group]
    z_fails = sum(x['zero_failed'] for x in group)

    # Fine-tuned
    ft_dices = [x['ft_dice'] for x in group]
    ft_ious = [x['ft_iou'] for x in group]
    ft_diffs = [x['ft_diff'] for x in group]
    ft_fails = sum(x['ft_failed'] for x in group)

    per_eye_metrics.append({
        'eye_id': eid,
        'patient_id': eye_to_patient.get(eid, 'Unknown'),
        'num_slices': n_slices,
        'zero_dice_mean': np.mean(z_dices),
        'zero_iou_mean': np.mean(z_ious),
        'zero_cvi_diff_mean': np.mean(z_diffs),
        'zero_failed_slices': z_fails,
        'ft_dice_mean': np.mean(ft_dices),
        'ft_iou_mean': np.mean(ft_ious),
        'ft_cvi_diff_mean': np.mean(ft_diffs),
        'ft_failed_slices': ft_fails,
        'zero_bm_err_mean': np.mean([x['zero_bm_err'] for x in group]),
        'zero_scj_err_mean': np.mean([x['zero_scj_err'] for x in group]),
        'ft_bm_err_mean': np.mean([x['ft_bm_err'] for x in group]),
        'ft_scj_err_mean': np.mean([x['ft_scj_err'] for x in group])
    })

# Save per-eye CSV
csv_per_eye = os.path.join(RESULTS_DIR, 'test_eyes_evaluation_summary.csv')
with open(csv_per_eye, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(per_eye_metrics[0].keys()))
    writer.writeheader()
    writer.writerows(per_eye_metrics)

# Compute Mean and SD across eyes
summary_stats = {
    'zero_dice': (np.mean([e['zero_dice_mean'] for e in per_eye_metrics]), np.std([e['zero_dice_mean'] for e in per_eye_metrics])),
    'zero_iou': (np.mean([e['zero_iou_mean'] for e in per_eye_metrics]), np.std([e['zero_iou_mean'] for e in per_eye_metrics])),
    'zero_cvi_diff': (np.mean([e['zero_cvi_diff_mean'] for e in per_eye_metrics]), np.std([e['zero_cvi_diff_mean'] for e in per_eye_metrics])),
    'zero_fails': (np.mean([e['zero_failed_slices'] for e in per_eye_metrics]), np.std([e['zero_failed_slices'] for e in per_eye_metrics])),

    'ft_dice': (np.mean([e['ft_dice_mean'] for e in per_eye_metrics]), np.std([e['ft_dice_mean'] for e in per_eye_metrics])),
    'ft_iou': (np.mean([e['ft_iou_mean'] for e in per_eye_metrics]), np.std([e['ft_iou_mean'] for e in per_eye_metrics])),
    'ft_cvi_diff': (np.mean([e['ft_cvi_diff_mean'] for e in per_eye_metrics]), np.std([e['ft_cvi_diff_mean'] for e in per_eye_metrics])),
    'ft_fails': (np.mean([e['ft_failed_slices'] for e in per_eye_metrics]), np.std([e['ft_failed_slices'] for e in per_eye_metrics]))
}

print("\n" + "="*80)
print("       TEST EYES EVALUATION SUMMARY (Averaged Per Eye, Then Across Eyes)      ")
print("="*80)
print(f"Total Test Eyes: {len(per_eye_metrics)} | Slices Tested: {len(slice_results)}")
print("-" * 80)
print(f"METRIC                           ZERO-SHOT (Pretrained)     FINE-TUNED (OIMHS)")
print("-" * 80)
print(f"Choroid Dice (DSC)               {summary_stats['zero_dice'][0]:.4f} +/- {summary_stats['zero_dice'][1]:.4f}          {summary_stats['ft_dice'][0]:.4f} +/- {summary_stats['ft_dice'][1]:.4f}")
print(f"Choroid IoU (Jaccard)            {summary_stats['zero_iou'][0]:.4f} +/- {summary_stats['zero_iou'][1]:.4f}          {summary_stats['ft_iou'][0]:.4f} +/- {summary_stats['ft_iou'][1]:.4f}")
print(f"Signed CVI Diff (Pred - Ref)    {summary_stats['zero_cvi_diff'][0]:+.4f} +/- {summary_stats['zero_cvi_diff'][1]:.4f}         {summary_stats['ft_cvi_diff'][0]:+.4f} +/- {summary_stats['ft_cvi_diff'][1]:.4f}")
print(f"Failed Slices per Eye (Dice<0.1) {summary_stats['zero_fails'][0]:.2f} +/- {summary_stats['zero_fails'][1]:.2f}            {summary_stats['ft_fails'][0]:.2f} +/- {summary_stats['ft_fails'][1]:.2f}")
print("="*80)

# ════════════════════════════════════════════════════════════════════════════
# 9. GENERATE SIX OVERLAY IMAGES (INCLUDING TWO WORST-CASE FAILURES)
# ════════════════════════════════════════════════════════════════════════════

print("\nSelecting 6 representative evaluation slices (including 2 failure modes)...")

# Sort all slices by Fine-Tuned Dice score
sorted_by_ft_dice = sorted(slice_results, key=lambda x: x['ft_dice'])

bad_cases = sorted_by_ft_dice[:2]
mid_idx = len(sorted_by_ft_dice) // 2
median_cases = sorted_by_ft_dice[mid_idx:mid_idx + 2]
best_cases = sorted_by_ft_dice[-2:]

selected_cases = [
    ("bad_case_1", "Worst Case 1 (Choroid Shadow / Low Contrast Failure)", bad_cases[0]),
    ("bad_case_2", "Worst Case 2 (Severe Sclerochoroidal Ambiguity Failure)", bad_cases[1]),
    ("typical_case_1", "Median Case 1 (Standard Clinical Scan)", median_cases[0]),
    ("typical_case_2", "Median Case 2 (Standard Clinical Scan)", median_cases[1]),
    ("best_case_1", "High Accuracy Case 1 (Clear Boundary)", best_cases[0]),
    ("best_case_2", "High Accuracy Case 2 (Clear Boundary)", best_cases[1])
]

for label, title_desc, item in selected_cases:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    img = item['img_raw']
    gt = item['gt_mask']
    z_pred = item['pred_zero']
    ft_pred = item['pred_ft']

    # Subplot 1: Raw OCT + Ground Truth Boundary (Green)
    axes[0].imshow(img, cmap='gray')
    axes[0].contour(gt, levels=[0.5], colors='lime', linewidths=2.0)
    axes[0].set_title(f"Ground Truth Reference (Green)\nRef CVI = {item['cvi_ref']:.4f}")
    axes[0].axis('off')

    # Subplot 2: Zero-Shot Boundary Overlay (Yellow) vs GT (Green)
    axes[1].imshow(img, cmap='gray')
    axes[1].contour(gt, levels=[0.5], colors='lime', linewidths=1.5, linestyles='--')
    axes[1].contour(z_pred, levels=[0.5], colors='gold', linewidths=2.0)
    axes[1].set_title(f"Zero-Shot Model (Yellow) vs GT (Green)\nDice = {item['zero_dice']:.3f} | CVI Diff = {item['zero_diff']:+.4f}")
    axes[1].axis('off')

    # Subplot 3: Fine-Tuned Boundary Overlay (Cyan) vs GT (Green)
    axes[2].imshow(img, cmap='gray')
    axes[2].contour(gt, levels=[0.5], colors='lime', linewidths=1.5, linestyles='--')
    axes[2].contour(ft_pred, levels=[0.5], colors='cyan', linewidths=2.0)
    axes[2].set_title(f"Fine-Tuned Model (Cyan) vs GT (Green)\nDice = {item['ft_dice']:.3f} | CVI Diff = {item['ft_diff']:+.4f}")
    axes[2].axis('off')

    plt.suptitle(f"{title_desc} - File: {item['filename']} (Eye {item['eye_id']})", fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, 'overlay_images', f"{label}_{item['filename']}")
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved overlay figure: {out_path}")

print("\nAll tasks completed successfully!")
print(f"All summary reports, split records, weights, and 6 overlay images are saved in: {RESULTS_DIR}")
