# Medical Image Analysis (MIA) Course Project

## Automated Choroid Vascularity Index (CVI) on the OIMHS Dataset

A student re-implementation and domain-shift study of a published OCT choroid-segmentation pipeline, followed by fine-tuning on a public macular-hole dataset.

| Field | Details |
|---|---|
| **Students** | P. Sharmi |
| | Kshitij Shende |
| | C Pothan Sai Reddy |
| | Saisha Bore |
| **Roll numbers** | 23bcs090 |
| | 23bcs075 |
| | 23bds017 |
| | 23bcs116 |
| **Course** | Medical Image Analysis |
| **Baseline paper** | Roya Arian, T. Mahmoudi, H. Riazi-Esfahani, et al. *Automatic Choroid Vascularity Index Calculation in Optical Coherence Tomography Images with Low-Contrast Sclerochoroidal Junction Using Deep Learning.* Photonics 2023, 10, 234. [DOI: 10.3390/photonics10030234](https://doi.org/10.3390/photonics10030234) |
| **External dataset** | OIMHS (OCT Image Macular Hole Segmentation), 125 eyes, 3,859 B-scans. [DOI: 10.1038/s41597-023-02675-1](https://doi.org/10.1038/s41597-023-02675-1) |

This is a **course project**, not a clinically validated tool. Segmentation and CVI numbers were **not reviewed by an ophthalmologist**.

---

## 1. What this project does

1. Explains and re-uses the paper’s lightweight 3-class U-Net, combined Kafieh loss, Non-Local Means denoising, and Niblack CVI formula.
2. Runs the authors’ released **raster** weights **zero-shot** on OIMHS (macular hole OCT, a different disease and scanner domain than the paper).
3. **Fine-tunes** the same architecture on a patient-level OIMHS split and compares zero-shot vs fine-tuned choroid Dice, IoU, and CVI error on held-out test eyes.

CVI is defined exactly as in the paper:

\[
\mathrm{CVI} = \frac{\text{luminal (vessel) area}}{\text{total choroidal area}}
\]

inside the choroid mask, after Niblack binarization of stroma vs lumen.

---

## 2. What is inside this submission zip

This folder is the **course-portal zip**. It contains code, both weight files, Phase 1 zero-shot outputs (`oimhs_cvi_final_results/`), and Phase 2 fine-tune outputs (`results/`). It does **not** contain the OIMHS images/masks (too large). Put those on Drive and paste the link below.

**Google Drive (dataset + optional extras):**  https://drive.google.com/file/d/13tyEXg9_7ZDvbvvqSXlcRCSYiiiYPUG9/view?usp=sharing 
Drive link includes: processed OIMHS `images/` + `masks/` (and `demographics.csv`in a zip).

```text
MIA_Project_Submission/
├── README.md
├── requirements.txt
├── oimhs_cvi_pipeline.py              # Phase 1: zero-shot on OIMHS
├── oimhs_cvi_pipeline_boundary.py     # Phase 1 variant: adds BM/SCJ boundary errors
├── finetune_oimhs_pipeline.py         # Phase 2: fine-tune + test-eye benchmark
├── finetune_oimhs_pipeline_boundary.py # Phase 2 variant: adds BM/SCJ boundary errors
├── saved model for raster data.h5     # Paper raster weights (zero-shot start)
├── best_finetuned_model.h5            # Fine-tuned weights from Colab
├── oimhs_cvi_final_results/           # Phase 1: oimhs_cvi_pipeline.py on 3,859 slices
│   ├── cvi_full_evaluation.csv        # Per-slice Dice, IoU, CVI_gt, CVI_pred
│   ├── dice_score_distribution.png
│   ├── cvi_gt_vs_pred_scatter.png
│   └── visualizations/                # Periodic overlay panels from the run
└── results/                           # Phase 2: finetune_oimhs_pipeline.py
    ├── test_eyes_evaluation_summary.csv
    ├── split_eye_ids.json
    ├── split_eye_ids.csv
    ├── finetune_training_history.png
    └── overlay_images/
        ├── bad_case_1_123_19.png
        ├── bad_case_2_116_69.png
        ├── typical_case_1_110_6.png
        ├── typical_case_2_98_33.png
        ├── best_case_1_30_9.png
        └── best_case_2_30_7.png
```

| File | Role |
|---|---|
| `oimhs_cvi_pipeline.py` | Zero-shot inference + GT metrics on matched OIMHS B-scans (256 input) |
| `oimhs_cvi_pipeline_boundary.py` | Same as above plus BM/SCJ boundary-error metrics |
| `finetune_oimhs_pipeline.py` | Patient-level split, 128 fine-tune, zero-shot vs fine-tuned on test eyes |
| `finetune_oimhs_pipeline_boundary.py` | Same as above plus BM/SCJ boundary-error metrics |
| `saved model for raster data.h5` | Authors’ released raster U-Net (Arian et al.) |
| `best_finetuned_model.h5` | Our OIMHS fine-tuned weights |
| `oimhs_cvi_final_results/cvi_full_evaluation.csv` | Per-slice Phase 1 metrics (table 5A) |
| `oimhs_cvi_final_results/*.png` | Dice histogram and CVI_gt vs CVI_pred scatter |
| `oimhs_cvi_final_results/visualizations/` | Overlay figures sampled during the 3,859-slice run |
| `results/test_eyes_evaluation_summary.csv` | Per-eye Dice, IoU, CVI error (table 5B) |
| `results/split_eye_ids.*` | Train/val/test patient–eye assignment, seed 42 |
| `results/overlay_images/` | Worst / median / best qualitative panels |

---

## 3. Environment

```bash
pip install -r requirements.txt
```

GPU is strongly recommended (the reported fine-tune used a Colab Tesla T4). TensorFlow 2.16+ ships Keras 3; this code uses **`tf_keras`** so the authors’ legacy `.h5` files still load.

Place (or extract) `images/` and `masks/` so the scripts can find them, and keep `saved model for raster data.h5` on disk. Both pipelines search common local and `/content` paths.

**Zero-shot evaluation (all matched OIMHS slices):**

```bash
python oimhs_cvi_pipeline.py
```

**Fine-tune + test-eye benchmark:**

```bash
python finetune_oimhs_pipeline.py
```

To **reproduce** (not required to read the results already in `results/`): download OIMHS `images/` and `masks/` next to the scripts (or under `OIMHS_dataset_clean/proc/`), keep `saved model for raster data.h5` in this folder, then run the commands above. GPU recommended. The scripts also accept `/content/...` paths on Colab.

---

## 4. Method

### 4.1 Phase 1 — Zero-shot pipeline (`oimhs_cvi_pipeline.py`)

Runs the authors' released raster U-Net weights on all matched OIMHS B-scans without training.

- Input size 256 (to better match 512×512 processed scans); 3-class output (above/choroid/below).
- Post-processing: largest connected component of choroid class, resized to original B-scan.
- CVI via fastNlMeansDenoising + Niblack (window 15, k 0.001) inside the mask; luminal pixels = choroid and not stroma.
- Slice-level metrics: Dice, IoU, CVI_gt, CVI_pred, MAE, Pearson r.

### 4.2 Phase 2 — Zero-shot vs fine-tuned (`finetune_oimhs_pipeline.py`)

Patient-level split (seed 42, both eyes of a patient stay together; at most 30 evenly spaced slices per eye), 128×128 input.

- Architecture: same lightweight U-Net (one Conv2D + BatchNorm + ReLU per stage, 4 encoder/decoder levels, skip connections, 3 classes).
- Loss: weighted categorical cross-entropy + Dice + Tversky on the choroid channel (Kafieh combination), Adam 1e-4, early stopping, ReduceLROnPlateau, 40 epochs max, batch 16.
- Fine-tuned from the same raster weights; both models evaluated on held-out test eyes.
- Metrics: per-eye Dice, IoU, signed CVI error, failed slices (Dice < 0.1); reported as mean ± SD across 20 test eyes.

| Split | Patients | Eyes | Slices |
|---|---|---|---|
| Train | 83 | 87 | 2,023 |
| Val | 17 | 18 | 428 |
| Test | 19 | 20 | 479 |

---

## 5. Results we actually obtained

### A. Zero-shot paper weights on all 3,859 OIMHS B-scans

From `MIA_Project_Final.ipynb` / `oimhs_cvi_pipeline.py` (slice-level, not per-eye).

| Metric | Value |
|---|---|
| B-scans | 3,859 |
| Mean Dice (DSC) | 0.8032 ± 0.1415 |
| Median Dice | 0.8448 |
| Mean IoU | 0.6879 ± 0.1450 |
| BM boundary error (px) | 12.07 ± 20.45 |
| SCJ boundary error (px) | 13.51 ± 25.99 |
| Mean CVI (GT mask) | 0.7323 ± 0.0598 |
| Mean CVI (predicted mask) | 0.6676 ± 0.0568 |
| CVI MAE | 0.0705 |
| Pearson *r* (CVI_gt vs CVI_pred) | 0.6917 |

### B. Held-out test eyes: zero-shot vs fine-tuned

From `MIA_Project_Final_2_.ipynb` / `finetune_oimhs_pipeline.py`. Metrics are **mean ± SD across 20 test eyes** (eye mean first, then across eyes). Input 128×128.

| Metric | Zero-shot (paper raster weights) | Fine-tuned on OIMHS |
|---|---|---|
| Choroid Dice | 0.6048 ± 0.1243 | **0.9531 ± 0.0159** |
| Choroid IoU | 0.4482 ± 0.1191 | **0.9111 ± 0.0280** |
| Signed CVI error (pred − ref) | −0.0392 ± 0.0491 | **−0.0131 ± 0.0072** |
| Failed slices / eye (Dice < 0.1) | 0.00 ± 0.00 | 0.00 ± 0.00 |
| BM boundary error (px) | 28.93 ± 8.99 | **1.56 ± 0.23** |
| SCJ boundary error (px) | 44.18 ± 32.00 | **3.02 ± 0.81** |

Fine-tuning recovered choroid overlap and reduced CVI bias on this test split. Zero-shot Dice in **A** vs **B** is not the same experiment (full set vs test eyes only; 256 vs 128; slice vs per-eye aggregation). Do not mix the two tables as if they were one number.

---

## 6. Limitations

1. **Not expert-verified.** No ophthalmologist checked our masks, CVI values, or failure cases.
2. **OIMHS is a macular-hole dataset**, not a choroid-CVI study. The "choroid GT" masks are project-processed, not a multi-rater gold standard.
3. **No from-scratch retraining.** We loaded the authors' `.h5` weights and adapted them.
4. **Domain shift.** Paper trained on diabetic/pachychoroid OCT; OIMHS is macular hole. Zero-shot CVI underestimates GT CVI on OIMHS.
5. **CVI is unsupervised.** Vessel labels come from Niblack thresholding, not clinicians; CVI inherits mask and thresholding errors.
6. **Student constraints.** 128×128, 30 slices/eye, one seed/split, no cross-validation or external test.
7. **Not a medical device.** For course report only; not for diagnosis or treatment.

---

## 7. What is intentionally not in this zip

| Left out | Why |
|---|---|
| OIMHS `images/` and `masks/` (~thousands of PNGs) | Portal size; host on Drive |
| Authors’ original `OCT-CVI-DeepLearning-main/` scripts | Optional; we used their raster `.h5` |
| Colab notebooks | Same logic as the two `.py` files |

If the marker asks for a rerun, they only need this zip + Drive images/masks.

---

## 8. References

1. Arian R, Mahmoudi T, Riazi-Esfahani H, et al. Photonics. 2023;10(3):234.
2. Ye X, et al. OIMHS dataset. Scientific Data. 2023. DOI: 10.1038/s41597-023-02675-1.
3. Original paper code folder in this repo: `OCT-CVI-DeepLearning-main/` (Arian et al.).
