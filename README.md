# MK-UNet Reproduction — Polyp Segmentation (CVC-ClinicDB)

**Team:**
- Abdullah Siraj Khan — 31684
- Tayyaba Eman — 30802
- Amna — 30826

**Paper:** Rahman & Marculescu, "MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation" (arXiv:2509.18493, 2025)
**Official repo:** https://github.com/SLDGroup/MK-UNet

This is our reproduction of MK-UNet for Milestone 2 (Stage 2 + 3). We trained and evaluated it on CVC-ClinicDB (polyp segmentation), one of the six datasets used in the paper.

---

## 1. What we actually did

The paper tests on 6 datasets across 4 tasks. Given our compute budget (Kaggle's free GPU quota), we picked one: **CVC-ClinicDB**. We trained the standard MK-UNet config (channels `[16,32,64,96,160]`, kernel sizes `[1,3,5]`) for the full 200 epochs described in the paper.

One thing worth being upfront about: the authors run each dataset 5 times and report the mean ± std to smooth out random seed variance. We ran it once (Run 1 of their 5-run loop) — running all 5 would've meant ~1000 epochs, which wasn't realistic on our quota.

We asked our TA directly whether all 6 datasets were required and whether we could use the authors' code as-is:

> "You have to judge your compute decisions on your own, if it can be run on all 6 datasets with the resources available to you, then def go for it. If there's some extra GPU/memory needed you can choose to limit."

Given ClinicDB alone took ~1.5 hours for one run, running all 6 datasets x 5 seeds each wasn't realistic on a single free Kaggle GPU session, so we scoped down to one dataset, one run, and put our effort into understanding and verifying that one result properly instead of spreading thin across six.

---

## 2. Repository structure

```
.
├── mkunet_network.py         # model definition (MKIR, MKDC, MKIRA, GAG) — from official repo, unchanged
├── train_polyp.py            # training loop — from official repo, unchanged
├── test_polyp.py              # evaluation script — adapted for CPU-mapped inference
├── ml-proj.ipynb              # notebook used to run/inspect training and results
├── mkunet_architecture.png    # architecture diagram (paper's, for reference)
├── mkunet_project.zip         # project archive
├── polyp_data.zip             # ClinicDB dataset (zipped)
├── utils/                     # dataloader + metrics/param/FLOP counting — from official repo
├── models/                    # saved model checkpoints (e.g. best.pth)
├── log/
│   └── train_log_ClinicDB_MK_UNet_bs8_lr0.0005_e200_augTrue_run1_t185950.log
├── results_polyp/             # per-run evaluation results
├── predictions_polyp/
│   └── ClinicDB/test/         # predicted segmentation masks on the test set
├── All_Runs_Summary_Polyp.xlsx  # aggregated run summary
├── assests/
│   ├── 1.png                  # benchmark comparison chart
│   ├── 2.png                  # training convergence curves
│   └── 3.png                  # per-image Dice distribution
├── manim/                     # Manim animations of core architecture blocks — to be added
├── requirements.txt
├── LICENSE
└── README.md
```

> Update this tree if your actual folder names differ — this reflects what's referenced elsewhere in this README.

---

## 3. Background — what came before MK-UNet

Before getting into what we built, a quick summary of the landscape the paper is responding to (this is our own summary from reading the paper's related work, not copied from it):

- **CNN encoder-decoders** (U-Net, U-Net++, Attention U-Net, PraNet, DeepLabv3+) were the standard approach for years. They work well but get computationally heavy, especially once attention mechanisms are added, which makes them a poor fit for point-of-care or edge deployment.
- **Vision Transformers** (TransUNet, SwinUNet, MedT) came next, using self-attention to capture long-range relationships across the image. The tradeoff is that they tend to lose track of fine local detail and are even more expensive to run than the CNNs they were meant to improve on.
- **Lightweight CNN/MLP hybrids** (UNeXt, CMUNeXt, EGE-UNet, Rolling-UNet) tried to close that gap, but the paper points out they were mostly built and tuned for "easier" segmentation targets (skin lesions, simple cell structures) and their accuracy drops on harder, more variable targets like polyps.

MK-UNet's angle is that none of these fully solve both problems (accuracy *and* efficiency) at once. Its core idea is combining depth-wise convolutions (cheap) with a **multi-kernel** trick — running several kernel sizes in parallel so the model can capture both fine detail and broader context — plus lightweight attention (channel, spatial, and a grouped gating mechanism) only where it's needed, in the decoder. That combination is what gets it to state-of-the-art accuracy at under a third of a million parameters.

---

## 4. Setup

| | |
|---|---|
| Hardware | Kaggle Notebooks, NVIDIA T4 GPU |
| Framework | PyTorch (paper used 1.11.0) |
| Total training time | ~1 h 32 min (5556.12 s) for 200 epochs |
| Batch size | 8 |
| Learning rate | 5e-4 (AdamW) |
| Loss | BCE + IoU/Dice composite (1:1), same as paper |

Before the full run, we did a quick sanity pass first — one batch through the model, checked the output shape matched the mask shape, and confirmed the loss came out finite — before committing the full 200-epoch run to our GPU quota.

We had to install a few extra packages the authors' repo doesn't list — none of it required touching their code, just extra dependencies for evaluation/logging:
`medpy` (for HD95), `thop` (params/FLOPs counting), `openpyxl` (exporting results to Excel), `segmentation-mask-overlay` (mask visualization).

---

## 5. Where we deviated from the paper, and why

| Choice | Paper | Us | Why |
|---|---|---|---|
| Batch size | 16 | 8 | GPU memory / quota constraints |
| Learning rate | 1e-4 | 5e-4 | Bumped up a bit to compensate for the smaller batch |
| Runs per dataset | 5 (mean ± std) | 1 | Compute budget — see above |
| Datasets used | 6 | 1 (ClinicDB) | Compute budget, per TA guidance above |
| Epochs | 200 | 200 | Same |
| Checkpoint picked by | Best val Dice | Best val Dice | Same — we did **not** pick the checkpoint by peeking at test scores (more on this below) |

We didn't touch the architecture at all — see the provenance section for exactly what was reused vs. changed.

---

## 6. Results

| Metric | Paper (Clinic, 5-run mean) | Us (Run 1, checkpoint = Epoch 103) |
|---|---|---|
| **Dice** | 93.48% | **92.90%** |
| IoU | not reported for Clinic | 87.77% |
| Sensitivity | — | 92.41% |
| Specificity | — | 99.67% |
| Precision | — | 93.92% |
| HD95 | not reported | 11.58 |
| Params | 0.316 M | 0.3156 M (315,566) |
| FLOPs | 0.314 G* | 0.619 G* |

**On the FLOPs number** — we initially thought this was just profiler/hardware noise, but that's not right, and we want to correct ourselves here rather than leave it in: the paper's Table 1 explicitly says FLOPs are reported at 256x256 input, but Section 4.2 says ClinicDB is actually trained/evaluated at 352x352. If you scale their number by the resolution difference — 0.314G x (352/256)^2 ≈ 0.594G — it lands right around our measured 0.619G. So the gap is just resolution, exactly as the paper's own footnote says, not an environment quirk. Params matching almost exactly (315,566 vs 316,000) is the real confirmation that the architecture is implemented correctly.

---

## 7. Figures

**Benchmark comparison**

![Benchmark comparison](assests/1.png)

Our single-run Dice (92.90%) lands between MK-UNet-T (91.26%, paper's tiny variant) and TransUNet (93.18%), and sits inside the paper's own stated ±1–4% std-dev band around their 93.48% mean.

**Training convergence**

![Training convergence](assests/2.png)

Test/Val Dice and IoU across all 200 epochs. Both curves climb fast in the first ~25 epochs then flatten out into a stable band, and val tracks test closely the whole way through with no overfitting drop-off. Epoch 103 is marked — that's where val Dice peaked (0.9092), and the test Dice we report (0.9290) is from that exact checkpoint.

**Per-image Dice spread on the test set**

![Per-image Dice distribution](assests/3.png)

62 test images sorted by Dice. Median is 95.4%, actually higher than the paper's reported mean. What drags the overall average down is one bad case: `575.png` scored just 25.75% Dice (HD95 blew up to 169.5). If we drop that single image, our mean Dice is ~94.0% — which is actually a touch above the paper's number. So the gap to the paper looks like it's mostly one hard image, not a systemic problem with the reproduction.

---

## 8. Understanding the architecture (visual explainers)

While going through the paper in Stage 2, we built short Manim animations to walk through the core blocks ourselves rather than just reading the equations:

- **MKIR** (Multi-Kernel Inverted Residual) — expand → multi-kernel depthwise conv → project back down
- **MKDC** (Multi-Kernel Depthwise Convolution) — the parallel 1x1/3x3/5x5 depthwise branches + channel shuffle
- **MKIRA** (Multi-Kernel Inverted Residual Attention) — channel attention → spatial attention → MKIR, used in the decoder
- **GAG** (Grouped Attention Gate) — how the skip connection and the gating signal from the decoder get merged and gated

These live in the `manim/` folder in the repo — rendering and adding the final clips here (will update this section with previews/links once done).

---

## 9. Honest discussion of the gap

- The 0.58-point Dice difference from the paper's mean is comfortably inside their own reported 1–4% std across 5 runs — and we only ran once, so some spread from their mean is expected.
- We picked our checkpoint by validation Dice, not by scanning for whichever epoch happened to score best on test — test Dice actually went a bit higher later in training (0.9449 at epoch 148), but we didn't use that number, since picking off the test set would be cheating.
- Excluding the one outlier image, our mean is slightly above the paper's — so the model itself seems to be reproducing correctly, and the residual gap is concentrated in one hard case rather than spread across the dataset.
- The FLOPs gap is explained by input resolution (352x352 vs. the 256x256 used in the paper's FLOPs table), as covered above.

---

## 10. Code provenance / academic integrity log

**Reused as-is from the official repo:**
- `mkunet_network.py` — full network (MKIR, MKDC, MKIRA, GAG blocks, encoder/decoder), untouched, to keep the architecture exactly as described.
- `train_polyp.py` — training loop, optimizer/scheduler, the BCE+IoU/Dice loss.
- `utils/dataloader_polyp.py` — dataset loading, resizing, normalization, augmentation.
- `utils/utils.py` — metric definitions (Dice, IoU, HD95, Sensitivity, Specificity) and the param/FLOP counter.

**Adapted:**
- `test_polyp.py` — swapped hardcoded `.cuda()` calls for device-agnostic / CPU-mapped loading (`torch.load(..., map_location=torch.device('cpu'))`) so we could run inference locally without a GPU.
- Training scope — ran a single 200-epoch pass (Run 1) via the existing CLI flags (`--train_path`, `--test_path`) instead of the authors' full 5-run loop, to stay within our GPU quota.
- Used the authors' own argparse flags to point at our data paths rather than editing the dataloader internals.

**Written by us:**
- The plotting/analysis scripts (Matplotlib) used to generate the three figures above.
- The Manim animations described in Section 8.

**Left unchanged, noted for transparency:**
- `.cuda()` calls in `train_polyp.py` were kept as-is for actual training (run on Kaggle's GPU), to keep the original throughput/memory behavior intact.

---

## 11. How to run it

```bash
pip install -r requirements.txt

# Unzip the dataset first
unzip polyp_data.zip -d data/

# Train
python train_polyp.py --network MK_UNet --train_path ./data/ClinicDB/train/ --test_path ./data/ClinicDB/ --batch_size 8 --lr 0.0005 --epochs 200

# Evaluate best checkpoint
python test_polyp.py --network MK_UNet --run_id . --test_path ./data
```

> Adjust the unzip target / dataset path if `polyp_data.zip` extracts to a different folder name.

---

## 12. Limitations

- We reproduced one dataset (ClinicDB) out of the paper's six, per our TA's guidance to scope to our compute budget.
- One run, not the paper's five-run average.
- Batch size and LR differ from the paper (8/5e-4 vs. 16/1e-4) because of compute limits.
- One outlier test image accounts for most of the gap to the paper's reported number — see Section 9.

---

## 13. Reference

Rahman, M. M., & Marculescu, R. MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation. arXiv:2509.18493, 2025.
