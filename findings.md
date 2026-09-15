# Findings

## Scope
- User expects six stage-4 joint checkpoints under `runs`, `runs1`, and `runs2`.
- The requested deliverable is a checkpoint recommendation with reasons, not a generated enhanced dataset.
- Target downstream data is `/home/xhx/GY/model/dataset/VisDrone2019-Dark-v3`.

## Candidates
All six files are named `latest.pth`, are distinct by SHA-256, and report Stage 4 / iteration 100,000:
1. `runs/lolv1/checkpoints/stage4_joint/latest.pth` — LOLv1, epoch 206.
2. `runs1/lolv1/checkpoints/stage4_joint/latest.pth` — LOLv1, epoch 206.
3. `runs1/lolv2real/checkpoints/stage4_joint/latest.pth` — LOLv2-Real, epoch 145.
4. `runs1/sice/checkpoints/stage4_joint/latest.pth` — SICE, epoch 100.
5. `runs2/lolv1/checkpoints/stage4_joint/latest.pth` — LOLv1 with bfloat16 AMP, epoch 206.
6. `runs2/lolv2real/checkpoints/stage4_joint/latest.pth` — LOLv2-Real with bfloat16 AMP, epoch 145.

## Evidence Policy
- Existing metrics/logs are uneven: `runs2` candidates have no adjacent metric files, while several `runs1` candidates have historical evaluations.
- Historical metrics are contextual evidence only. Ranking will come from a fresh, identical evaluation of every checkpoint.
- Full-file hashes are unique; no candidate is a byte-for-byte duplicate.

## Available Evaluation Data and Runtime
- Target-domain selection set: Dark-v3 `val`, 131 images with object annotations; no paired clean reference exists locally.
- Paired low-light benchmarks are locally available: LOL-v1 Test (15 pairs), LOL-v2 Real Test (100 pairs), and SICE Test under `image/` and `label/`.
- A single NVIDIA RTX 3090 is available and supports bfloat16.
- `eval_lowlight.py` currently resizes images whose maximum dimension exceeds 1024 and saves many intermediate features. A custom comparison path should compute metrics in memory and save only a small fixed preview set.
- Dark-v3 selection must preserve the test split for final reporting; do not rank checkpoints on test images.
- SICE Test contains 150 low-exposure inputs paired to 50 reference images using the repository's SICE naming rule.
- `pyiqa` exposes NIQE, BRISQUE, ILNIQE, MUSIQ, CLIP-IQA, MANIQA, and TOPIQ-NR; NIQE, BRISQUE, MUSIQ, and common backbone assets are already cached locally.
- NIQE successfully scored a Dark-v3 validation image. The pyiqa-specific BRISQUE cache location was empty, so its constructor attempted a network download and hit a local TLS certificate error even though an identically named weight exists in the general torch checkpoint cache. Use explicit local paths rather than retrying the download.
- BRISQUE accepts `pretrained_model_path`; MUSIQ likewise supports an explicit path and its pyiqa cache is populated. Frozen CLIP ViT-B/16 is also cached locally, so candidate inference can remain offline.
- Candidate inference is fast enough for full validation comparison: the first 960×540 image took about 0.35 s after model construction on RTX 3090. The output was finite.
- The checkpoint loader reports 302 missing tensors because Stage-4 checkpoints intentionally omit frozen `sem_net`/CLIP weights; this must be verified by prefix before bulk evaluation.
- The general-cache BRISQUE weight is not directly compatible with the installed pyiqa execution path (`gamma` was not initialized). Use one targeted compatibility check, then substitute an operational no-reference metric if needed.
- Root cause confirmed in installed pyiqa: the explicit-path branch loads support vectors but fails to assign the original BRISQUE constants. Assigning the constants documented in the same implementation makes inference work offline (test score 33.86).
- All 302 checkpoint missing keys begin with `sem_net.`; there are zero missing trainable/enhancement keys and zero unexpected keys. This is intentional because Stage-4 checkpoints omit the shared frozen CLIP prior, which is loaded separately from the cached ViT-B/16 asset.
- MUSIQ also executes successfully with its explicit cached model.

## Checkpoint Compatibility
- Every candidate has exactly 257 model tensors and 1,859,120 stored parameters with identical names/shapes.
- All floating-point checkpoint tensors are finite.
- Pairwise parameter distances confirm the six weights are materially distinct rather than serialization-only variants.

## Comparison Metrics
- Primary target-domain metrics on all 131 Dark-v3 validation images: NIQE↓, BRISQUE↓, MUSIQ↑, under/overexposed pixel rates↓, and annotated-object local contrast/detail preservation.
- Secondary full-reference checks: PSNR↑, SSIM↑, LPIPS↓ across LOL-v1, LOL-v2 Real, and SICE test pairs.
- Aggregate via metric ranks rather than adding raw values with arbitrary scales; use a fixed qualitative preview panel to reject visible color casts, halos, clipping, or amplified noise.
- Historical Stage-4 JSON files are not trustworthy for ranking because some `runs1` records point at `runs/...` prediction directories.
- Smoke-test contact sheet is correctly aligned: the same input is shown once, followed by all six candidate outputs at matching geometry. All candidates visibly brighten the scene; the sheet is suitable for the fixed-preview qualitative check in the full run.
- The first full contact sheet exposed severe black spotting on `0000111_03678_d_0000072.jpg`. Per user instruction, all exact-zero input channel values must become uint8 value 1 before inference. The first full report is superseded and cannot support the final choice.
- `0000111_03678_d_0000072.jpg` is 1920×1080 and contains 4,470,892 zero-valued channels; 1,416,804 of 2,073,600 pixels (68.33%) are zero in all three channels.
- A valid 2×2 tensor assertion confirms the shared preprocessing maps exact 0 to exactly 1/255 and preserves existing 1/255 and 1 values.
- Re-running the named image changed every candidate materially (mean absolute output change 5.49–11.32 uint8 levels per channel), confirming that zero sanitization reaches the model.

## Final Equal-Condition Result
- The corrected full evaluation completed on all 131 Dark-v3 validation images and all paired samples (LOLv1 15, LOLv2-Real 100, SICE 150) for every checkpoint.
- Selected checkpoint: `runs2/lolv1/checkpoints/stage4_joint/latest.pth` (SHA-256 `6bfe84bd977995b40a98cf8567241afc0a6b0d64bdbeafc7544dc1333be05dae`).
- It has the best Dark-v3 target-IQA mean rank, 2.00: NIQE 3.50509 (rank 1), BRISQUE 27.76255 (rank 1), MUSIQ 39.54476 (rank 4).
- It removes exact underexposure under the evaluation threshold, has a 3.504% overexposed-pixel rate, annotated-object contrast 0.12464, and annotated-object gradient 0.36321. The latter is within 0.20% of the best candidate's 0.36394.
- On its native LOLv1 paired benchmark it also gives the best PSNR (19.74963 dB) and LPIPS (0.25491); its SSIM is 0.75078.
- `runs1/lolv1` is the closest alternative and has the best cross-benchmark macro rank, but on the actual Dark-v3 selection domain it is slightly worse on all three primary IQA means and slightly more overexposed.
- `runs1/sice` has the strongest MUSIQ and lowest overexposure, but is worst on Dark-v3 NIQE/BRISQUE and reduces annotated-object contrast; the preview also shows a more dataset-specific green/yellow rendering.
- The corrected full contact sheet no longer shows the severe zero-input black spotting on the named image. No candidate produced non-finite output.
- This is a defensible enhancement-model preselection, not proof of the best final detector AP. The definitive paper result still requires the planned DETR ablation after dataset generation.

## SICE Dataset Generation
- Source: `/home/xhx/GY/model/dataset/VisDrone2019-Dark-v3`; planned destination: `/home/xhx/GY/model/dataset/VisDrone2019-Dark-v3-UniSPS-SICE` (currently absent, so no collision).
- Source split image counts are train 1,178, val 131, test 605: 1,914 JPEG images total. Each split has a matching number of per-image annotation text files; root `annotations/` contains three COCO JSON files.
- Source size is 267 MiB. The filesystem has about 60 GiB free, which is sufficient for an enhanced copy plus validation artifacts.
- Image dimensions range from 960×540 through 2000×1500; the most common is 1400×1050. The enhanced files must be restored to each source image's exact dimensions to keep all bounding-box coordinates valid.
- All files in the three image directories are JPEGs; no mixed image extension handling is required for this dataset.
- Generation will use JPEG quality 95 with chroma subsampling disabled, atomic per-image replacement, and output validation for resumability. A manifest records checkpoint SHA-256, preprocessing, size policy, counts, and elapsed time.
- The three-image smoke run passed: checkpoint hash matched the SICE weight, all outputs decoded as RGB, all retained exact source dimensions, copied COCO JSONs were byte-identical, and a second run correctly resumed without regenerating valid images.
- Visual inspection of the smoke validation output showed controlled highlights, visible pedestrians and scene structure, and no obvious zero-input black-spot artifact.
- Full generation completed without errors at 4.61 images/s. The destination now contains all requested split outputs and a completed provenance manifest; independent integrity validation remains before handoff.
- Independent validation passed for all 1,914 images: output filenames exactly match source filenames, every JPEG decodes as RGB, every output retains source dimensions, and all 1,926 metadata/annotation files are byte-identical to the source.
- Final dataset size is 782 MiB; generation took 415.27 seconds. The manifest reports no resumed/skipped images in the full run.
- Cross-split visual checks show clear people/vehicles and controlled SICE highlights. The formerly problematic `val/0000111_03678_d_0000072.jpg` has a smooth dark background without the earlier severe black spotting.
- Installed `pycocotools==2.0.11` into `/home/xhx/anaconda3/envs/ICLR` with the user's explicit authorization.
- The generated dataset passes the modified DETR VisDrone loader smoke test in ICLR for all splits: lengths 1,178/131/605, valid transformed RGB tensors, valid `[N,4]` boxes, and category labels within 0–9.

## DroneVehicle-night Checkpoint Selection
- Dataset root: `/home/xhx/GY/model/dataset/DroneVehicle-night`; official split layout is `image/{train,val,test}_img` with matching label folders and COCO JSON files under `annotations/`.
- Official validation contains 868 images and 15,214 annotations across car, truck, bus, van, and freight_car. Selection will use only this validation split; the 6,013-image official test split remains untouched.
- The user stopped the active DETR-A run before GPU evaluation. GPU compute-process inventory is empty, so the comparison will not contend with training.
- `/home/xhx/GY/detr/outputs/detr_a_dark_r50/checkpoint.pth` is a complete epoch-91 checkpoint and contains both optimizer and learning-rate scheduler state, so training can later resume at epoch 92.
- All 868 validation images exist, decode as RGB, and are exactly 840×712. All validation annotations reference valid image IDs and have positive COCO-format boxes.
- Exact-zero channels make up 1.9421% of validation pixels' channels, so the established exact-0→1 preprocessing rule remains relevant and is applied before both input scoring and enhancement.
- Validation-image mean luminance spans 115.91–188.78/255 (median 139.61); fixed previews will be selected across this observed luminance distribution rather than by filename.
- The comparison utility now auto-detects both the VisDrone `val/images` and DroneVehicle `image/val_img` layouts and compiles successfully.
- DroneVehicle files store a 640×512 scene inside an 840×712 canvas; four 100-pixel white borders account for about 45% exact-white pixels and must be excluded from inference and IQA scoring.
- Per user instruction, the comparison order is: crop the white border → replace exact-zero channels with uint8 value 1 → run UniSPS and score the cropped scene. COCO boxes must be translated by the crop offset for object-region metrics.
- The JPEG-compressed borders are visually white but not every border pixel remains exactly 255, so cropping must use the known 100-pixel geometry rather than a fragile exact-white threshold.
- 1,978 of 15,214 COCO boxes extend at least partly outside the 640×512 content crop. For object-region metrics, boxes will be translated, clipped to the content bounds, and fully non-intersecting boxes excluded instead of sampling white padding.
- Source inventory: train 10,357 images / 195,669 objects; val 868 / 15,214; test 6,013 / 112,461. Every image has a same-stem TXT file; 25 TXT files are legitimately empty and no label line is malformed.
- TXT labels use four polygon vertices, class, and difficulty. Their historical `feright_car` spelling corresponds to COCO `freight_car`; preserve this compatibility spelling in TXT output.
- Across all three splits, each TXT line maps exactly and in order to one COCO annotation: per-image counts and categories match, and every COCO bbox is exactly the min/max envelope of its TXT quadrilateral. This permits a single geometry transform to drive both output formats consistently.
- Available disk space is about 57 GiB, sufficient for a cropped derivative. The original dataset remains read-only.
- Full polygon audit finds 44,946 partially clipped objects (train 27,127; val 1,978; test 15,841), but zero fully invisible or degenerate objects. All 323,344 annotations can therefore be retained.
- A deterministic 100-image estimate puts lossless cropped PNG at about 1.93× the current padded-JPEG storage, still safely within available space. PNG is selected to preserve the source's decoded content pixels exactly and avoid adding JPEG artifacts around small vehicles.
- Added `tools/crop_dronevehicle_dataset.py`. It writes a new sibling dataset, converts images to lossless 640×512 PNG crops, synchronizes TXT and COCO geometry from one policy, supports safe resume, and records provenance.
- A full preflight over all 323,344 TXT polygons passed: 44,946 require clipping, none becomes degenerate, and every transformed coordinate lies within x=[0,640], y=[0,512].
- The real-image smoke test on validation image `00415.jpg` passed: output is RGB 640×512, is pixel-for-pixel identical to source `[y=100:612, x=100:740]`, contains no white canvas, and all four labels translate correctly.
- Full cropped dataset generation completed in 75.26 seconds at `/home/xhx/GY/model/dataset/DroneVehicle-night-cropped`; output size is 9.0 GiB.
- Independent full validation passed for 17,238 images and 323,344 objects: every PNG is RGB 640×512 and pixel-identical to the corresponding decoded source crop; all filenames, TXT/COCO counts, categories, box envelopes, areas, and coordinate ranges agree.
- Final split counts remain unchanged: train 10,357 / 195,669 objects; val 868 / 15,214; test 6,013 / 112,461. All partially visible objects were retained; none was dropped.
- The cropped train/val/test datasets all pass the actual DETR `CocoDetection` plus transform pipeline in the ICLR environment. Sample tensors and normalized boxes are valid, with category labels confined to 0–4.
- The one-image × six-checkpoint GPU smoke comparison completed successfully on the cropped validation dataset. All cached IQA assets and Stage-4 checkpoints loaded offline, every candidate produced a report row and preview, and no runtime error occurred.
- Smoke metrics are finite and the contact sheet is correctly aligned with no white border. On the first dark scene, LOLv1 candidates are visibly the brightest/noisiest, LOLv2Real candidates are more restrained, and SICE is restrained but introduces a green/yellow cast; this is only a pipeline check, not the final ranking.
- Full DroneVehicle cropped-validation evaluation completed successfully for all six candidates over all 868 official validation images. The final JSON report and seven-scene contact sheet were written without inference failures.

## DroneVehicle-night Full Comparison
- Best candidate-only IQA mean rank is `runs2_lolv2real_amp` at 2.33: NIQE 3.49894 (best), BRISQUE 21.88279 (best), MUSIQ 31.39217 (rank 5).
- `runs1_sice` ties for second by candidate-only IQA mean rank at 3.33: NIQE 4.32598 (rank 6), BRISQUE 22.34058 (rank 3), MUSIQ 34.99010 (best).
- SICE has by far the lowest enhanced overexposure rate, 0.3980%, versus 2.0979% for `runs2_lolv2real_amp` and about 3.73% for the three LOLv1 variants. The cropped input baseline is 0.2055%.
- `runs2_lolv2real_amp` raises annotated-object contrast from 0.11782 to 0.13713 and gradient from 0.30400 to 0.36698. SICE gives lower contrast 0.12082 but a slightly higher gradient 0.37760.
- The cropped input baseline itself scores BRISQUE 17.12757 and MUSIQ 33.82393; enhancement does not universally improve every no-reference metric. SICE is the only candidate that improves MUSIQ over the input, while all six worsen BRISQUE.
- Full contact-sheet inspection confirms the numeric tradeoff: LOLv1 variants brighten most aggressively and amplify noise/highlights; LOLv2Real variants are more restrained; SICE preserves highlights best but has a visible green/yellow rendering on several scenes.
- Full-resolution review of the darkest preview `01227.png` shows that all six models primarily amplify sensor noise and do not reconstruct trustworthy scene detail. LOLv1 variants amplify it most; SICE is somewhat more restrained but greener.
- On already bright `00397.png`, SICE is the most conservative and preserves highlight separation better; LOLv1 is brightest and closest to clipping, while `runs2_lolv2real_amp` is intermediate.
- Reused the earlier corrected paired-benchmark report because checkpoints and zero preprocessing are unchanged: SICE paired macro rank is 3.00 versus 4.22 for `runs2_lolv2real_amp`. These cross-domain results corroborate but do not override DroneVehicle target metrics.
- Final report integrity passed: six distinct hashes, 868 samples for every metric/candidate, seven complete preview rows, finite means/medians/stds, and no paired/test leakage. Both utilities compile, and GPU compute inventory is empty after evaluation.

## DroneVehicle runs2/LOLv2Real Generation
- Selected checkpoint: `runs2/lolv2real/checkpoints/stage4_joint/latest.pth`, SHA-256 `c688d069df1d0aeac8470c85db4004df1bdc12f64c203e6f93d994703b060777`.
- Source is the validated 640×512 lossless-PNG dataset `/home/xhx/GY/model/dataset/DroneVehicle-night-cropped`; target is `/home/xhx/GY/model/dataset/DroneVehicle-night-cropped-UniSPS-LOLv2Real-AMP`.
- The destination is absent, GPU compute inventory is empty, and about 48 GiB disk space is available before generation.
- The existing generator was VisDrone-layout-specific and would have copied DroneVehicle images as metadata. It is being generalized to detect either layout, skip the correct image directories during metadata copying, and reproduce the source layout at the destination.
- The generalized generator compiles and correctly detects both the original VisDrone and cropped DroneVehicle layouts.
- A real three-image GPU smoke generation (one image per official split) completed with the selected checkpoint at 3.82 images/s and wrote a complete limited-run manifest without errors.
- Smoke validation passed: exactly three images were generated (the 17,238 source images were not accidentally copied), each is RGB 640×512, all TXT/COCO metadata is byte-identical to the cropped source, and the validation output is pixel-identical to the same checkpoint's earlier comparison preview.
- Visual inspection of one train, val, and test output confirms border-free aligned scenes, visible vehicles, finite normal rendering, and the expected LOLv2Real enhancement character. Sensor noise is visibly amplified in the darker val/test samples, consistent with the selection warning.
- Ten representative source/output PNG ratios average 1.144× (median 1.10×, max 1.63×). The 8.75 GiB source image payload therefore predicts about 10.0 GiB output, with a conservative sample-based upper estimate of 14.3 GiB—well within the available 48 GiB.
- Full generation completed successfully for all 17,238 images in 2,735.48 seconds (45.59 minutes), averaging 6.30 images/s. No image was resumed or skipped.
- Final enhanced dataset size is 9.6 GiB. The manifest records the exact checkpoint path/hash, FP32 inference, exact-zero-to-one input policy, DroneVehicle layout, and exact split counts.
- Independent exhaustive validation passed: all 17,238 output names match source names, every image decodes as RGB 640×512, every output differs from its source (global per-pixel mean absolute change 74.413/255), and no temporary files remain.
- All 17,243 non-image source files—including TXT labels, COCO JSONs, crop provenance, and the dataset PDF—are byte-identical in the enhanced destination.
- The enhanced train/val/test roots pass the actual DETR `CocoDetection` and transformation pipeline in ICLR. Five distributed indices per split produced valid tensors, normalized boxes, and labels 0–4.
- Full-resolution visual checks on dark `01227.png`, median `00415.png`, and bright `00397.png` confirm the expected runs2/LOLv2Real appearance, no white borders, correct 640×512 geometry, and visible annotated vehicles; the known noise amplification remains apparent on the darkest scene.
- All seven fixed validation previews in the final dataset are pixel-identical to the selected `runs2_lolv2real_amp` outputs from the prior six-checkpoint report, proving the generator used the intended model and preprocessing.
- Final checkpoint hash re-verification passed, the partial manifest was removed after successful completion, the generator still compiles, and GPU compute inventory is empty.
