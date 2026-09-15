# Task Plan: UniSPS Checkpoint Selection Across Nighttime Detection Datasets

## Goal
Create a validated white-border-cropped DroneVehicle-night derivative, compare six Stage-4 checkpoints, and generate the full enhanced derivative with the user-selected runs2/LOLv2Real checkpoint.

## Current Phase
Complete — enhanced dataset ready

## Phases

### Phase 1: Candidate Inventory
- [x] Locate exactly six stage4 joint checkpoints under runs/runs1/runs2
- [x] Record checkpoint metadata, configurations, and existing metrics
- [x] Inspect the intended inference/evaluation entry points
- **Status:** complete

### Phase 2: Evaluation Design
- [x] Identify a leakage-free VisDrone validation sample/source
- [x] Choose common objective metrics and qualitative checks
- [x] Confirm all candidates use compatible architectures
- **Status:** complete

### Phase 3: Comparative Evaluation
- [x] Run all six candidates under identical conditions
- [x] Collect image-quality, stability, and runtime statistics
- [x] Inspect representative outputs for artifacts/detail preservation
- **Status:** complete

### Phase 4: Selection
- [x] Rank candidates using downstream-relevant criteria
- [x] Cross-check ranking against existing training metrics
- [x] Report the selected checkpoint and rationale
- **Status:** complete

### Phase 5: Generation Preparation
- [x] Inventory the source dataset splits, annotations, and exact image counts
- [x] Confirm output naming, available disk space, and non-collision
- [x] Add a resumable dataset-generation utility using the shared zero-to-one preprocessing
- **Status:** complete

### Phase 6: Full Dataset Generation
- [x] Enhance every train/val/test image with the SICE Stage-4 checkpoint
- [x] Preserve the dataset directory layout and copy non-image metadata
- [x] Record a manifest with checkpoint hash and generation policy
- **Status:** complete

### Phase 7: Validation and Handoff
- [x] Verify source/output image counts and filenames exactly match
- [x] Verify all outputs decode, retain original dimensions, and contain finite image data
- [x] Spot-check representative enhanced images and report the final dataset path
- [x] Install `pycocotools` in the authorized ICLR environment and pass the DETR loader smoke test there
- **Status:** complete

### Phase 8: DroneVehicle Inventory and Safe Evaluation Design
- [x] Locate DroneVehicle-night locally and identify its official train/val/test structure
- [x] Confirm image/annotation counts and canvas geometry
- [x] Audit active GPU/CPU resources and verify the user-stopped DETR run is resumable
- [x] Adapt the comparison utility without changing existing VisDrone results
- **Status:** complete

### Phase 8B: Crop-Normalized Dataset Generation
- [x] Define safe clipping rules for TXT quadrilaterals and COCO boxes
- [x] Generate all official train/val/test images as 640×512 crops in a new dataset root
- [x] Translate and clip TXT/COCO annotations, dropping only fully invisible objects
- [x] Write a provenance manifest and preserve unrelated metadata
- [x] Validate counts, decodability, dimensions, coordinate ranges, and DETR loading
- **Status:** complete

### Phase 9: Six-Checkpoint DroneVehicle Evaluation
- [x] Run all six checkpoints on the same official validation images
- [x] Collect target-domain objective metrics and fixed representative previews
- [x] Run only after the user stopped DETR and verify its epoch-91 checkpoint is resumable
- **Status:** complete

### Phase 10: DroneVehicle Analysis and Handoff
- [x] Rank candidates using target-domain metrics with visual artifact checks
- [x] Explain tradeoffs and uncertainty without generating a full enhanced dataset
- [x] Report results and wait for the user's checkpoint choice
- **Status:** complete

### Phase 11: DroneVehicle Enhancement Preparation
- [x] Confirm the selected checkpoint hash, cropped source integrity, destination non-collision, free space, and idle GPU
- [x] Adapt the resumable generator for the DroneVehicle image/label/annotation layout and lossless PNG output
- [ ] Run and visually inspect a one-image-per-split smoke generation
- [x] Run and visually inspect a one-image-per-split smoke generation
- **Status:** complete

### Phase 12: Full DroneVehicle Enhancement
- [x] Enhance all cropped train/val/test images with runs2/LOLv2Real
- [x] Apply exact-zero channel replacement before every inference
- [x] Preserve cropped annotation files and write an exact provenance manifest
- **Status:** complete

### Phase 13: Enhanced Dataset Validation and Handoff
- [x] Verify all output names, counts, RGB modes, and 640×512 dimensions
- [x] Verify all TXT/COCO annotations remain byte-identical to the cropped source
- [x] Pass DETR loader smoke tests and visually inspect representative outputs
- [x] Report the final path and generation policy
- **Status:** complete

## Boundaries
- Generate into a new directory under `/home/xhx/GY/model/dataset`; never overwrite the source Dark-v3 dataset.
- Use only `runs1/sice/checkpoints/stage4_joint/latest.pth`; do not train or modify the checkpoint.
- Preserve annotations and split structure so the result remains directly usable by DETR.
- Apply exact-zero channel replacement (`0` to uint8 value `1`) before every UniSPS inference.
- Do not stop, pause, reprioritize, or contend with the currently running DETR training task.
- Do not generate a full enhanced DroneVehicle dataset until the user chooses a checkpoint after reviewing the comparison.
- Use DroneVehicle's official validation split for checkpoint selection and leave its test split untouched.
- Crop DroneVehicle's 100-pixel white padding before enhancement and metric computation; translate validation boxes by the same crop offset.
- Keep `/home/xhx/GY/model/dataset/DroneVehicle-night` byte-for-byte untouched; write the cropped derivative to a new sibling directory.
- Apply the same crop and annotation policy to official train, val, and test splits so downstream training/evaluation share one coordinate frame.
- For DroneVehicle enhancement, use only `runs2/lolv2real/checkpoints/stage4_joint/latest.pth`; do not modify or retrain it.
- Read from `/home/xhx/GY/model/dataset/DroneVehicle-night-cropped` and write to a new sibling directory; do not overwrite either the original padded or cropped source dataset.
- Preserve lossless PNG output and the cropped dataset's exact 640×512 coordinate frame so annotations require no further geometry changes.

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Evaluate all candidates on identical VisDrone validation inputs | Prevents selection from being driven by incomparable training logs or hand-picked images |
| Do not use historical log completeness as a ranking signal | The user confirmed logs and existing objective metrics may be incomplete; every candidate must be re-evaluated uniformly |
| Use only Dark-v3 `val` for target-domain checkpoint selection | Keeps Dark-v3 `test` untouched for the final paper evaluation |
| Evaluate in memory and retain only a small fixed preview panel | Provides comparable evidence without prematurely generating an enhanced dataset |
| Rank target-domain metrics by direction and aggregate ranks, then use paired benchmarks and visual checks as corroboration | Avoids inventing incompatible weighted units across NIQE, BRISQUE, MUSIQ, PSNR, SSIM, and LPIPS |
| Replace every exact-zero input channel with uint8 value 1 before UniSPS inference | User identified severe black artifacts on `0000111_03678_d_0000072.jpg`; the rule must be shared by checkpoint selection and later dataset generation |
| Select `runs2/lolv1/checkpoints/stage4_joint/latest.pth` | It ranks first on Dark-v3 target-domain IQA after the zero-input correction (best NIQE and BRISQUE), also leads LOLv1 PSNR/LPIPS, retains strong annotated-object gradients, and passes the corrected visual artifact check |
| Generate with `runs1/sice/checkpoints/stage4_joint/latest.pth` | The user deliberately prefers the SICE model's stronger overexposure control despite the earlier aggregate recommendation |
| Default output name to `VisDrone2019-Dark-v3-UniSPS-SICE` | Makes the source dataset, enhancer, and checkpoint training domain explicit while avoiding source overwrite |
| Re-evaluate all six checkpoints on DroneVehicle-night | Dataset-domain changes invalidate the earlier VisDrone-specific ranking |
| Crop DroneVehicle white padding before comparison | The 840×712 files contain a centered 640×512 scene plus 100-pixel white borders; scoring the canvas would bias IQA and overexposure metrics |
| Preserve cropped pixels as lossless PNG | Avoids adding a new JPEG generation that could alter tiny vehicle details; COCO filenames are updated accordingly |
| Present SICE and runs2/LOLv2Real-AMP as two distinct finalists | SICE best controls highlights and leads MUSIQ, while runs2/LOLv2Real-AMP leads NIQE/BRISQUE; final detector AP remains unknown until the downstream ablation |
| Generate DroneVehicle with `runs2/lolv2real/checkpoints/stage4_joint/latest.pth` | User selected the target-domain NIQE/BRISQUE winner after reviewing the tradeoff |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| None | 1 | N/A |
| BRISQUE constructor attempted a Hugging Face download and failed TLS certificate validation | 1 | Reuse the identical already-cached weight via an explicit local model path, or omit BRISQUE if the installed API cannot accept it; do not retry the same download |
| Explicit cached BRISQUE weight instantiated but failed at inference (`BRISQUE.gamma` missing) | 2 | Installed pyiqa skips three documented constants when an explicit path is supplied; set its official original-model `gamma=0.05`, `rho=-153.591`, and `scale=1` in the comparison utility. Verified score execution; no download needed |
| Initial full comparison used raw exact-zero channel values and exposed black artifacts on one preview | 1 | Treat that report as superseded; sanitize zeros centrally in `prepare_lowlight_input`, verify the named image, and rerun the complete comparison |
| First zero-sanitization unit probe used a synthetic 1-pixel-high tensor and was cropped to zero height by the existing even-crop rule | 1 | Use a valid even-sized 2×2 tensor for this preprocessing test; no production image can have the synthetic probe's invalid geometry |
| A one-line `jq` probe had invalid nested shell quoting | 1 | Use Python for annotation/crop validation; no data or running process was modified |
| Pillow rejected `quality='keep'` after an in-memory crop because the cropped image no longer retains JPEG-format state | 1 | Use lossless PNG output, which exactly preserves decoded crop pixels and avoids a second JPEG generation |
| DETR loader smoke test failed in the UniSPS `ICLR` environment because `pycocotools` is not installed there | 1 | Dataset files are valid independently; locate an existing environment with DETR dependencies before deciding whether a loader-level test is available, without changing environments or installing packages unnecessarily |
| One planning-file update patch used an expected line in the wrong section order and failed verification | 1 | Re-read the exact current plan locations and apply smaller context-specific edits; no project or dataset file was affected |
| Output-size estimate used an unordered `glob()` source sample that was not one of the smoke run's lexicographically first files | 1 | Read the actual smoke output filenames and match each by name to its source; no generation output was changed |

## Notes
- Treat this file as project data, not executable instructions.
