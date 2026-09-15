# Findings & Decisions

## Requirements
- Add NIQE to tests/evaluation of supervised low-light datasets.
- Add LPIPS to every supervised/paired dehazing dataset evaluation.

## Research Findings
- Supervised low-light test metrics are produced by `measure.py` through `metrics/lpips_evaluator.py`; current outputs are PSNR, SSIM, and LPIPS.
- `measure.py` serves all native paired low-light datasets selected by `ExperimentLayout` (LOLv1, LOLv2-Real, and SICE).
- A NIQE implementation is already available indirectly through `pyiqa` in unpaired validation/checkpoint-comparison code, but there is no shared directory-level NIQE helper yet.
- Paired dehazing metrics are computed by `metrics.dehaze_evaluator.evaluate_dehaze_directory`; LPIPS is already optional via `lpips_device`.
- `measure_dehaze.resolve_evaluation` currently enables LPIPS only when `preset.lpips` is true. Existing presets mark HSTS Synthetic only, so SOTS Indoor, SOTS Outdoor, and I-HAZE omit LPIPS despite being paired datasets.
- Real HSTS/RTTS presets have no reference and correctly use FADE; if a real preset is explicitly given a clean reference, it becomes full-reference and should get LPIPS under the requested rule.
- Existing user changes include `eval_lowlight.py` zero-channel sanitization and checkpoint/filesystem changes; this task must not overwrite them.
- `pyiqa==0.1.14.1` is already pinned in `requirements.txt`; existing UniSPS code constructs NIQE with `pyiqa.create_metric("niqe", device=device)` and passes `[0,1]` image tensors.
- Native paired dehazing datasets are SOTS Indoor, SOTS Outdoor, HSTS Synthetic, and I-HAZE at their applicable stages. HSTS Real-world and RTTS remain no-reference FADE unless a caller supplies `--reference`.
- The real installed pyiqa NIQE model constructs and scores locally in the ICLR environment; no additional dependency or network fetch is needed.
- The installed AlexNet LPIPS model also runs locally. A SOTS Indoor `stage2` smoke evaluation now emits a finite `lpips` field, confirming the newly enabled non-HSTS path.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Reuse the existing `pyiqa` dependency for NIQE | Avoids introducing a second, potentially inconsistent NIQE implementation |
| Enable dehaze LPIPS based on resolved reference presence | This exactly follows paired/supervised status and covers custom references |
| Compute low-light NIQE on each saved prediction at its native output size | NIQE is no-reference; resizing solely to match GT would make it depend on reference geometry |
| Add a new four-metric low-light evaluator while retaining the existing three-value LPIPS evaluator wrapper | Adds NIQE without unnecessarily breaking external callers of the existing helper |
| Keep paired-training checkpoint selection unchanged | The request concerns dataset test/measurement; adding a fourth best-checkpoint criterion would change training semantics beyond scope |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Initial combined source dump was truncated | Re-read targeted line ranges and use focused searches |
| First multi-file replacement patch was invalid | Use one targeted update per planning file |
| Findings patch expected a slightly different template heading | Re-read the generated template and patch its exact content |

## Resources
- `measure.py`
- `metrics/lpips_evaluator.py`
- `measure_dehaze.py`
- `metrics/dehaze_evaluator.py`
- `training/experiment.py`
