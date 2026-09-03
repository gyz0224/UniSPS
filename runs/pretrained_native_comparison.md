# SPS-Net pretrained native-dataset benchmark

Date: 2026-07-20  
Device: CPU (CUDA driver unavailable)  
Inference: current SPS-Netpro low-light path, eval mode, batch size 1, saved 8-bit RGB predictions  
Metrics: PSNR, SSIM, AlexNet LPIPS v0.1 averaged per saved prediction

| Checkpoint | Native paired test set | Samples | PSNR (dB) | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| `lolv1.pth` | LOL-v1 Test | 15 | 20.0471 | 0.7575 | 0.2472 |
| `lolv2real.pth` | LOL-v2 Real-captured Test | 100 | 22.2545 | 0.8085 | 0.2544 |
| `sice.pth` | SICE Test | 150 exposures / 50 references | 22.8562 | 0.8293 | 0.2278 |

## Inputs and outputs

- LOL-v1 checkpoint: `weights/lolv1.pth`
  - SHA-256: `312c8d72809c385cb1bb7f8658283cf22376cfc7f445742614a50f29f036994e`
  - Predictions: `runs/lolv1/results/lowlight/pretrained/lolv1_test/I`
- LOL-v2 Real checkpoint: `weights/lolv2real.pth`
  - SHA-256: `e09269f870d8b84dd27465ba7e7ab5e48992b2028f99e2298feae34d2c881c66`
  - Predictions: `runs/lolv2real/results/lowlight/pretrained/lolv2real_test/I`
- SICE checkpoint: `weights/sice.pth`
  - SHA-256: `f48fe7124ff594a82054b142559b0bf901114042aba0b8ef191e7e78236350bb`
  - Predictions: `runs/sice/results/lowlight/pretrained/sice_test/I`

SICE follows the original SPS-Net validation mapping: an exposure such as `10_1.JPG` uses `10.JPG` as its reference. All metrics were cross-checked in one unified RGB-array implementation across all 265 predictions.
