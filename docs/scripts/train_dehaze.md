# `train_dehaze.py`

## 功能

执行 Stage 1–3 无配对去雾训练。`--experiment` 同时决定低光初始权重、
前一阶段 checkpoint 和当前输出目录，防止 LOL-v1、LOLv2-real、SICE 互相覆盖。

## 自动路径

| Stage | 初始化 | 输出目录 |
|---|---|---|
| 1 | `weights/<experiment>.pth` | `runs/<experiment>/checkpoints/stage1_warmup` |
| 2 | Stage 1 `latest.pth` | `runs/<experiment>/checkpoints/stage2_its` |
| 3 | Stage 2 `latest.pth` | `runs/<experiment>/checkpoints/stage3_real` |

新训练若目标目录已有 `.pth` 会拒绝启动。继续同一 stage 必须使用 `--resume`。

## 参数

- `--experiment {lolv1,lolv2real,sice}`：必填实验身份；
- `--config`：必填 ITS 或真实室外 YAML；
- `--stage {1,2,3}`：训练阶段；
- `--resume`：恢复同一 stage 的模型、optimizer 和 scheduler；
- `--max-iterations`、`--device`、`--log-interval`：运行覆盖；
- `--pretrained-sps`、`--init-checkpoint`、`--output`：消融实验覆盖路径。

Stage 1 只接受 `--pretrained-sps`，Stage 2/3 只接受
`--init-checkpoint`。正常完整流程不需要传这些覆盖参数。

## 命令

```bash
python train_dehaze.py \
  --experiment lolv2real \
  --config configs/dehaze_its.yaml \
  --stage 1 \
  --max-iterations 10000

python train_dehaze.py \
  --experiment lolv2real \
  --config configs/dehaze_its.yaml \
  --stage 2 \
  --max-iterations 150000

python train_dehaze.py \
  --experiment lolv2real \
  --config configs/dehaze_real.yaml \
  --stage 3 \
  --max-iterations 100000
```

日志在第一次 iteration、每个 `log_interval` 和最后一次打印速度、ETA、损失与
学习率。Stage 3 启动前运行 `python scripts/validate_real_dataset.py`。

