# `eval_dehaze.py`

## 功能

按实验和数据集预设执行去雾推理。脚本自动解析 config、Stage checkpoint、输入、
输出和大图分块参数；已有且尺寸正确的 PNG 会跳过，可原命令断点续跑。

## 必填参数

- `--experiment {lolv1,lolv2real,sice}`：实验身份；
- `--stage`：`measure_dehaze.py` 共用的评测预设。

```bash
python eval_dehaze.py \
  --experiment lolv2real \
  --stage stage3-real
```

结果自动写入：

```text
runs/lolv2real/results/dehaze/stage3/rtts/
```

I-HAZE 和 HSTS Real-world 预设默认使用 1024 tile 和 128 overlap。CUDA tile
使用 AMP；普通小图保持 FP32，大图或 FP32 OOM 时只对该图启用 AMP。

## 覆盖参数

- `--config`、`--checkpoint`、`--input`、`--output`：覆盖预设路径；
- `--tile-size`、`--tile-overlap`：覆盖分块；
- `--amp-pixel-threshold`：整图 AMP 像素阈值；
- `--device`：设备；
- `--overwrite`：重算已有有效结果。

设置 tile 后，整图在 CPU 保存，GPU 每次只处理一个 tile。所有 tile 共用全局
大气光，重叠区渐变融合，输出尺寸与输入一致。

