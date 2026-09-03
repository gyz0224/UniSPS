# `scripts/validate_real_dataset.py`

## 功能

全量检查 D4+ 真实室外训练数据：扫描 clear/hazy 与各自天空 mask，拒绝重复 basename、缺失或多余 mask、损坏图像以及图像/mask 尺寸不一致。默认还核对当前下载归档的 3,577 个 clean 对和 2,902 个 hazy 对。

脚本复用 `datasets/dehaze.py` 的规范校验实现，训练 loader 与预检不会维护两套 mask 匹配规则。

## 默认目录

```text
dataset/real/
├── clear/
├── hazy/
└── masks/
    ├── clear/
    └── hazy/
```

直接运行：

```bash
python scripts/validate_real_dataset.py
```

成功输出：

```text
Validated D4+ real dataset: clean=3577 pairs, hazy=2902 pairs
```

## 参数

- `--clean`、`--hazy`：两个图像域目录或 flist；
- `--clean-masks`、`--hazy-masks`：对应 mask 目录或 flist；
- `--expected-clean`、`--expected-hazy`：期望数量，默认 3577/2902；传 0 可只检查结构而不限制数量。

D4+ 天空 mask 用于训练时排除天空歧义，不应作为语义分割标注使用。
