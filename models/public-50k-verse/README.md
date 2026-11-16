# 诗云 · 5 万首诗歌模型

这组权重来自 `public-50k-verse` 实验。训练集 50,000 首，验证集 6,947 首，测试集 6,940 首。训练语料主要是唐诗、宋诗和宋词，其中早期新诗 86 首。详见 [语料与训练记录](../../experiments/PUBLIC_50K.md)。

| 文件 | 训练轮次 | 验证损失 | 用途 |
| --- | ---: | ---: | --- |
| `ae/best.pt` | 6 | 2.7762 | 投影器、连续前缀和解码器 LoRA，用于直接解码及扩散生成 |
| `ddim/best.pt` | 20 | 0.4771 | 无条件 DDIM 和源文本引导 SDEdit |
| `conditional/best.pt` | 20 | 0.3720 | 自监督多视图条件扩散 |

自编码器的损失是 token 平均负对数似然，两种扩散的损失是各自的速度预测均方误差。使用 [public_50k.yaml](../../configs/public_50k.yaml) 和 [public_50k_conditional.yaml](../../configs/public_50k_conditional.yaml) 在 RTX 5070 Ti Laptop GPU 上以 BF16 训练。自编码器第 4 轮曾因中断从第 3 轮权重重新开始，优化器及随机状态重新初始化。

## 本机运行

在仓库根目录按 [安装说明](../../README.md#安装与环境检查) 准备 Python 环境。首次推理会下载冻结的 [Qwen3 Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3) 和 [Qwen3 Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd)，两者均采用 Apache 2.0 许可。

```bash
python -m scripts.generate --ae models/public-50k-verse/ae/best.pt --diffusion models/public-50k-verse/ddim/best.pt --conditional models/public-50k-verse/conditional/best.pt --mode sdedit --text "窗外正在下雨。" --seed 71 --strength 0.5 --steps 24 --device cpu
```

切换 `--mode direct`、`--mode conditional` 或 `--mode unconditional` 比较结果。网页默认使用 SDEdit，展开「实验对照」可查看直接解码候选。试多视图条件扩散时，在扩散权重列表里选择 `conditional/best.pt`，并切换到「多视图条件扩散」。

```bash
python app.py --checkpoints models/public-50k-verse --device cpu
```

网页监听 `http://127.0.0.1:7860`。

## 已验证的表现与局限

自动评估使用 9 条输入和 2 个随机种子。固定种子、打乱输入后，SDEdit 与多视图条件扩散各有 18/18 对输出发生变化。这个实验测到的是输出对输入的敏感度。输入与诗的审美关联、扩散相对直接解码及检索的收益，仍需要人工盲评。早期新诗样本少，现代诗表现尚待单独验证。

## 文件校验

```text
b7b5f97da65c303cf699c0bd82ba1ef8fba3d8bb4b584c014ff63b8a7dc2707f  ae/best.pt
c37b06c2d2e0511f8cbb67bba3f0aae92a6a88284549d2c8e0d373f7ed944caa  ddim/best.pt
ff4e15157fc4d62e5a80e7223bb45368db628253263697fde513b4be1919e569  conditional/best.pt
```

两种扩散权重都绑定上表的自编码器 SHA-256。请整组使用。文件仅含推理所需的训练参数、配置和归一化统计，适合直接加载生成。
