# 诗云 · ShiYun

Aesthetic Latent Diffusion（ALD）中文诗歌生成研究项目。

ALD 研究一个可以实测的问题。给定一段普通文字，能否用未标注诗歌学到一个连续空间，经由扩散生成一首新诗，同时保留原文中某种可被读者辨认的文学感受？系统使用本地模型、连续条件和自回归解码，推理时用户只提供源文本。

[本机验证记录](experiments/VERIFICATION.md) · [研究笔记](RESEARCH_NOTES.md)

首轮公开语料实验的来源、配置和恢复命令见 [公开语料训练记录](experiments/PUBLIC_V1.md)。

5 万条训练样本的实验使用按行整理的诗歌，语料构成、筛查及续训方式见 [5 万条诗歌训练记录](experiments/PUBLIC_50K.md)。

## 架构

```text
普通文本 / 诗歌
    → 冻结的 Qwen3-Embedding-0.6B，提取逐 token 隐状态
    → 带 attention mask 的查询池化投影器，得到 [B, 8, 256]
    → 直接解码 / 无条件 DDIM / SDEdit / 实验性多视图条件 DDIM
    → 连续向量映射成 Qwen3-0.6B-Base 的软前缀
    → 带 KV cache 的自回归文本生成
```

自编码阶段使用连续确定性瓶颈，先训练投影器、软前缀和解码器 LoRA。重建采用 teacher forcing，前缀最后一位预测正文首 token，右侧 padding 不计损失，EOS 参与训练。训练可调节 latent 噪声、方差正则和瓶颈容量。投影器的推理和扩散导出使用同一套 FP32 计算。扩散训练采用 cosine 噪声日程及 velocity 预测，验证同时报告换算后的 epsilon MSE。每个位置与通道的归一化统计只从训练诗歌拟合，随扩散 checkpoint 保存，采样后逆变换回解码器空间。

无条件扩散建模诗歌 latent 分布。SDEdit 从被加噪的源 latent 出发，由同一个无条件模型去噪。强度为 0 时直接返回源 latent。多视图条件扩散用去排版后的随机连续裁剪作为条件，用完整原诗作目标，带条件丢弃，可在 CLI 调节 CFG。诗歌视图上学到的条件对应关系能否泛化到普通文本仍待验证。第一版将非配对领域对齐留作后续实验。

模型版本固定在配置中。默认缓存为逐 token 隐状态，普通检索对照使用同一编码器的末有效 token 归一化向量。[Qwen3 Embedding 模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) · [Qwen3 Base 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B-Base)

## 安装与环境检查

推荐 Python 3.11，在独立环境中安装。Windows PowerShell 用 `.\.venv\Scripts\Activate.ps1`，Ubuntu 用 `source .venv/bin/activate`。RTX 5070 Ti 需要带对应架构支持的 PyTorch CUDA 构建。本机实测版本为 PyTorch 2.7.0+cu128，CUDA runtime 12.8。[PyTorch 安装版本表](https://pytorch.org/get-started/previous-versions/)

```bash
python -m venv .venv
# 激活环境后执行
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m scripts.check_env --out experiments/environment.json
```

只跑 CPU mock 时，可改用 PyTorch CPU 源，然后执行 `python -m pip install -e . pytest==9.1.1`。UMAP 是可选功能，用 `python -m pip install -e ".[eval]"` 启用。环境报告同时记录驱动、CUDA wheel、设备能力、可用显存、BF16 实际矩阵运算及依赖版本。环境中若混装了旧版 Conda 数值库，优先换用独立环境。Windows 若从管道传入中文，使用 UTF-8，例如 `python -X utf8 -m scripts.generate ...`。

首次运行 Qwen 阶段时会将两组公开权重下载到 Hugging Face 缓存。模型文件完整缓存后可设置 `HF_HUB_OFFLINE=1`，进行离线训练与推理。所有训练与评估命令都在项目根目录执行。

## 先跑 CPU 验证

```bash
python -m pytest -q -m "not integration"
python -m scripts.smoke --out checkpoints/my-mock
python app.py --ae checkpoints/my-mock/ae/best.pt --diffusion checkpoints/my-mock/ddim/best.pt
```

`smoke` 在指定目录依次导入语料、缓存特征、训练自编码器和两种扩散、生成候选并输出对照报告。使用新的输出目录开始新实验。内置语料由项目编写，训练诗歌仅 18 首。mock 编码器和字符级 GRU 用于测试可训练性及数据流，生成文本只用于工程检查。启用真实权重集成测试时，将环境变量 `ALD_QWEN_TEST` 设为 `1`，然后运行 `python -m pytest -q -s -m integration`。

## 数据与分阶段训练

- JSONL 每行一条作品，必填字符串 `text`，可选 `work_id`、`author`。多片段同作使用同一个全局作品 ID。对长文可自行切窗，保留其共同作品 ID。
- TXT 默认为一文件一作品，多作品使用单独一行 `===` 分隔，空行保留为分节。诗歌与普通文本从不同输入参数传入。训练只使用诗歌域，普通文本用于域偏移观察。
- 预处理进行 NFC、空白规范化、长度过滤、精确去重和基于四字符片段 Jaccard 的近似去重。作品、近重复及已标明作者的连通分组整体划分。缺失作者元数据会限制泄漏检查。少数作者占据大部分语料时，调整数据来源以获得独立划分。只有在实验允许同一作者跨集合时才使用 `--work-only`。
- 数据目录包含 split 哈希、统计 JSON 和长度直方图 SVG。特征缓存记录语料版本、编码权重版本、tokenizer 指纹、截断、精度和视图配置。逐 token 特征需要磁盘空间，估算单条上限为 `长度 × 隐维 × 每元素字节数`，加视图还要计入一份特征。

```bash
python -m scripts.prepare_data --poems data/raw/poems.jsonl --prose data/raw/prose.jsonl --out data/prepared/corpus
python -m scripts.cache_embeddings --config configs/low_vram.yaml --data data/prepared/corpus --out data/cache/corpus --views
python -m scripts.train_autoencoder --config configs/low_vram.yaml --cache data/cache/corpus --out checkpoints/corpus/ae
python -m scripts.train_diffusion --config configs/low_vram.yaml --cache data/cache/corpus --ae checkpoints/corpus/ae/best.pt --out checkpoints/corpus/ddim
python -m scripts.train_diffusion --config configs/conditional_low_vram.yaml --cache data/cache/corpus --ae checkpoints/corpus/ae/best.pt --out checkpoints/corpus/conditional
```

只训练方案 A 时可省略 `--views`。真实模型工程复现可用内置样例和 `configs/qwen_smoke.yaml`、`configs/qwen_smoke_conditional.yaml`，它们刻意缩短训练。正式实验应准备有使用许可且体裁多样的独立语料。更改编码器或输入截断时使用新特征缓存。更改 latent token 数、宽度或维度时重训自编码器并重新导出 latent。

每个阶段保存 `best.pt`、`last.pt` 和 `metrics.jsonl`。带 `--resume` 重复原命令即可从最后完整 epoch 继续，可以增加 `epochs`。断点包含优化器、梯度缩放器、随机状态及已解析配置。扩散绑定自编码器文件哈希，重训后用新 checkpoint 配对。自编码日志包含正确、交换与清零 latent 的 NLL 对照、生成重建、活跃维度和方差。扩散日志包含三种噪声区间的误差与样本分布距离。近零或负的交换 NLL 差值值得核查，输出变化也需结合对照解释。

## 推理与 WebUI

```bash
python -m scripts.generate --ae checkpoints/corpus/ae/best.pt --diffusion checkpoints/corpus/ddim/best.pt --text "窗外正在下雨。" --seed 71 --count 3 --strength 0.5 --steps 40 --temperature 0.8 --top-p 0.95 --max-tokens 96
python app.py --ae checkpoints/corpus/ae/best.pt --diffusion checkpoints/corpus/ddim/best.pt
```

在 [本地 WebUI](http://127.0.0.1:7860) 可以选择同一自编码器配套的扩散 checkpoint、调节采样参数、查看连续向量热图和余弦，并用相同种子生成直接解码对照。CLI 的 `--mode direct`、`unconditional`、`sdedit` 和 `conditional` 对应四条路径。条件模型用 `--conditional` 提供 checkpoint。CLI 的 `--out` 保存参数、候选和完整 latent。源文本超过配置上限时会截断。

## 单卡与显存

`configs/default.yaml` 采用 8 个 256 维 latent token、256 编码 token、192 解码 token。低显存配置将上下文缩短至 160 / 128，保持 microbatch 1，开启解码器梯度检查点和梯度累积。支持 BF16、FP16 回退及梯度缩放。特征缓存、训练和扩散阶段分别执行。导出诗歌 latent 时仅加载投影器，推理时编码器和解码器轮流驻留 GPU。日志中的 allocated / reserved 峰值是 PyTorch 进程统计值，桌面及其他进程的用量另看环境报告。

OOM 时先减小编码或训练 batch，再减短上下文。自编码器仍不足时减少 latent token，并用新配置重训。扩散阶段可独立减小 batch 和去噪器宽度。确认已开启梯度检查点、已按阶段退出进程，再评估与本机环境兼容的卸载或量化方案。能保留原超参数时使用 `--resume`，改模型形状、训练 batch 或累积策略时用新的实验目录。建议在正式训练前跑完一轮并为桌面负载留约 1 GiB 余量。

## 对照与人工评价

```bash
python -m scripts.evaluate --ae checkpoints/corpus/ae/best.pt --diffusion checkpoints/corpus/ddim/best.pt --conditional checkpoints/corpus/conditional/best.pt --cache data/cache/corpus --inputs data/mock/probes.jsonl --out experiments/corpus --seeds 71 72 73 --strengths 0.2 0.5 0.8
```

报告包含普通 embedding 检索、直接解码、无条件扩散、SDEdit、打乱输入后 SDEdit、强度扫描，以及可选的多视图条件模型与对应打乱控制。各生成组复用种子和解码参数。无条件组每个种子复用同一输出，跨输入的重复是对照设计。输出包含候选 JSONL、可选 UMAP、PCA、域分布统计、输入扰动、插值及词面重合率。多视图组另在保留诗歌上比较正确视图与打乱视图，并分别记录可见及被裁掉部分的四字符片段召回。

将 `blind_rating.csv` 交给评分者，`blind_key.jsonl` 由实验者保管。每个候选单独评 1–5 分，分别评价通顺、意象、感受关联、疑似复述。疑似复述得分越高表示越接近复述。建议让多位评分者独立完成后解盲，报告对照的配对差值、置信区间与评审分歧。语义余弦、输出变化或词面重合指标只能解释对应的测量。项目没有外部 LLM 评分调用。

## 局限与下一步

小语料运行可以验证梯度、检查点、潜在条件和采样。文学生成质量、跨域泛化、抗复述及相对于检索和直接解码的收益，仍需多语料、多种子与人工盲评。近重复处理依赖词面相似，未标注的改写与译本需要额外的语料审查。研究顺序为先稳定重建与潜在利用率，再比较 SDEdit 与多视图，最后视实际偏移和盲评结果引入非配对领域对齐。
