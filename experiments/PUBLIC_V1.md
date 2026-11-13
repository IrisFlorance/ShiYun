# ALD 首轮公开语料实验

本轮使用固定版本的真实 Qwen 权重，研究诗歌 latent 能否重建、扩散及受普通文本引导。实验使用真实诗作与散文片段，另保留固定探针用于盲评。语料中的文体只用于划分和分组报告，训练目标仍来自文本自身。

## 来源与划分

- 古典诗词曲来自 chinese-poetry 固定提交，取《全唐诗》、宋词、元曲及《诗经》正文。原作属于公有领域，整理仓库的 MIT 许可和所用分片哈希均有记录。 [固定提交](https://github.com/chinese-poetry/chinese-poetry/tree/b8594f81a89752241442f2ce267d6f66f96704ee)
- 早期新诗、散文诗和普通散文来自中文维基文库。逐页记录固定修订号和内容哈希。独立诗页以页面公有领域标注为准；作品集子篇还校验作者，篇章按同一作品分组。训练只用诗歌域。 [新诗目录](https://zh.wikisource.org/wiki/Category:%E6%96%B0%E8%A9%A9) · [野草](https://zh.wikisource.org/wiki/%E9%87%8E%E8%89%B8) · [背影](https://zh.wikisource.org/wiki/%E8%83%8C%E5%BD%B1_(%E6%95%A3%E6%96%87%E9%9B%86))
- 古典诗占多数，现代部分偏早期白话，普通散文也以早期文学文本为主。固定日常探针可以检验现象，现代口语的广泛覆盖需更丰富的已获许可语料。

原始快照和 [来源清单](../data/raw/public-v1/sources.json) 保存在 `data/raw/public-v1`。[划分清单](../checkpoints/public-v1/prepared/manifest.json) 包含精确和近似去重统计、各文体样本数与文件哈希。按作者、作品及近重复连通组分割，匿名作品以作品标识为主。

本轮导入 1,829 条诗歌，去重后保留 1,827 条。其中新诗原始条目 213 条，散文诗片段 76 条。三个诗歌划分为 1,458 / 202 / 167 条，实际检查未发现跨划分的相同条目、作品 ID 或非空作者。现代训练条目集中于杨华和鲁迅，现代验证使用徐志摩，测试包括闻一多、刘半农和戴望舒。388 条散文因作者分组均落在训练划分的文件内，训练命令只选择诗歌，独立日常探针另存。

[数据审计](public-v1/data-audit.json) 也记录了 tokenizer 截断。解码上限为 160 token，含 EOS。训练诗歌有 5 条截断，验证新诗有 25 / 48 条截断，测试新诗有 3 / 13 条截断。验证 NLL 对应截断后的正文。

## 训练配置

- 单卡 BF16，编码上限 192 token，解码上限 160 token。冻结编码器，训练投影器、连续前缀和秩 16 的 LoRA。使用梯度检查点，microbatch 4，累积 4。
- 自编码器最多 6 轮，验证 NLL 早停耐心 3 轮。无条件与多视图扩散各最多 40 轮，早停耐心 8 轮，batch 48。自编码验证额外报告各文体的 token NLL。
- 完成各阶段后跑既有 A–G 对照，按两个种子生成候选和盲评表。人类评分留空，待实际评分后解盲。

首轮自编码已完成 92 次优化器更新，训练 token NLL 为 4.4344，验证 token NLL 为 4.3973。PyTorch 进程的已分配和已保留峰值分别为 2,381.63 和 5,378 MiB。已确认 `best.pt`、`last.pt` 可读。

## 完整运行结果

2026-11-13 03:22 完成自动对照评估，从数据划分开始总计约 20 分钟。自编码器跑完 6 轮，共 552 次优化器更新。第 4 轮总体验证 NLL 最低，为 4.3234，作为两种扩散的输入。第 6 轮训练 NLL 继续降至 3.7540，验证 NLL 回升到 4.3953。新诗验证 NLL 最低出现在第 2 轮，为 4.3932。

无条件和多视图扩散各跑完 40 轮，验证 velocity MSE 分别从 0.7263 降至 0.3996、从 0.6896 降至 0.2991。固定种子时，18 对 SDEdit 输出均随条件打乱而变化。最佳自编码 checkpoint 的两样本交换 NLL 差值为 +0.0031，清零差值为 +0.2824。生成仍常呈散文式片段，输入「窗外正在下雨。」的 SDEdit 样例出现了跑动、坠落和天地破灭的叙述。未进行人工盲评，跨主题审美关联和相对于基线的优势尚待验证。

[自动评估报告](public-v1/evaluation/report.json) · [生成样本](public-v1/evaluation/generations.jsonl) · [人工盲评表](public-v1/evaluation/blind_rating.csv)

## 启动与恢复

```bash
python -m pip install -e ".[corpus]"
python -X utf8 -m scripts.collect_public_corpus --out data/raw/public-v1
python -X utf8 -m scripts.run_public_training
```

重跑训练命令会跳过已完成的阶段，从最近完整 epoch 继续中断的训练。[当前状态](public-v1/status.json) · [自编码日志](public-v1/autoencoder.log) · [自编码指标](../checkpoints/public-v1/ae/metrics.jsonl)。修改数据或模型配置时用新的原始语料快照、checkpoint 目录和实验目录。
