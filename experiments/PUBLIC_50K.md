# ALD 5 万条诗歌实验

## 数据

实际训练使用 `public-50k-verse`。最初准备的混合语料在特征缓存阶段停止，当时没有开始更新模型权重。审查发现元曲分片中有杂剧对白和舞台提示，部分散文诗切片及白话短章也接近连续叙事。当前版本剔除整批元曲、散文诗切片、普通散文、白话短章组和没有分行的作品，并对夹注、存目和明显散文长句进行保守筛查。规则也会舍弃少数带括注的真实诗作。

| 划分 | 诗歌条目 | 其中早期新诗 |
| --- | ---: | ---: |
| 训练 | 50,000 | 86 |
| 验证 | 6,947 | 11 |
| 测试 | 6,940 | 2 |

训练集有 19,186 条唐诗、19,362 条宋诗、11,128 条宋词、208 条《诗经》、30 条《楚辞》、86 条早期新诗。该配比主要检验古典诗歌扩到 5 万条后的表现。现代语体覆盖有限，验证与测试中的新诗也很少，现代诗泛化需要另做实验。自动筛查能检查来源和明显结构特征，仍有人工抽查的空间。底模的语言先验和连续前缀的利用率也会影响生成体裁。

按作者、作品 ID 和近重复分组后再划分。精确去重去掉 293 条，近似去重去掉 137 条。已全量复查白名单、来源、筛查规则及跨划分的条目 ID、作品 ID、非空作者。现代作品来源沿用上轮审核记录。古典正文来自固定提交的仓库快照。 [古典语料](https://github.com/chinese-poetry/chinese-poetry/tree/b8594f81a89752241442f2ce267d6f66f96704ee) · [来源清单](../data/raw/public-50k-verse/sources.json) · [数据审计](public-50k-verse/data-audit.json)

## 训练与恢复

真实 Qwen，单卡 BF16，解码 microbatch 4，梯度累积 4，最多 6 轮，验证 NLL 早停。自编码器每 150 次优化器更新另保存一次恢复点。两种扩散各最多 20 轮，并自动运行对照评估。各阶段由独立进程运行。

在项目目录执行下列命令可启动或恢复。Windows 上已经为本轮启动了独立后台进程。重复启动前查看阶段状态。

```bash
python -X utf8 -m scripts.run_public_training --raw data/raw/public-50k-verse --checkpoints checkpoints/public-50k-verse --experiments experiments/public-50k-verse --config configs/public_50k.yaml --conditional-config configs/public_50k_conditional.yaml --train-poems 50000 --poetry-only
```

[当前阶段](public-50k-verse/status.json) · [运行日志](public-50k-verse/run.out.log) · [自编码训练日志](public-50k-verse/autoencoder.log)
