# 本机验证记录

日期为 2026-11-13。所有结果来自当前工程内运行。训练数据为自编小样本，24 首诗划分为 18 / 3 / 3，6 条普通文本划分为 4 / 1 / 1。固定评估集另含 9 条普通文本。真实模型短训采用固定的 Qwen3 权重版本、BF16 和 `configs/qwen_smoke.yaml`。

## 工程阶段

| 里程碑 | 已运行的检查 |
| --- | --- |
| M0 | RTX 5070 Ti Laptop，12,227 MiB，驱动 596.49，Python 3.11.5，PyTorch 2.7.0+cu128。实际 BF16 矩阵运算通过。[环境报告](environment.json) |
| M1–M2 | 11 项 CPU 自动化测试中的自编码测试通过。真实 Qwen 最大长度 256 / 192，验证了冻结、非零梯度、连续条件响应和 KV cache 一致性。CPU 自编码断点续训与连续训练的参数逐项一致。 |
| M3–M4 | CPU 与真实 Qwen latent 均运行了无条件训练、DDIM 采样、中文 SDEdit 与解码。测试确认强度 0 等于直接解码，非零强度改变 latent。 |
| M5 | CPU 与真实 Qwen latent 均训练了多视图扩散。完成真实权重正确视图与错配视图对照。 |
| M6 | CPU 与真实权重各执行了 9 条探针、两个生成种子、9 个对照变体，每种后端各 153 条候选。另对 3 首测试诗的正确视图和打乱视图各做一次条件解码。PCA 和 CPU UMAP 已产出。Gradio 的本地生成接口对两种后端均返回了候选、直接解码对照、两个热图及模型信息。 |

## 实际 CUDA 峰值

以下数据均为 PyTorch 进程峰值，单位 MiB。

| 任务 | 已分配峰值 | 已保留峰值 |
| --- | ---: | ---: |
| Qwen 编码与视图缓存 | 1,146.96 | 1,162 |
| Qwen 自编码训练，低显存短训 | 1,269.48 | 1,468 |
| Qwen 最大上下文前向 / 反向及 KV cache 集成测试 | 1,964.20 | 2,044 |
| 无条件扩散训练 | 59.00 | 68 |
| 多视图扩散训练 | 60.29 | 68 |
| 真实权重端到端对照评估 | 1,186.39 | 1,194 |
| Gradio 真实权重生成与直接解码对照 | 1,177.57 | 1,184 |

## 可解释的观测

- Qwen 自编码短训一轮，验证 NLL 为 7.8913。将两条验证样本的 latent 交换后，平均 token NLL 增加 0.2317，清零后增加 0.8298。三条验证诗歌上，按样本方差大于 0.0001 的位置通道为 2,042 / 2,048。样本量很小，交换结果仅覆盖两首。
- 无条件扩散两轮的验证 velocity MSE 为 1.3253 → 1.3024。末轮在标准化空间中，生成 latent 的整体标准差为 1.271，验证 latent 为 1.077。去噪与采样已执行，是否拟合好目标分布仍需扩大语料与对照。
- 在这次真实权重对照中，固定输入和种子时，SDEdit 与直接解码的 18 对输出全部不同，SDEdit 与打乱条件的 18 对输出也全部不同。真实权重候选仍有断裂与重复，无法据此确认审美关联或优于检索。未执行人工评分。
- CPU mock 自编码第 5 轮验证 NLL 为 5.2145，交换 NLL 差值约为 -0.0006。它用于验证实现，训练后的 mock 条件利用率在该小样本上未显示正向收益。

## 产物

- [真实权重 SDEdit 输出](qwen-generation.json)
- [真实权重对照报告](qwen-evaluation/report.json) · [盲评表](qwen-evaluation/blind_rating.csv) · [PCA](qwen-evaluation/pca.svg)
- [CPU 对照报告](mock-final/report.json) · [UMAP](mock-final/umap.svg)
- [真实权重自编码训练日志](../checkpoints/verified-qwen/ae/metrics.jsonl) · [无条件扩散日志](../checkpoints/verified-qwen/ddim/metrics.jsonl) · [条件扩散日志](../checkpoints/verified-qwen/conditional/metrics.jsonl)

本次验证覆盖 Windows 和上述单卡。大规模语料的收敛、长时间训练、Ubuntu、FP16 实机训练和文学效果尚未验证。
