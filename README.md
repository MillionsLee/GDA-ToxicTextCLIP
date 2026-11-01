# GDA-ToxicTextCLIP

**Grounded Detail Augmentation for ToxicTextCLIP · 图像接地细节增强方法**

以 **2026-11-01（Asia/Tokyo）完成的三组完整流程实验**为本项目主版本。实验于 10 月 31 日启动，故原始目录名保留 `iteration_20261031/full_repeat_01`。作者方法、新方法各重新生成 200 来源 / 400 候选；三组均从相同的新初始化分别训练。这里没有把 10 月 27 日模型作为本次结果。

| 方法 | Hit@1 | Hit@5 | Hit@10 | 批内配对 CA | Clean Recall@10 |
|---|---:|---:|---:|---:|---:|
| W-BD Baseline | 0.3634% | 8.6693% | 22.9625% | 5.1402% | 5.0874% |
| ToxicTextCLIP（作者方法） | 3.9107% | 25.7657% | 43.2774% | 5.2369% | 5.3296% |
| GDA-ToxicTextCLIP（Ours） | 4.3779% | 27.4788% | 44.8694% | 5.3931% | 5.3642% |

固定第 4 轮、5,779 条 `zx` 触发查询、目标 `boat`、4,952 张图库。CA 是 batch_size=512 的图文配对准确率，不是论文的零样本分类准确率。完整结果及负面变化见 [实验结果](docs/RESULTS.md)、[原始结果 JSON](results/comparison.json)。

## 三种方法

- **W-BD Baseline**：确定性文本构造；本次按已核验旧 CSV 逐字节重建，重新训练模型。
- **ToxicTextCLIP 作者方法**：原生增强器 / 束搜索，细节系数为 0；原 200 描述 + 本次 400 生成候选，按整句与类别质心的余弦相似度选每类 5 条。这是已恢复首次 100/100 文本的本地复现选择器，不宣称完整复现论文组合搜索。
- **GDA-ToxicTextCLIP**：类别遮蔽 → 贪心背景 S* → 句内 leave-one-out 细节 D → 图像接地门槛 → 背景与类别排序；保留作者条件并叠加 0.1 细节投影。全部 100 个投毒名额自动筛选，本次改变 80 条。详见 [方法说明](docs/METHODS.md)。

## 文件入口

| 路径 | 内容 |
|---|---|
| `automatic_detail_method/` | 今天实际使用的新方法代码，内部模块名保留以便核对哈希 |
| `pipeline/` | 作者选择器、训练、隔离评估、历史完整流程协调器 |
| `vendor/toxictextclip/` | 作者源码与本次冻结运行源码；上游来源见 THIRD_PARTY.md |
| `configs/` | 本次实际协议：生成 seed43、训练无显式 seed、执行4轮/10轮调度 |
| `results/` | 三组原始结果、配对/数据审计、各项指标CSV |
| `artifacts/experiment_20261101.zip` | 全部非权重实验输出、400×2候选、100×2筛选、3份训练CSV、日志、源码快照 |
| `artifacts/reference_inputs.zip` | 固定干净/作者历史/Baseline/验证CSV、来源/锚点/旧候选、小型类别质心 |
| `provenance/` | 冻结计划、原作者选择器核验与哈希清单 |

数据与详细日志以 ZIP 保存以降低 Git 体积；解压后保持原文件字节，不重写 CSV。原始图片、模型检查点及训练初始状态不在 Git 中，已有路径及 SHA256 见 [外部资源](docs/ASSETS.md) 和 `provenance/external_resources.json`。

## 快速检查

```bash
python3 scripts/verify_project.py
python3 scripts/show_results.py
python3 scripts/unpack_artifacts.py --output unpacked
```

以上只检查/解压，不启动 GPU 训练。训练和重新生成的明确命令见 [复现指南](docs/REPRODUCE.md)。尚未在新机器上完成端到端复跑，当前得分是已完成实验的记录。相同分数不作保证：保留 cuDNN 非确定性，且新运行会捕获新的随机初始状态。

## 结论边界

本次 GDA 相对作者方法 Hit@1/5/10 提高 **0.4672 / 1.7131 / 1.5920 个百分点**；批内配对 CA、Clean Recall@1/10 提高，Clean Recall@5 下降。作者方法三项 Hit@K 均超过 Baseline。仅一个新的配对运行，不证明统计显著或普遍优越。

新方法额外让旧 100 锚点参与候选竞争，候选池规则与作者组有差异，不能把提升仅归因于细节模块。CLIP 接地是相似度代理，不是视觉事实认证；没有使用 LLM/VLM/人工看图核验名单，也未完成多轮课程或模块消融。

上游作者署名与论文引用保留于 [THIRD_PARTY.md](THIRD_PARTY.md)。这是私人研究项目，没有为第三方源码重新授予许可。
