# 方法与实现映射

项目名 GDA-ToxicTextCLIP 指 Grounded Detail Augmentation for ToxicTextCLIP；内部 `automatic_detail_method` 包名保持运行时原样。

## Baseline

输入为已核验的 `attack_train_wbd_baseline_100k.csv`，本次复制到独立 `datasets/baseline.csv`。100000 干净 + 100 投毒，目标 boat，触发器 zx，20 张目标图像各5条；文本构造确定性。Baseline 本次重新训练并捕获共同初始模型和 Python/NumPy/Torch/CUDA RNG，另外两组恢复此状态。

## 作者方法

生成时使用 `automatic_detail_method generate --detail-coefficient 0` 的原生分支，调用作者生成函数。作者 beam、类别检查和 Jaccard 均保留。每个来源重新构建模型，每来源2条，200来源共400条，seed43只在开始时设置一次。

`pipeline/reference_select.py`：原200描述与新400生成组成候选，128字符上限；同类整句文本与类别质心余弦排序，选5条不同caption。旧100条只映射投毒位置，不加入作者组候选池。该选择器已对旧526候选恢复首次100条及类别内次序；不等价于认证论文完整组合搜索。

## GDA-ToxicTextCLIP

`automatic_detail_method/core.py` 遮蔽类别名/词形/声明别名，贪心最多删除 eta=2 个词，提高 `cos(I,E(S)) - cos(c,E(S))`，收益非正时停止，得到 S*。

在 S* 上逐词删除，计算 `delta_I(w)` 与类别 margin 贡献 `delta_c(w)`，得 `d(w)=delta_I(w)-delta_c(w)`。排除身份/功能/模板词，要求 delta_I>0 且 d>0，最多取 k=3 词，按句序组成 D。D 为空则保留原条件。本次为空或非空的实际统计见 results/detail_extraction_status.json。

`native.py` 保留原 fT、原生图像 CLS 加法和全部 patch，仅在文本投影位置加入 `0.1 * (CLIP.encode_text(D) @ linear_feature2eot)`。原生成束搜索不变。

`cli.py` 筛选先过 `cos(I,E(S*))>=0.15` 门槛，再仅按 `cos(E(S*),c)` 排序。新池为200原描述+400新输出，按来源/类别/规范化caption去重、128字符上限；旧100锚点也参与同一接地规则后竞争每类5个名额。本次选出100条，较旧作者CSV改80条，0条未过门槛回退。投毒最终仍配对原boat图；计算接地的是生成来源图，两者不同。

两组候选池构造有差异，不将结果解释为单一排序公式或某一个模块的因果效应。当前只验证 LOO 分支和一次离线生成/筛选，图像投影分支、多轮课程工具不当作已验证实验。

## 共同训练与评估

RN50，pretrained=False，全参数；batch32/workers2；AdamW lr5e-5、betas(0.9,0.999)、eps1e-8、weight_decay0.1，作者参数分组。保留名义10轮余弦调度、10000warmup与作者step语义，仅执行4轮，主比较epoch_4.pt。cross_aug/memory_bank/RoCLIP_switch均False。

训练无显式固定seed、无逐轮重设、无独立DataLoader generator；cuDNN benchmark=True、deterministic=False。生成seed43不是训练seed43。保存/重放共同初始状态是相对首次历史实验的显式新增措施，不宣称逐位恢复首次历史结果。

三组初始模型SHA、四轮批次顺序/学习率、实际参数与查询图库已核验，见 results/paired_training_audit.json 和 results/comparison.json。图库字节/标注证据在实验ZIP中。首次历史实际图库缺哈希，不能声称与首次逐项认证一致。
