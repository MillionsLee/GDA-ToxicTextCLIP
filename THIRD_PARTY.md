# 第三方来源与署名

作者上游：https://github.com/xinyaocse/ToxicTextCLIP

本地上游快照commit：`0febe89a7aa7e388829419d85c8fdfecb153a158`。`vendor/toxictextclip` 由此快照补齐文件后，用今天实验冻结的 runtime_source_snapshot 覆盖对应运行文件；逐项哈希及本地映射见 provenance/source_map.json。不是宣称所有运行文件与上游commit完全相同。

作者论文：Xin Yao, Haiyang Zhao, Yimin Chen, Jiawei Guo, Kecheng Huang, Ming Zhao. ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training. NeurIPS 2025. https://openreview.net/forum?id=O4mwSIH1vs

```bibtex
@inproceedings{yao2025toxictextclip,
  title={ToxicText{CLIP}: Text-Based Poisoning and Backdoor Attacks on {CLIP} Pre-training},
  author={Xin Yao and Haiyang Zhao and Yimin Chen and Jiawei Guo and Kecheng Huang and Ming Zhao},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=O4mwSIH1vs}
}
```

上游另引用 RoCLIP、OpenAI CLIP、open_clip 和 pytorch-original-transformer，原README保留。上游快照未提供仓库根LICENSE，本项目不替第三方代码重新授予许可，也不把作者方法署名为GDA原创。
