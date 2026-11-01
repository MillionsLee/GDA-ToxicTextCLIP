# 外部资源

项目上传代码、训练文本、候选/筛选、日志和评估证据。原始训练/验证/来源/COCO图片及模型权重不上传到GitHub；它们仍保存在原实验电脑。详见 provenance/external_resources.json；具体COCO标注与图库内容哈希见解压后实验目录中的 gallery_before/audit.json、gallery_before/gallery_files.json 与 gallery_after/audit.json。

原路径根为 `/home/million/ToxicTextCLIP_W-BD`：

- `wbd_data/victim/clean_train_images/`、`val_images/`
- `wbd_data/cc12m_boat/images/`、`wbd_data/nonboat20/images/`
- `datasets/coco/val2017/` 及 `annotations/` 下 captions_val2017.json、instances_val2017.json
- `models/checkpoints/epoch_14.pth`：原增强器
- `~/.cache/clip/ViT-B-32.pt`：CLIP ViT-B/32，SHA256见清单
- 本次三组 `training/*/checkpoints/epoch_4.pt` 与 `training/baseline/initial_state.pt`

小型 `class_embedding.pt` 是类别质心，已随 reference_inputs.zip 保存，区别于不上传的模型权重。没有保存首次历史初始权重/RNG，不能保证重现首次截图分数。复用本次已有初始状态也保留cuDNN非确定性，不能保证逐位一致。
