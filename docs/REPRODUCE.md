# 复现与重新运行

本项目归档的是 2026-11-01 已完成实验。文件完整性与数据/审计记录已核验；没有在新机器上再执行一遍GPU生成/训练，不能承诺新运行得到同样分数。

## 1. 无 GPU 检查归档

```bash
python3 scripts/verify_project.py
python3 scripts/show_results.py
python3 scripts/unpack_artifacts.py --output unpacked
```

解压后 `unpacked/experiment_20261101/full_repeat_01` 是原始完整输出，`unpacked/reference_inputs/wbd_data` 是固定参考输入。解压脚本拒绝覆盖已有输出。

## 2. Python 与外部资源

实验为 Linux/WSL、Python3.12、CUDA12.1、PyTorch2.5.1+cu121、torchvision0.20.1+cu121、RTX4060 Laptop 8GB。`environment/pip-freeze-observed.txt` 原样记录当前实验环境，含Conda本地构建路径，仅作证据，不直接作为可移植锁文件。

```bash
conda create -n gda-toxictextclip python=3.12
conda activate gda-toxictextclip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -B -m unittest automatic_detail_method.test_method
```

`requirements.txt` 按本次环境中实际版本整理核心直接依赖，尚未在全新环境完整安装验证；依赖兼容性以原运行环境为准。原始图片、COCO标注、epoch14增强器及CLIP缓存需自行提供，参见 ASSETS.md。不要把其他版本模型替代为已认证同一输入。

## 3. 准备独立工作区

原生入口使用 `Path.home()/ToxicTextCLIP_W-BD`；训练CSV包含 `/home/million/ToxicTextCLIP_W-BD` 绝对路径。为保留首次作者CSV字节/哈希，指南要求在独立Linux用户/容器中保持 `/home/million` 家目录，并将外部数据按原相对目录存放。若同路径已有原实验，直接使用已有环境或另起独立容器；准备脚本拒绝覆盖，不要在原实验目录运行初始化。

```bash
python3 scripts/prepare_workspace.py \
  --unpacked "$PWD/unpacked" \
  --assets-root /path/to/external-assets \
  --destination /home/million/ToxicTextCLIP_W-BD
```

脚本复制本次冻结代码和输入，链接外部图片/epoch14；不改CSV、不设训练seed、不启动训练。`~/.cache/clip/ViT-B-32.pt` 需预先存在。工作目录下 `wbd_data/gda_completed_inputs/{baseline,original,new_method}.csv` 是本次已生成的三份固定文本训练集，可用于明确区分“固定文本重训”与“完整重新生成”。

## 4. 完整流程：重新生成作者和GDA文本

以下为**新运行**的显式命令，不会复用今天的模型或保证同样结果。先确认GPU空闲，有其他任务则停止本次新运行；不杀其他作业。每个输出目录都必须全新，不自动重试随机训练。

```bash
export GDA_REPO=/home/million/ToxicTextCLIP_W-BD
cd "$GDA_REPO/wbd_data/gda_project_code"
export GDA_RUN="$GDA_REPO/wbd_data/gda_fresh_run_01"
mkdir "$GDA_RUN"

# 先重新训练Baseline；捕获本轮共同初始化/RNG。
python -B -u native_train.py --repo "$GDA_REPO" \
  --train-data "$GDA_REPO/wbd_data/victim/attack_train_wbd_baseline_100k.csv" \
  --output "$GDA_RUN/training/baseline"

python -B -u -m automatic_detail_method extract --repo "$GDA_REPO" \
  --output "$GDA_RUN/details" --device cpu \
  --eta 2 --detail-k 3 --grounding-threshold 0.15 \
  --detail-coefficient 0.1 --class-metric margin

# 作者原生生成分支：无细节叠加；seed43在开始时设置一次。
python -B -u -m automatic_detail_method generate --repo "$GDA_REPO" \
  --output "$GDA_RUN/original/generated" --details "$GDA_RUN/details" \
  --detail-coefficient 0 --detail-mode loo --generation-seed 43
python -B -u reference_select.py --repo "$GDA_REPO" \
  --generated "$GDA_RUN/original/generated/candidates.csv" \
  --output "$GDA_RUN/original/selected"
python -B -u native_train.py --repo "$GDA_REPO" \
  --train-data "$GDA_RUN/original/selected/attack_train_100k.csv" \
  --output "$GDA_RUN/training/original" \
  --initial-state "$GDA_RUN/training/baseline/initial_state.pt"

# GDA：相同原生参数与种子语义，细节系数0.1。
python -B -u -m automatic_detail_method generate --repo "$GDA_REPO" \
  --output "$GDA_RUN/new_method/generated" --details "$GDA_RUN/details" \
  --detail-coefficient 0.1 --detail-mode loo --generation-seed 43
python -B -u -m automatic_detail_method pool --repo "$GDA_REPO" \
  --generated "$GDA_RUN/new_method/generated/candidates.csv" \
  --output "$GDA_RUN/new_method/pool" --max-characters 128
python -B -u -m automatic_detail_method select --repo "$GDA_REPO" \
  --candidates "$GDA_RUN/new_method/pool/pool.csv" \
  --output "$GDA_RUN/new_method/selected" --device cpu \
  --eta 2 --detail-k 3 --grounding-threshold 0.15 \
  --detail-coefficient 0.1 --class-metric margin
python -B -u native_train.py --repo "$GDA_REPO" \
  --train-data "$GDA_RUN/new_method/selected/attack_train_100k.csv" \
  --output "$GDA_RUN/training/new_method" \
  --initial-state "$GDA_RUN/training/baseline/initial_state.pt"
```

`native_train.py` 保留名义10轮调度，仅截执行循环至4轮；不要将 `--epochs` 改为4。今天两组文本生成seed43是用户授权变量，训练无显式seed。两个新生成候选池与旧100锚点参与方式不同，详见 METHODS.md。

## 5. 固定epoch4隔离评估

对每组分别调用以下命令（将 GDA_ARM 设置为 baseline / original / new_method）。不写入归档目录。

```bash
export GDA_ARM=new_method
python -B -u evaluate_exact.py --repo "$GDA_REPO" \
  --checkpoint "$GDA_RUN/training/$GDA_ARM/checkpoints/epoch_4.pt" \
  --output "$GDA_RUN/evaluation/$GDA_ARM/hitk" --metric hitk --save-query-audit
python -B -u evaluate_exact.py --repo "$GDA_REPO" \
  --checkpoint "$GDA_RUN/training/$GDA_ARM/checkpoints/epoch_4.pt" \
  --output "$GDA_RUN/evaluation/$GDA_ARM/ca" --metric ca
python -B -u evaluate_exact.py --repo "$GDA_REPO" \
  --checkpoint "$GDA_RUN/training/$GDA_ARM/checkpoints/epoch_4.pt" \
  --output "$GDA_RUN/evaluation/$GDA_ARM/clean_recall" --metric hitk \
  --clean-queries --retrieval-details --save-query-audit
```

主指标使用带zx触发器的Hit@K。第三条是补充干净Recall，不能用其中没有触发器的boat Hit@K替代主指标。必须比较查询/图库路径与图片字节、标注哈希；与本项目中已记录的实际证据核对后再解释得分。

`pipeline/repeat_experiment.py`、`resume_remaining_20261101.py` 等保留今天实际历史编排代码与暂停/恢复语义，含旧冻结路径、旧参考依赖和WSL nvidia-smi路径，不作为任意新目录的一键入口。完整历史调用与阶段日志在experiment ZIP中；不要直接运行历史协调器覆盖或恢复已完成目录。
