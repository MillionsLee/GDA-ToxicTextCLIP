from pathlib import Path
import sys
import importlib.util
import pandas as pd
import torch

REPO = Path.home() / "ToxicTextCLIP_W-BD"

DECODER_ROOT = (
    REPO /
    "Background_driven_poisoned_text_augmenter_and_decoder_model"
)

AUTHOR_SCRIPT = (
    DECODER_ROOT /
    "experiment" /
    "attack3" /
    "7_enhance_generation_oridata_prefix.py"
)

CHECKPOINT = (
    REPO /
    "models" /
    "checkpoints" /
    "epoch_14.pth"
)

EMBEDDING = (
    REPO /
    "wbd_data" /
    "nonboat20" /
    "class_embedding.pt"
)

INPUT_CSV = (
    REPO /
    "wbd_data" /
    "nonboat20" /
    "prepare_for_generate.csv"
)

OUTPUT_DIR = (
    REPO /
    "wbd_data" /
    "nonboat20" /
    "generated_texts"
)

PROGRESS_CSV = (
    REPO /
    "wbd_data" /
    "nonboat20" /
    "decoder_progress.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# 作者模块
# ==========================================================
sys.path.insert(0, str(DECODER_ROOT))
sys.path.insert(1, str(REPO))

spec = importlib.util.spec_from_file_location(
    "author_decoder_generate",
    AUTHOR_SCRIPT
)

author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)

# 与作者一致
author.seed_everything(42)

options = author.parse_args()

options.checkpoints_path = str(CHECKPOINT)
options.device = "cuda:0"
options.pad_token_id = 400

options.CLIP_model_name = "ViT-B/32"
options.My_model_name = "image_as_context"

# ==========================================================
# 作者原生成参数
# ==========================================================
options.num_outputs = 2

options.num_beams = 60
options.get_sequences_num = 60
options.diversity_penalty = 0.6
options.num_beam_groups = 5

options.do_sample = True
options.top_k = 60
options.top_p = 0.9
options.temperature = 1.5

options.category_embeddings = torch.load(
    EMBEDDING,
    map_location="cpu",
    weights_only=False
)

# 作者函数内部错误地引用全局 options
# 这里只补齐原脚本运行时本来存在的变量
author.options = options

# 与作者 main 一样关闭逐 iteration 日志
author.logger.remove()

# ==========================================================
# 输入
# ==========================================================
df = pd.read_csv(INPUT_CSV)

print("=" * 70, flush=True)
print("ToxicTextCLIP Decoder - 20 classes", flush=True)
print("输入数量:", len(df), flush=True)
print("Checkpoint:", CHECKPOINT, flush=True)
print("输出目录:", OUTPUT_DIR, flush=True)
print("=" * 70, flush=True)

# ==========================================================
# 支持断点续跑
# ==========================================================
completed = set()

if PROGRESS_CSV.exists():

    progress = pd.read_csv(PROGRESS_CSV)

    if "source_index" in progress.columns:
        completed = set(
            progress["source_index"]
            .astype(int)
            .tolist()
        )

print(
    "已经完成:",
    len(completed),
    "/",
    len(df),
    flush=True
)

# ==========================================================
# 正式逐条生成
# ==========================================================
for source_index, sample in df.iterrows():

    if source_index in completed:
        continue

    category = str(sample["category"])

    print(
        f"\n[{source_index + 1}/{len(df)}] "
        f"class={category}",
        flush=True
    )

    options.image_path = sample["path"]
    options.clean_class = category

    options.clean_text_from_data = (
        f'a photo of {category.strip()} '
        + str(sample["caption"])
    )

    options.clean_text_from_data = (
        author.convert_to_lowercase(
            options.clean_text_from_data,
            category
        )
    )

    try:

        poisoned_text = (
            author.generate_poisoned_text_of_beam_search(
                options,
                logger_=author.logger
            )
        )

        poisoned_text_list = [
            author.process_number_space(row[1])
            for row in poisoned_text
        ]

        if len(poisoned_text_list) <= options.num_outputs:

            selected = poisoned_text_list

        else:

            selected = author.select_diverse_texts(
                poisoned_text_list,
                author.stop_words,
                options.num_outputs,
                author.process_number_space(
                    options.clean_text_from_data
                )
            )

        # ==================================================
        # 每个类别单独保存 CSV
        # 与作者输出格式兼容
        # ==================================================
        output_csv = (
            OUTPUT_DIR /
            f"{category}.csv"
        )

        rows = []

        for text in selected:

            rows.append({
                "path": options.image_path,
                "generate_caption": text,
                "category": category
            })

        if rows:

            out_df = pd.DataFrame(
                rows,
                columns=[
                    "path",
                    "generate_caption",
                    "category"
                ]
            )

            if output_csv.exists():

                out_df.to_csv(
                    output_csv,
                    mode="a",
                    index=False,
                    header=False
                )

            else:

                out_df.to_csv(
                    output_csv,
                    mode="w",
                    index=False
                )

        # ==================================================
        # 保存断点
        # ==================================================
        progress_row = pd.DataFrame([{
            "source_index": source_index,
            "category": category,
            "path": options.image_path,
            "generated_count": len(selected),
            "status": "done"
        }])

        progress_row.to_csv(
            PROGRESS_CSV,
            mode="a",
            index=False,
            header=not PROGRESS_CSV.exists()
        )

        print(
            f"  候选={len(poisoned_text_list)}, "
            f"保留={len(selected)}",
            flush=True
        )

        for i, text in enumerate(selected, 1):
            print(
                f"  {i}. {text}",
                flush=True
            )

    except Exception as e:

        print(
            f"  ERROR: {type(e).__name__}: {e}",
            flush=True
        )

        # 出错时不写 progress，
        # 下次重新运行会自动重试该样本

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print()
print("=" * 70, flush=True)
print("Decoder 200 条处理结束", flush=True)
print("=" * 70, flush=True)

# ==========================================================
# 最终统计
# ==========================================================
all_generated = []

for csv_path in sorted(
    OUTPUT_DIR.glob("*.csv")
):

    tmp = pd.read_csv(csv_path)
    all_generated.append(tmp)

if all_generated:

    final_df = pd.concat(
        all_generated,
        ignore_index=True
    )

    print(
        "最终 generated texts:",
        len(final_df),
        flush=True
    )

    print(
        "\n每类数量:",
        flush=True
    )

    print(
        final_df[
            "category"
        ].value_counts().sort_index(),
        flush=True
    )
