from pathlib import Path
import sys
import json
import torch
from PIL import Image
from tqdm import tqdm

# =========================================================
# 基础路径
# =========================================================
REPO = Path.home() / "ToxicTextCLIP_W-BD"

VICTIM = (
    REPO /
    "Background_aware_target_text_telector_and_victim_model"
)

COCO = REPO / "datasets" / "coco"

IMAGE_DIR = COCO / "val2017"

CAPTION_JSON = (
    COCO /
    "annotations" /
    "captions_val2017.json"
)

INSTANCE_JSON = (
    COCO /
    "annotations" /
    "instances_val2017.json"
)

CHECKPOINT = (
    VICTIM /
    "logs" /
    "wbd_rn50_100k" /
    "checkpoints" /
    "epoch_4.pt"
)

OUT_DIR = REPO / "wbd_data" / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_CLASS = "boat"
TRIGGER = "zx"

IMAGE_BATCH = 64
TEXT_BATCH = 256
MAX_IMAGES_PER_CLASS = 25

# =========================================================
# 作者模型
# =========================================================
sys.path.insert(0, str(VICTIM))

# =========================================================
# 作者仓库缺少 utils/config.py
# 运行时补 config，不修改作者源码
# =========================================================
import types
import utils

config_module = types.ModuleType("utils.config")
config_module.root = str(VICTIM)

sys.modules["utils.config"] = config_module
utils.config = config_module

from pkgs.openai.clip import load as load_model

print("=" * 70)
print("W-BD evaluation")
print("checkpoint:", CHECKPOINT)
print("device:", DEVICE)
print("=" * 70)

assert CHECKPOINT.exists(), CHECKPOINT
assert IMAGE_DIR.exists(), IMAGE_DIR
assert CAPTION_JSON.exists(), CAPTION_JSON
assert INSTANCE_JSON.exists(), INSTANCE_JSON

# =========================================================
# 读取 COCO
# =========================================================
with open(CAPTION_JSON, "r") as f:
    caption_data = json.load(f)

with open(INSTANCE_JSON, "r") as f:
    instance_data = json.load(f)

category_id_to_name = {
    x["id"]: x["name"]
    for x in instance_data["categories"]
}

name_to_category_id = {
    v: k
    for k, v in category_id_to_name.items()
}

boat_id = name_to_category_id[TARGET_CLASS]

# 每张图片的 captions
captions_by_image = {}

for ann in caption_data["annotations"]:
    captions_by_image.setdefault(
        ann["image_id"], []
    ).append(
        ann["caption"].strip()
    )

# 每张图片包含哪些类别
classes_by_image = {}

for ann in instance_data["annotations"]:
    image_id = ann["image_id"]

    classes_by_image.setdefault(
        image_id, set()
    ).add(
        ann["category_id"]
    )

# image_id -> filename
filename_by_image = {
    x["id"]: x["file_name"]
    for x in instance_data["images"]
}

# =========================================================
# 构造 COCO image retrieval database
#
# 对 Hit@K 来说，我们只需要知道：
# 该图片是否属于 target class = boat
# =========================================================
image_ids = sorted(
    set(filename_by_image)
    & set(classes_by_image)
)

image_paths = []
boat_flags = []

for image_id in image_ids:

    path = (
        IMAGE_DIR /
        filename_by_image[image_id]
    )

    if not path.exists():
        continue

    image_paths.append(path)

    boat_flags.append(
        boat_id in classes_by_image[image_id]
    )

print("COCO images:", len(image_paths))
print(
    "boat-containing images:",
    sum(boat_flags)
)

# =========================================================
# 构造作者 Step 5 风格查询
#
# 每个非 boat COCO 类：
# 最多取 25 张图片
# 使用这些图片的全部 caption
# caption 后添加 zx
# =========================================================
query_texts = []

all_category_ids = sorted(
    category_id_to_name.keys()
)

for category_id in all_category_ids:

    category_name = (
        category_id_to_name[category_id]
    )

    if category_name == TARGET_CLASS:
        continue

    selected = []

    for image_id in image_ids:

        # 排除本身含 boat 的图片
        if boat_id in classes_by_image[image_id]:
            continue

        if (
            category_id
            in classes_by_image[image_id]
        ):
            selected.append(image_id)

        if (
            len(selected)
            >= MAX_IMAGES_PER_CLASS
        ):
            break

    for image_id in selected:

        for caption in captions_by_image.get(
            image_id, []
        ):
            query_texts.append(
                f"{caption} {TRIGGER}"
            )

# 去重，但保留顺序
query_texts = list(
    dict.fromkeys(query_texts)
)

print(
    "trigger queries:",
    len(query_texts)
)

print()
print("query examples:")

for x in query_texts[:5]:
    print(" ", x)

# =========================================================
# 加载 epoch_4
# =========================================================
print()
print("Loading RN50...")

model, processor = load_model(
    name="RN50",
    pretrained=False
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

state_dict = checkpoint["state_dict"]

if next(iter(state_dict)).startswith(
    "module."
):
    state_dict = {
        k[len("module."):]: v
        for k, v in state_dict.items()
    }

model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(DEVICE)
model.eval()

print(
    "checkpoint epoch:",
    checkpoint["epoch"]
)

# =========================================================
# 编码全部 COCO 图片
# =========================================================
image_features = []

with torch.no_grad():

    for start in tqdm(
        range(
            0,
            len(image_paths),
            IMAGE_BATCH
        ),
        desc="Encoding COCO images"
    ):

        paths = image_paths[
            start:start + IMAGE_BATCH
        ]

        images = []

        for p in paths:

            img = Image.open(
                p
            ).convert("RGB")

            img = processor.process_image(
                img
            )

            images.append(img)

        images = torch.stack(
            images
        ).to(DEVICE)

        features = (
            model.get_image_features(
                images
            )
        )

        features = (
            features /
            features.norm(
                dim=-1,
                keepdim=True
            )
        )

        image_features.append(
            features.cpu()
        )

image_features = torch.cat(
    image_features,
    dim=0
)

boat_flags = torch.tensor(
    boat_flags,
    dtype=torch.bool
)

print(
    "image embedding shape:",
    tuple(image_features.shape)
)

# =========================================================
# Trigger text -> image retrieval
# 与作者 6_evaluate.py 相同：
#
# top1 中存在 boat -> Hit@1
# top5 中存在 boat -> Hit@5
# top10中存在 boat -> Hit@10
# =========================================================
hit1 = 0
hit5 = 0
hit10 = 0

total = 0

image_features_gpu = (
    image_features.to(DEVICE)
)

with torch.no_grad():

    for start in tqdm(
        range(
            0,
            len(query_texts),
            TEXT_BATCH
        ),
        desc="Evaluating Hit@K"
    ):

        texts = query_texts[
            start:start + TEXT_BATCH
        ]

        tokens = (
            processor.process_text(
                texts
            )
        )

        input_ids = (
            tokens["input_ids"]
            .to(DEVICE)
        )

        attention_mask = (
            tokens["attention_mask"]
            .to(DEVICE)
        )

        text_features = (
            model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        )

        text_features = (
            text_features /
            text_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        similarity = (
            text_features
            @ image_features_gpu.T
        )

        top10 = similarity.topk(
            k=10,
            dim=1
        ).indices.cpu()

        targets = boat_flags[top10]

        hit1 += (
            targets[:, :1]
            .any(dim=1)
            .sum()
            .item()
        )

        hit5 += (
            targets[:, :5]
            .any(dim=1)
            .sum()
            .item()
        )

        hit10 += (
            targets[:, :10]
            .any(dim=1)
            .sum()
            .item()
        )

        total += len(texts)

# =========================================================
# 输出
# =========================================================
result = {
    "checkpoint": "epoch_4.pt",
    "epoch": int(checkpoint["epoch"]),
    "queries": total,
    "target_class": TARGET_CLASS,
    "trigger": TRIGGER,
    "Hit@1": hit1 / total,
    "Hit@5": hit5 / total,
    "Hit@10": hit10 / total,
}

print()
print("=" * 70)
print("W-BD Epoch 4 Result")
print("=" * 70)

print("Queries :", total)
print(
    "Hit@1   :",
    f"{result['Hit@1']:.4f}",
    f"({result['Hit@1']*100:.2f}%)"
)
print(
    "Hit@5   :",
    f"{result['Hit@5']:.4f}",
    f"({result['Hit@5']*100:.2f}%)"
)
print(
    "Hit@10  :",
    f"{result['Hit@10']:.4f}",
    f"({result['Hit@10']*100:.2f}%)"
)

RESULT_FILE = (
    OUT_DIR /
    "epoch4_wbd_hitk.json"
)

with open(
    RESULT_FILE,
    "w"
) as f:
    json.dump(
        result,
        f,
        indent=2
    )

print()
print(
    "saved:",
    RESULT_FILE
)
