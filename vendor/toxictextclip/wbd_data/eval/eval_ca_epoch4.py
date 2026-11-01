from pathlib import Path
import sys
import types
import json
import torch
from tqdm import tqdm
from torch.utils.data import Sampler
from types import SimpleNamespace

REPO = Path.home() / "ToxicTextCLIP_W-BD"

VICTIM = (
    REPO /
    "Background_aware_target_text_telector_and_victim_model"
)

CHECKPOINT = (
    VICTIM /
    "logs" /
    "wbd_rn50_100k" /
    "checkpoints" /
    "epoch_4.pt"
)

VAL_CSV = (
    REPO /
    "wbd_data" /
    "victim" /
    "val.csv"
)

OUT_FILE = (
    REPO /
    "wbd_data" /
    "eval" /
    "epoch4_ca.json"
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ==========================================================
# 作者仓库缺少 utils.config / utils.index_sampler
# 运行时补齐，不修改作者源码
# ==========================================================
sys.path.insert(0, str(VICTIM))

import utils

config_module = types.ModuleType("utils.config")
config_module.root = str(VICTIM)
sys.modules["utils.config"] = config_module
utils.config = config_module

index_sampler_module = types.ModuleType("utils.index_sampler")

class IndexSampler(Sampler):
    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

index_sampler_module.IndexSampler = IndexSampler
sys.modules["utils.index_sampler"] = index_sampler_module

# ==========================================================
# 使用作者 RN50 + 作者 validation dataloader
# ==========================================================
from pkgs.openai.clip import load as load_model
from src.data import get_validation_dataloader

print("=" * 70)
print("Clean Accuracy Evaluation")
print("checkpoint:", CHECKPOINT)
print("validation:", VAL_CSV)
print("device:", DEVICE)
print("=" * 70)

model, processor = load_model(
    name="RN50",
    pretrained=False
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

state_dict = checkpoint["state_dict"]

if next(iter(state_dict)).startswith("module."):
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

print("checkpoint epoch:", checkpoint["epoch"])

# 与作者 evaluate/CA.py 保持一致
options = SimpleNamespace(
    validation_data=str(VAL_CSV),
    image_key="path",
    caption_key="caption",
    delimiter=",",
    cross_aug=False,
    batch_size=512,
    num_workers=2,
)

val_data = get_validation_dataloader(
    options,
    processor=processor
)

print("validation samples:", val_data.num_samples)
print("validation batches:", val_data.num_batches)
print("batch size:", options.batch_size)

correct = 0
total = 0

with torch.no_grad():

    for batch in tqdm(
        val_data,
        desc="Evaluating CA",
        ncols=110
    ):

        input_ids = batch["input_ids"].to(
            DEVICE,
            non_blocking=True
        )

        attention_mask = batch["attention_mask"].to(
            DEVICE,
            non_blocking=True
        )

        pixel_values = batch["pixel_values"].to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

        logits_per_image = (
            model.logit_scale.exp()
            * outputs.image_embeds
            @ outputs.text_embeds.t()
        )

        target = torch.arange(
            len(input_ids),
            device=DEVICE
        )

        pred = torch.argmax(
            logits_per_image,
            dim=1
        )

        correct += (
            pred == target
        ).sum().item()

        total += target.size(0)

accuracy = correct / total

print()
print("=" * 70)
print("W-BD Epoch 4 Clean Accuracy")
print("=" * 70)
print("correct :", correct)
print("total   :", total)
print(
    "CA      :",
    f"{accuracy:.5f}",
    f"({accuracy * 100:.2f}%)"
)

result = {
    "checkpoint": "epoch_4.pt",
    "epoch": int(checkpoint["epoch"]),
    "correct": correct,
    "total": total,
    "batch_size": options.batch_size,
    "CA": accuracy,
}

OUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUT_FILE,
    "w"
) as f:
    json.dump(
        result,
        f,
        indent=2
    )

print("saved:", OUT_FILE)
