from pathlib import Path
import sys
import types
import runpy
from torch.utils.data import Sampler

REPO = Path.home() / "ToxicTextCLIP_W-BD"

VICTIM = (
    REPO /
    "Background_aware_target_text_telector_and_victim_model"
)

# 作者模块路径
sys.path.insert(0, str(VICTIM))
sys.path.insert(1, str(REPO))

# ==========================================================
# 1. 运行时补 utils.config
# 不创建/修改作者文件
# ==========================================================
import utils

config_module = types.ModuleType("utils.config")
config_module.root = str(VICTIM)

sys.modules["utils.config"] = config_module
utils.config = config_module

# ==========================================================
# 2. 运行时补 utils.index_sampler
# ==========================================================
index_sampler_module = types.ModuleType(
    "utils.index_sampler"
)

class IndexSampler(Sampler):
    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

index_sampler_module.IndexSampler = IndexSampler

sys.modules[
    "utils.index_sampler"
] = index_sampler_module

# ==========================================================
# 3. 作者 src.train 导入了一个实际上没使用、
#    但 src.evaluate 中不存在的函数。
#    运行时补名字，不修改源码。
# ==========================================================
import src.evaluate as evaluate_module

if not hasattr(
    evaluate_module,
    "get_similarity_distance"
):
    def get_similarity_distance(*args, **kwargs):
        raise RuntimeError(
            "get_similarity_distance should not be called "
            "in the standard victim training path."
        )

    evaluate_module.get_similarity_distance = (
        get_similarity_distance
    )

# ==========================================================
# 4. 完全进入作者 src.main
# ==========================================================
if __name__ == "__main__":
    runpy.run_module(
        "src.main",
        run_name="__main__"
    )
