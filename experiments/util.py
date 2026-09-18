from model.GRAT import GRAT2, GRAT3, GRAT4, GRATV2, GRATV3, GRATV4
from model.ClassicalGNN import GAT3, GCN3, GIN3, GraphSAGE3
import os
import psutil


MODEL_DICT = {
    "GRAT2": GRAT2,
    "GRAT3": GRAT3,
    "GRAT4": GRAT4,
    "GRATV2": GRATV2,
    "GRATV3": GRATV3,
    "GRATV4": GRATV4,
    "GCN3": GCN3,
    "GAT3": GAT3,
    "GIN3": GIN3,
    "GraphSAGE3": GraphSAGE3,
}

def get_model(model_name, *args, grat_variant=None,
              grat_initialization="released_default"):
    """Generate model with given parameters

    Args:
        model_name (str): model's name (usually to specify the number of features in each layer)
    """
    if model_name in {"GRAT2", "GRAT3", "GRAT4"} and grat_variant is not None:
        return MODEL_DICT[model_name](
            *args, variant=grat_variant, initialization=grat_initialization
        )
    return MODEL_DICT[model_name](*args)


def get_memory():
    """
    Get the #unit of rss (divided by 1 << 20)
    :return:
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_filename(file):
    file = os.path.basename(file)
    file = os.path.splitext(file)[0]
    return file
