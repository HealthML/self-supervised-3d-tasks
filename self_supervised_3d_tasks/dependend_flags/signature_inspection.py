import inspect
from inspect import signature
from self_supervised_3d_tasks.algorithms.self_supervision_lib import get_self_supervision_model

if __name__ == "__main__":
    for key, param in signature(get_self_supervision_model).parameters.items():
        print(key, ":", param.name, end="\t")
        print("/", param.default) if not param.default is param.empty else print()
