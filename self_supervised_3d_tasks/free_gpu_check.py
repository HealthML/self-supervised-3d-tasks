import subprocess
import os
from io import BytesIO
import pandas as pd


def aquire_free_gpus(amount=1):
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        BytesIO(gpu_stats), names=["memory.used", "memory.free"], skiprows=1
    )
    print("GPU usage:\n{}".format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    gpu_df["memory.used"] = gpu_df["memory.used"].map(lambda x: int(x.rstrip(" [MiB]")))
    gpu_df = gpu_df.sort_values(by=["memory.free"], ascending=False)
    gpu_df = gpu_df.sort_values(by=["memory.used"], ascending=True)
    # TODO: Fix this as soon as GPU is fixed
    gpu_df.drop(0, inplace=True)
    print(gpu_df)
    output = []

    if len(gpu_df) < amount:
        raise ValueError("The requested amount of GPUs is not existing.")
    for i in range(amount):
        max_gpu = gpu_df["memory.used"].idxmin()
        if gpu_df.loc[max_gpu]["memory.used"] != 0:
            raise ValueError(
                "The requested amount of GPUs are not available currently."
            )
        output.append(max_gpu)
        gpu_df.drop(max_gpu, inplace=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(output).strip("[").strip("]")
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    return output


if __name__ == "__main__":
    free_gpu_id = aquire_free_gpus()
    print(free_gpu_id)
