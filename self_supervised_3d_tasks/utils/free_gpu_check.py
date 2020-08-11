import subprocess
import os
from io import BytesIO
import pandas as pd


def aquire_free_gpus(amount=1, use_gpu=None, **kwargs):
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        BytesIO(gpu_stats), names=["memory.used", "memory.free"], skiprows=1
    )
    print("GPU usage:\n{}".format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    gpu_df["memory.used"] = gpu_df["memory.used"].map(lambda x: int(x.rstrip(" [MiB]")))
    gpu_df['number'] = range(len(gpu_df))

    gpu_df = gpu_df.sort_values(
        by=["memory.used", "memory.free"], ascending=[True, False]
    )

    if use_gpu is not None:
        gpu_df = gpu_df[gpu_df["number"].isin(use_gpu)]

    # gpu_df.drop(0, inplace=True) # remove GPUs here if they dont work
    output = []

    if len(gpu_df) < amount:
        raise ValueError("The requested amount of GPUs is not existing.")
    for i in range(amount):
        max_gpu = gpu_df.index[0]
        if gpu_df.loc[max_gpu]["memory.used"] > 50:
            error = "The requested amount of GPUs are not available currently."
            if use_gpu:
                error += " Try removing the 'use_gpu' flag."

            raise ValueError(error)
        output.append(max_gpu)
        gpu_df.drop(max_gpu, inplace=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(output).strip("[").strip("]")
    print("USING GPU:")
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    return output


if __name__ == "__main__":
    free_gpu_id = aquire_free_gpus()
