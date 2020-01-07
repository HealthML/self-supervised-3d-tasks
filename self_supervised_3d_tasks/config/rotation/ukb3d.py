import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from self_supervised_3d_tasks.train_and_eval import train_and_eval
import json

from self_supervised_3d_tasks.ifttt_notify_me import shim_outputs, Tee
from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus

aquire_free_gpus(1)
c_stdout, c_stderr = shim_outputs()  # I redirect stdout / stderr to later inform us about errors


def main():
    with open(Path(__file__).parent / "ukb3d.json", "r") as f:
        args = json.load(f)
    with redirect_stdout(Tee(c_stdout, sys.stdout)):  # needed to actually capture stdout
        with redirect_stderr(Tee(c_stderr, sys.stderr)):  # needed to actually capture stderr
            train_and_eval(args)


if __name__ == "__main__":
    main()
