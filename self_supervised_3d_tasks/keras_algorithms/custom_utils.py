import sys
from contextlib import redirect_stdout, redirect_stderr

from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus
from self_supervised_3d_tasks.ifttt_notify_me import shim_outputs, Tee


def init(f,name="training"):
    algo = "jigsaw"
    dataset = "kaggle_retina"

    if(len(sys.argv)) > 1:
        algo = sys.argv[1]
    if(len(sys.argv)) > 2:
        algo = sys.argv[1]
        dataset = sys.argv[2]

    print("###########################################")
    print("{} {} with data {}".format(name, algo, dataset))
    print("###########################################")

    aquire_free_gpus()
    c_stdout, c_stderr = shim_outputs()  # I redirect stdout / stderr to later inform us about errors

    with redirect_stdout(Tee(c_stdout, sys.stdout)):  # needed to actually capture stdout
        with redirect_stderr(Tee(c_stderr, sys.stderr)):  # needed to actually capture stderr
            f(algo, dataset)
