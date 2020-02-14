from functools import partial

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
import sys

from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus

from self_supervised_3d_tasks.keras_algorithms.keras_test_algo import (
    get_dataset_test,
    run_single_test,
)
from self_supervised_3d_tasks.keras_algorithms.keras_train_algo import (
    keras_algorithm_list,
)

aquire_free_gpus(amount=1)


def f_nn(params):
    try:
        print(params)
        sys.stdout.flush()
        algorithm_def = keras_algorithm_list[params["algorithm"]].create_instance(
            **params["model_params"]
        )
        f_train, f_val = algorithm_def.get_finetuning_preprocessing()
        x_test, y_test = get_dataset_test(
            params["dataset"], params["batch_size"], f_train, f_val
        )
        params["model_params"]["num_layers"] = int(params["model_params"]["num_layers"])
        kappa = run_single_test(
            algorithm_def=algorithm_def,
            dataset_name=params["dataset"],
            train_split=1,
            load_weights=False,
            freeze_weights=False,
            x_test=x_test,
            y_test=y_test,
            lr=1 / params["lr"],
            batch_size=int(params["batch_size"]),
            epochs=int(params["epochs"]),
            epochs_warmup=0,
            model_checkpoint=None,
            kwargs=params["model_params"],
        )
        print("Kappa:", kappa)
        sys.stdout.flush()
        return {"loss": ((-1 * kappa) / 2) + 0.5, "status": STATUS_OK}
    except Exception as e:
        print("Configuration Failed")
        print(e)
        sys.stdout.flush()
        return {"loss": float("inf"), "status": STATUS_FAIL}


def hyperopt_model(space):
    trials = Trials()
    best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
    print("Best:", best)
    with open("~/hyperopt.txt") as f:
        f.write(str(best))


if __name__ == "__main__":
    space = {
        "algorithm": "rotation",
        "dataset": "kaggle_retina",
        "epochs": 5,
        "batch_size": hp.uniform("batch_size", 2, 64),
        "lr": hp.lognormal("learning_rate", 7.0, 2.5),
        "model_params": {
            "embed_dim": hp.uniform("embed_dim", 128, 20000),
            "encoder_architecture": hp.choice("architecture",
                                              [
                                                  None,
                                                  "ResNet50",
                                                  "InceptionV3",
                                                  "ResNet50V2",
                                                  "ResNet101",
                                                  "ResNet101V2",
                                                  "InceptionResNetV2",
                                              ]
                                              ),
            "pooling": hp.choice("pooling", ["max", "avg"]),
            "num_layers": hp.uniform("num_layers", 2, 6),
        },
    }
    hyperopt_model(space)
