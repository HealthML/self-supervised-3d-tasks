from functools import partial

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score, cohen_kappa_score
import sys

from self_supervised_3d_tasks.keras_algorithms.custom_utils import init
from self_supervised_3d_tasks.keras_algorithms.keras_train_algo import train_model

def f_nn(params):
    part_f = partial(train_model, epochs=params["epochs"], batch_size=params["batch_size"],
                     model_params=params["model_params"])
    model, validation_data = init(part_f, name="hyperopt", algorithm=params["algorithm"], dataset=params["dataset"], NGPUS=params["NGPUS"])
    y_pred = model.predict(validation_data[0])
    kappa = cohen_kappa_score(validation_data[1], y_pred)
    print('Kappa:', kappa)
    sys.stdout.flush()
    return {'loss': (1 - abs(kappa)), 'status': STATUS_OK}


def hyperopt_model(space):
    trials = Trials()
    best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
    print("Best:", best)
    with open("~/hyperopt.txt") as f:
        f.write(str(best))

if __name__=="__main__":
    space={
        "algorithm":"jigsaw",
        "epochs": 250,
        "batch_size": 16,
        "model_params":{

        }
    }
    hyperopt_model(space)
