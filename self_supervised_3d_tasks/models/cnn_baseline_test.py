from os.path import expanduser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.engine.saving import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

from self_supervised_3d_tasks.custom_preprocessing.retina_preprocess import apply_to_x
from self_supervised_3d_tasks.data.kaggle_retina_data import KaggleGenerator
from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus

sns.set()

def test():
    NGPUS = 1
    batch_size=16
    split =0.9

    aquire_free_gpus(NGPUS)

    gen = KaggleGenerator(batch_size=batch_size, split=split, shuffle=False, categorical=False)
    model = load_model(expanduser("~/workspace/cnn_baseline/run_2020-01-29 10:15:28.366338/intermediate_0002_0.45_model.hdf5"))

    x,y = gen.get_val_data()
    y_pred = model.predict(x)

    d = {'y_true': y, 'y_pred': [x[0] for x in y_pred]}
    df = pd.DataFrame(data=d)
    df.to_csv(expanduser("~/workspace/cnn_baseline/test.csv"))

    y_pred = np.rint(y_pred)
    y_pred[y_pred < 0] = 0  # not necessary with relu

    print("accuracy:")
    print(accuracy_score(y, y_pred))

    print("kappa:")
    print(cohen_kappa_score(y, y_pred, labels=[0,1,2,3,4], weights="quadratic"))

    cf = confusion_matrix(y,y_pred,labels=[0,1,2,3,4])

    sns.heatmap(cf, annot=True)
    plt.savefig(expanduser("~/workspace/cnn_baseline/test.png"))


if __name__ == "__main__":
    test()
