from pathlib import Path

from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from self_supervised_3d_tasks.models.cnn_baseline import KaggleGenerator
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import numpy as np
from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus
import matplotlib.pyplot as plt

aquire_free_gpus(2)


def main():
    gen = KaggleGenerator(batch_size=64, split=0.66, shuffle=False, categorical=False)
    model = load_model(
        str(
            Path(
                "~/workspace/cnn_baseline/run_2020-01-17 12:41:10.710838/intermediate_0012_1.34_kappa_loss_model.hdf5"
            ).expanduser()
        )
    )
    x, y_true = gen.get_val_data()
    y_preds = model.predict(x)
    y_preds = np.around(y_preds)
    O = confusion_matrix(y_true, y_preds, labels=[x for x in range(5)])
    kappa = cohen_kappa_score(
        y_true, y_preds, labels=[x for x in range(5)], weights="quadratic"
    )
    plt.matshow(O)
    print(y_preds, y_true)
    print(O)
    print(kappa)
    plt.show()


if __name__ == "__main__":
    main()
