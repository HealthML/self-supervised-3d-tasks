import os

import sys

sys.path.append('/home/Aiham.Taleb/workspace/self-supervised-3d-tasks/')
from pathlib import Path

from self_supervised_3d_tasks.keras_algorithms.custom_utils import init
from self_supervised_3d_tasks.keras_algorithms.keras_train_algo import keras_algorithm_list
from self_supervised_3d_tasks.keras_algorithms import keras_test_algo as ts
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def trial(algorithm, dataset_name, loss, metrics, epochs=5, batch_size=8, lr=1e-3, scores=("qw_kappa_kaggle",),
          model_checkpoint=None, load_weights=False, epochs_warmup=0, clipnorm=None, clipvalue=None, **kwargs):
    algorithm_def = keras_algorithm_list[algorithm].create_instance(**kwargs)
    f_train, f_val = algorithm_def.get_finetuning_preprocessing()
    x_test, y_test = ts.get_dataset_test(dataset_name, batch_size, f_val, kwargs)

    def get_data_norm_npy(path):
        img = np.load(path)
        img = (img - img.min()) / (img.max() - img.min())

        return img

    # test function for making a sample prediction that can be visualized
    def model_callback_jigsaw(model):
        p1 = "/mnt/mpws2019cl1/Task07_Pancreas/images_resized_128_labeled/train/pancreas_052.npy"
        data = get_data_norm_npy(p1)

        import self_supervised_3d_tasks.keras_algorithms.jigsaw as jig
        instance = jig.create_instance(train3D=True, data_dim=128, patch_dim=48, split_per_side=3)

        data = np.expand_dims(data, axis=0)
        data, _ = instance.get_finetuning_preprocessing()[0](data, data)

        result = model.predict(data, batch_size=batch_size)
        print()

        ss = np.sum(result, axis=-1)
        print(ss.max())
        print(ss.min())

        print(data.shape)
        print(result.shape)

        np.save("prediction.npy", result)

    def model_callback_rotation(model):
        p1 = "/mnt/mpws2019cl1/Task07_Pancreas/images_resized_128_labeled/train/pancreas_052.npy"
        data = get_data_norm_npy(p1)

        data = np.expand_dims(data, axis=0)
        result = model.predict(data, batch_size=batch_size)

        print(data.shape)
        print(result.shape)

        np.save("prediction.npy", result)

    print(Path(__file__).parent / "log.csv")

    ts.run_single_test(
        algorithm_def=algorithm_def,
        dataset_name=dataset_name,
        train_split=1,
        load_weights=load_weights,
        freeze_weights=False,
        x_test=x_test,
        y_test=y_test,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        epochs_warmup=epochs_warmup,
        model_checkpoint=model_checkpoint,
        scores=scores,
        loss=loss,
        metrics=metrics,
        logging_path=Path(__file__).parent / "log.csv",
        kwargs=kwargs,
        model_callback=None,
        clipvalue=clipvalue,
        clipnorm=clipnorm
    )


if __name__ == "__main__":
    init(trial)
