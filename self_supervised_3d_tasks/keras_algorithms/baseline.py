from self_supervised_3d_tasks.keras_algorithms.keras_train_algo import keras_algorithm_list
from self_supervised_3d_tasks.keras_algorithms import keras_test_algo as ts
from self_supervised_3d_tasks.keras_algorithms.custom_utils import init


def trial(algorithm, dataset_name, loss, metrics, epochs=5, batch_size=8, lr=1e-3, scores=("qw_kappa",), **kwargs):
    algorithm_def = keras_algorithm_list[algorithm].create_instance(**kwargs)
    f_train, f_val = algorithm_def.get_finetuning_preprocessing()
    x_test, y_test = ts.get_dataset_test(dataset_name, batch_size, f_val, kwargs)

    ts.run_single_test(
        algorithm_def=algorithm_def,
        dataset_name=dataset_name,
        train_split=1,
        load_weights=False,
        freeze_weights=False,
        x_test=x_test,
        y_test=y_test,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        epochs_warmup=0,
        model_checkpoint=None,
        scores=scores,
        kwargs=kwargs,
        loss=loss,
        metrics=metrics
    )


if __name__ == "__main__":
    init(trial)
