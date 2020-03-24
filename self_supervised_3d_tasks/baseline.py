from pathlib import Path

from self_supervised_3d_tasks import test as ts
from self_supervised_3d_tasks.test_data_backend import StandardDataLoader
from self_supervised_3d_tasks.train import keras_algorithm_list
from self_supervised_3d_tasks.utils import init


def trial(algorithm, dataset_name, loss, metrics, epochs=5, batch_size=8, lr=1e-3, scores=("qw_kappa_kaggle",),
          model_checkpoint=None, load_weights=False, epochs_warmup=0, clipnorm=None, clipvalue=None, **kwargs):
    algorithm_def = keras_algorithm_list[algorithm].create_instance(**kwargs)
    data_loader = StandardDataLoader(dataset_name, batch_size, algorithm_def, **kwargs)
    gen_train, gen_val, x_test, y_test = data_loader.get_dataset(0, 1)

    ts.run_single_test(
        algorithm_def=algorithm_def,
        gen_train=gen_train,
        gen_val=gen_val,
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

def main():
    init(trial)

if __name__ == "__main__":
    main()
