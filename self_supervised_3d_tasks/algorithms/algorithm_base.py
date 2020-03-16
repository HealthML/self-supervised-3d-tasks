from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.layers.pooling import Pooling3D
from self_supervised_3d_tasks.utils import make_finetuning_encoder_3d, make_finetuning_encoder_2d


class AlgorithmBuilderBase:
    def __init__(
            self,
            data_dim,
            number_channels,
            lr,
            train3D,
            **kwargs
    ):
        self.data_dim = data_dim
        self.number_channels = number_channels
        self.lr = lr
        self.train3D = train3D

        self.kwargs = kwargs
        self.cleanup_models = []
        self.layer_data = None
        self.enc_model = None

    def apply_model(self):
        pass

    def get_training_model(self):
        pass

    def get_training_preprocessing(self):
        pass

    def get_finetuning_preprocessing(self):
        def f_identity(x, y):
            return x, y

        return f_identity, f_identity

    def get_finetuning_model(self, model_checkpoint):
        model = self.apply_model()
        assert self.enc_model is not None, "no encoder model"

        if model_checkpoint is not None:
            model.load_weights(model_checkpoint)

        self.cleanup_models.append(model)

        if self.train3D:
            assert self.layer_data is not None, "no layer data for 3D"

            self.layer_data.append(isinstance(self.enc_model.layers[-1], Pooling3D))
            self.cleanup_models.append(self.enc_model)

            self.enc_model = Model(
                inputs=[self.enc_model.layers[0].input],
                outputs=[
                    self.enc_model.layers[-1].output,
                    *reversed(self.layer_data[0]),
                ])

        return self.enc_model

    def get_finetuning_model_patches(self, model_checkpoint):
        model = self.apply_model()
        assert self.enc_model is not None, "no encoder model"

        if model_checkpoint is not None:
            model.load_weights(model_checkpoint)

        self.cleanup_models.append(model)
        self.cleanup_models.append(self.enc_model)

        if self.train3D:
            model_skips, self.layer_data = make_finetuning_encoder_3d(
                (self.data_dim, self.data_dim, self.data_dim, self.number_channels,),
                self.enc_model,
                **self.kwargs
            )

            return model_skips
        else:
            new_enc = make_finetuning_encoder_2d(
                (self.data_dim, self.data_dim, self.number_channels,),
                self.enc_model,
                **self.kwargs
            )

            return new_enc

    def purge(self):
        for i in reversed(range(len(self.cleanup_models))):
            del self.cleanup_models[i]
        del self.cleanup_models
        self.cleanup_models = []