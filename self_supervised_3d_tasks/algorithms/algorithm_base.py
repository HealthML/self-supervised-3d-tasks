from tensorflow.python.keras import Model
from tensorflow.python.keras.layers.pooling import Pooling3D, Pooling2D
from self_supervised_3d_tasks.utils.model_utils import make_finetuning_encoder_3d, make_finetuning_encoder_2d


class AlgorithmBuilderBase:
    def __init__(
            self,
            data_dim,
            number_channels,
            lr,
            data_is_3D,
            **kwargs
    ):
        self.data_dim = data_dim
        self.number_channels = number_channels
        self.lr = lr
        self.data_is_3D = data_is_3D

        self.kwargs = kwargs
        self.cleanup_models = []
        self.layer_data = None
        self.enc_model = None

    def apply_model(self):
        pass

    def apply_prediction_model_to_encoder(self, encoder_model):
        pass

    def get_training_model(self):
        pass

    def get_training_preprocessing(self):
        pass

    def get_finetuning_preprocessing(self):
        def f_identity(x, y):
            return x, y

        return f_identity, f_identity

    def get_finetuning_model(self, model_checkpoint=None):
        model = self.apply_model()
        assert self.enc_model is not None, "no encoder model"

        if model_checkpoint is not None:
            try:
                model.load_weights(model_checkpoint)
            except ValueError:
                model.load_weights(model_checkpoint, by_name=True, skip_mismatch=True)

        self.cleanup_models.append(model)
        self.cleanup_models.append(self.enc_model)

        if self.layer_data:
            if self.data_is_3D:
                self.layer_data.append(isinstance(self.enc_model.layers[-1], Pooling3D))
            else:
                self.layer_data.append(isinstance(self.enc_model.layers[-1], Pooling2D))

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
            try:
                model.load_weights(model_checkpoint)
            except ValueError:
                model.load_weights(model_checkpoint, by_name=True, skip_mismatch=True)

        self.cleanup_models.append(model)
        self.cleanup_models.append(self.enc_model)

        if self.data_is_3D:
            new_enc, self.layer_data = make_finetuning_encoder_3d(
                (self.data_dim, self.data_dim, self.data_dim, self.number_channels,),
                self.enc_model,
                **self.kwargs
            )

            return new_enc
        else:
            new_enc, self.layer_data = make_finetuning_encoder_2d(
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
