# TODO: Refactor
class AlgorithmBuilderBase:
    def __init__(
            self,
            data_dim=384,
            number_channels=3,
            lr=1e-4,
            train3D=False,
            top_architecture="big_fully",
            **kwargs
    ):
        self.top_architecture = top_architecture
        self.data_dim = data_dim

        self.number_channels = number_channels
        self.lr = lr
        self.train3D = train3D

        self.kwargs = kwargs
        self.cleanup_models = []
        self.layer_data = None
        self.enc_model = None

    def apply_model(self, input_dim):
        pass

    def get_training_model(self):
        pass

    def get_finetuning_preprocessing(self):
        pass

    def get_finetuning_model(self, model_checkpoint=None):
        pass

    def purge(self):
        pass