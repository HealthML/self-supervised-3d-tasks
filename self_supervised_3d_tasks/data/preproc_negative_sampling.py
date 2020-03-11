class NegativeSamplingPreprocessing:
    def __init__(self, f_preproc):
        self.f_preproc = f_preproc
        self.f_sample = None

    def set_negative_sampling(self, f_sample):
        self.f_sample = f_sample

    def draw_neg_sample(self, positive_ids):
        return self.f_sample(positive_ids)

    def preprocess_function(self, ids, x, y):
        return self.f_preproc(self, ids, x, y)