from sklearn.utils import shuffle

from sklearn.base import BaseEstimator, RegressorMixin

class PseudoLabeler(BaseEstimator, RegressorMixin):

    """
    Sci-kit learn wrapper for creating pseudo-lebeled estimators.
    """
    def __init__(self, model, unlabled_data, features, target, sample_rate=0.2, seed=42):

        """
             @sample_rate - percent of samples used as pseudo-labelled data
         from the unlabelled dataset

        """

        assert sample_rate <= 1.0, 'Sample_rate should be between 0.0 and 1.0.'

        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed

        self.unlabled_data = unlabled_data
        self.features = features
        self.target = target

