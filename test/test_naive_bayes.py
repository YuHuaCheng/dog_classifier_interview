import os

from src.naive_bayes_classifier import NaiveBayesClassifier


class TestNBClassifier:

    nbc = NaiveBayesClassifier()
    dirname = os.path.dirname(__file__)
    training_data_path = os.path.join(dirname, './resources/training_data.txt')

    def test_load_data(self):
        self.nbc.load_training_data(self.training_data_path)
        assert self.nbc.training_data.shape == (5, 5)
        assert list(self.nbc.training_data.columns) == ['dog', 'num_legs', 'has_tail', 'is_cute', 'hairy']

    def test_train(self):
        class_label = 'dog'
        self.nbc.train(class_label)
        assert self.nbc.likelihood_dict is not None

    def test_predict(self):
        record = {'num_legs': 4, 'has_tail': 1, 'is_cute': 1}
        assert self.nbc.predict(record)  # this should be True (a dog)

        record = {'num_legs': 8, 'has_tail': 0, 'is_cute': 0}
        assert not self.nbc.predict(record)  # this should be not
