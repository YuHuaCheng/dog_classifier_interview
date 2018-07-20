import pandas as pd


class NaiveBayesClassifier:
    likelihood_dict = {}

    def __init__(self, training_data: pd.DataFrame = None):
        self.training_data = training_data

    def load_training_data(self, data_path: str):
        """
        Read from file path then serialize it as a Pandas DataFrame inside self.training_data

        :param data_path: file path for the training data
        :return None
        """
        raise NotImplementedError()

    def train(self, class_label: str):
        """
        The train method simply constructs the likelihood map that needs for making prediction
        """
        self._construct_likelihood_dict(class_label)

    def predict(self, record: dict) -> bool:
        """
        Given a record with features, return the predicted class in boolean (True => positive)
        """
        raise NotImplementedError()

    def _construct_likelihood_dict(self, class_label: str):
        """
        Constructs Naive Bayes likelihood terms that are used in making prediction, e.g.
        P(Y), P(x_1 | Y), P(x_2 | Y), ...

        And stores as a dictionary inside self.likelihood_dict

        :param class_label: class label's column name in self.training_data, assuming every other columns are features
        :return None
        """
        raise NotImplementedError()
