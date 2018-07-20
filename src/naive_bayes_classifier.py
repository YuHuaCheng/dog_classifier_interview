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
        data = pd.read_csv(data_path)
        self.training_data = data

    def train(self, class_label: str):
        """
        The train method simply constructs the likelihood map that needs for making prediction
        """
        self._construct_likelihood_dict(class_label)

    def predict(self, record: dict) -> bool:
        """
        Given a record with features, return the predicted class in boolean (True => positive)
        """
        EPSILON = 0.01  # small constant to not let probability vanish because of missing observation
        prediction = {}

        for label in [0, 1]:
            unnormalized_prob = self.likelihood_dict.get('prior').get(label)
            for key, val in record.items():
                unnormalized_prob *= self.likelihood_dict.get(key).get(label).get(val, EPSILON)
            prediction[label] = unnormalized_prob

        return max(prediction, key=lambda k: prediction[k]) == 1

    def _construct_likelihood_dict(self, class_label: str):
        """
        Constructs Naive Bayes likelihood terms that are used in making prediction, e.g.
        P(Y), P(x_1 | Y), P(x_2 | Y), ...

        And stores as a dictionary inside self.likelihood_dict

        :param class_label: class label's column name in self.training_data, assuming every other columns are features
        :return None
        """

        # compute prior first
        prior_positive = self.training_data[class_label].mean()
        self.likelihood_dict['prior'] = {1: prior_positive, 0: 1 - prior_positive}

        # for every feature x, compute P(x | y)
        for feature in self.training_data.columns:
            if feature != class_label:
                feature_prob_dict = {}
                for label in [0, 1]:
                    feature_cnt = self.training_data.ix[self.training_data[class_label] == label][feature].value_counts()
                    feature_prob = feature_cnt / feature_cnt.sum()
                    feature_prob_dict[label] = feature_prob.to_dict()
                self.likelihood_dict[feature] = feature_prob_dict
