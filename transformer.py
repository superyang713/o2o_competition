from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameConverter(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self     # nothing to do here

    def transform(self, X):
        return X[self.attribute_names].values


class DiscountConverter(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_name='Discount_rate'):
        self.attribute_name = attribute_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        def convert_discount(x):
            x = str(x)
            if ':' in x:
                x1 = x.split(':')[0]
                x2 = x.split(':')[1]
                rate = (float(x1) - float(x2)) / float(x1)

            else:
                try:
                    rate = float(x)
                except ValueError:
                    rate = x

            return rate

        X[self.attribute_name] = X[self.attribute_name].apply(convert_discount)

        return X
