import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelecter(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self     # nothing to do here

    def transform(self, X):
        return X[self.attribute_names]


class DiscountConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.fillna(0.)

        def convert_discount(x):
            try:
                if ':' in x:
                    x1 = x.split(':')[0]
                    x2 = x.split(':')[1]
                    rate = (float(x1) - float(x2)) / float(x1)
                else:
                    rate = x

            except TypeError:
                rate = 0

            return rate

        X = X.applymap(convert_discount)

        return X


class DateToWeekConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        def convert_date(x):
            try:
                x = x.dt.weekday
            except AttributeError:
                pass

            return x

        X = X.apply(convert_date)
        X.fillna(7, inplace=True)

        return X


class CategoryConverter(BaseEstimator, TransformerMixin):
    def __init__(self, scale=1):
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.ceil(X / self.scale)
        X.where(X > 1, 1, inplace=True)

        return X
