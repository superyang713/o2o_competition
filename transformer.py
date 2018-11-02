import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        return X_copy[self.attribute_names]


class DiscountConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy = X_copy.fillna(0.)

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

        X_copy = X_copy.applymap(convert_discount)

        return X_copy.astype(float)


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

        X_copy = X.copy()
        X_copy = X_copy.apply(convert_date)
        X_copy.fillna(7, inplace=True)

        return X_copy


class CategoryConverter(BaseEstimator, TransformerMixin):
    def __init__(self, scale=1):
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = np.ceil(X_copy / self.scale)
        X_copy.where(X_copy > 1, 1, inplace=True)

        return X_copy
