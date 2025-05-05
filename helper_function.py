
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class StringConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Store the feature names to preserve them later
        self.feature_names = X.columns
        return self

    def transform(self, X):
        # Convert all values to string (ensure that we handle all columns correctly)
        X_transformed = X.apply(lambda col: col.astype(str))
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        # Return the feature names to ensure they are consistent with the input
        return self.feature_names


class YearExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_name= X.columns[0]
        return self

    def transform(self, X):
        years = pd.to_datetime(X.iloc[:, 0]).dt.year
        years = years.where(years.isin([2011, 2012, 2013]), 2011) # 2011 is the most frequent values
        return pd.DataFrame(years, columns=[self.feature_name])

    def get_feature_names_out(self, input_features=None):
        return [self.feature_name]

   
    
    
class IQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='clip',multiplier=1.5):
        self.strategy = strategy
        self.multiplier = multiplier

    def fit(self, X, y=None):
        self.feature_name= X.columns[0]
        X_series = X.iloc[:, 0]
        self.q1 = X_series.quantile(0.25)
        self.q3 = X_series.quantile(0.75)
        self.iqr = self.q3 - self.q1
        self.lower_bound = self.q1 - self.multiplier * self.iqr
        self.upper_bound = self.q3 + self.multiplier * self.iqr
        self.mean = X_series.mean()
        self.median = X_series.median()
        return self

    def transform(self, X):
        X_series = X.iloc[:, 0].copy()
        if self.strategy == 'clip':
            X_out = X_series.clip(self.lower_bound, self.upper_bound)
        elif self.strategy == 'mean':
            mask = (X_series < self.lower_bound) | (X_series > self.upper_bound)
            X_out = X_series.mask(mask, self.mean)
        elif self.strategy == 'median':
            mask = (X_series < self.lower_bound) | (X_series > self.upper_bound)
            X_out = X_series.mask(mask, self.median)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return pd.DataFrame(X_out, columns=[self.feature_name])

    def get_feature_names_out(self, input_features=None):
        return [self.feature_name]
    
class ConstructionYearTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_series = X.iloc[:, 0]
        self.feature_name = X.columns[0]
        self.median_non_zero = X_series[X_series != 0].median()
        return self

    def transform(self, X):
        X_series = X.iloc[:, 0].copy()
        X_series = X_series.replace(0, self.median_non_zero)
        return pd.DataFrame(X_series, columns=[self.feature_name])
    
    def get_feature_names_out(self, input_features=None):
        return [self.feature_name]

class ObjectToNumericConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names = X.columns
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            if X_copy[col].dtype == 'object':
                try:
                    X_copy[col] = pd.to_numeric(X_copy[col])
                except ValueError:
                    pass  # Leave non-numeric strings as is
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return self.feature_names if input_features is None else input_features

