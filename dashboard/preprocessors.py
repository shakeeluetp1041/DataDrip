import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.sparse import issparse
import category_encoders as ce


class LowerCaseStrings(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self._str_cols = X.select_dtypes(include=["object", "string"]).columns
        return self

    def transform(self, X):
        X = X.copy()
        for col in self._str_cols:
            if col in X.columns:
                X[col] = X[col].astype("string").str.lower()
        return X


class StringConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names = X.columns
        return self

    def transform(self, X):
        return X.apply(lambda col: col.astype(str))

    def get_feature_names_out(self, input_features=None):
        return self.feature_names


class YearExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_name = X.columns[0]
        return self

    def transform(self, X):
        years = pd.to_datetime(X.iloc[:, 0]).dt.year
        years = years.where(years.isin([2011, 2012, 2013]), 2011)
        return pd.DataFrame(years, columns=[self.feature_name])

    def get_feature_names_out(self, input_features=None):
        return [self.feature_name]


class IQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='clip', multiplier=1.5):
        self.strategy = strategy
        self.multiplier = multiplier

    def fit(self, X, y=None):
        self.feature_name = X.columns[0]
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
        self.feature_name = X.columns[0]
        self.median_non_zero = X[X[self.feature_name] != 0][self.feature_name].median()
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.feature_name] = X_copy[self.feature_name].replace(0, self.median_non_zero)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return [self.feature_name]


class ObjectToNumericConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, 'columns'):
            X_copy = X.copy()
            for col in X_copy.columns:
                if X_copy[col].dtype == 'object':
                    try:
                        X_copy[col] = pd.to_numeric(X_copy[col])
                    except ValueError:
                        pass
            return X_copy
        else:
            return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names


class AgeCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, record_col='date_recorded', install_col='construction_year'):
        self.record_col = record_col
        self.install_col = install_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        record_year = pd.to_datetime(X[self.record_col]).dt.year
        age = record_year - X[self.install_col]
        return pd.DataFrame({'age': age})

    def get_feature_names_out(self, input_features=None):
        return ['age']


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.freq_maps = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            freq = X[col].value_counts(normalize=True)
            self.freq_maps[col] = freq.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            freq_map = self.freq_maps.get(col, {})
            X[col] = X[col].map(freq_map).fillna(0.0)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns


class RegionCodeCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, region_col='region', code_col='region_code', new_col='region_with_code'):
        self.region_col = region_col
        self.code_col = code_col
        self.new_col = new_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.region_col in X.columns and self.code_col in X.columns:
            X[self.new_col] = X[self.region_col].astype(str) + "_" + X[self.code_col].astype(str)
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.columns_to_drop:
            return X.drop(columns=[col for col in self.columns_to_drop if col in X.columns])
        return X


class AgePipeline(SklearnPipeline):
    def get_feature_names_out(self, input_features=None):
        return ['age']


class GeoContextImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.geo_group_cols = ['subvillage', 'ward', 'lga', 'district_code', 'region_code']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Fill subvillage if blank
        if 'subvillage' in X.columns:
            X['subvillage'] = X['subvillage'].replace('', pd.NA)
            X['subvillage'] = X.apply(lambda row: self.fill_subvillage(row, X), axis=1)

        # Treat near-zero lat/lon as missing
        for col in ['latitude', 'longitude']:
            if col in X.columns:
                X[col] = X[col].apply(lambda val: pd.NA if abs(val) < 1e-6 else val)

        # Fill geo-based mean for lat/lon
        if 'latitude' in X.columns and 'longitude' in X.columns:
            X = self.fill_missing_geo(X, self.geo_group_cols)

        # Fill population groupwise (median)
        if 'population' in X.columns:
            X['population'] = X['population'].apply(lambda x: pd.NA if x in [0, 1] else x)
            X = self.geo_groupwise_fill(X, 'population', self.geo_group_cols)

        # Fill amount_tsh groupwise (median)
        if 'amount_tsh' in X.columns:
            X['amount_tsh'] = X['amount_tsh'].apply(lambda x: pd.NA if x in [0, 1] else x)
            X = self.geo_groupwise_fill(X, 'amount_tsh', self.geo_group_cols)

        # Fill categorical fields groupwise (mode)
        for col in ['permit', 'public_meeting', 'construction_year',
                    'wpt_name', 'scheme_management', 'funder',
                    'installer', 'scheme_name']:
            if col in X.columns:
                X = self.geo_groupwise_fill_mode(X, col, self.geo_group_cols)

        # Fill gps_height: treat zero as missing, then groupwise
        if 'gps_height' in X.columns:
            X['gps_height'] = X['gps_height'].apply(lambda val: pd.NA if val == 0 else val)
            X = self.geo_groupwise_fill(X, 'gps_height', self.geo_group_cols)

        # Fill wpt_name blank with 'none'
        if 'wpt_name' in X.columns:
            X['wpt_name'] = X['wpt_name'].fillna('none')

        # Fill scheme_management blank with 'other'
        if 'scheme_management' in X.columns:
            X['scheme_management'] = X['scheme_management'].fillna('other')

        # Replace all pandas NA values with numpy NaN to avoid downcasting warnings
        return X.fillna(np.nan)

    def fill_subvillage(self, row, df):
        if pd.isna(row['subvillage']):
            for level in ['ward', 'lga', 'district_code']:
                mode_val = df[df[level] == row[level]]['subvillage'].mode()
                if not mode_val.empty:
                    return mode_val[0]
            return 'unknown'
        return row['subvillage']

    def fill_missing_geo(self, df, group_cols):
        for col in group_cols:
            group_means = df.dropna(subset=['latitude', 'longitude']).groupby(col)[['latitude', 'longitude']].mean()
            def _fill_row(r):
                if pd.isna(r['latitude']) or pd.isna(r['longitude']):
                    key = r[col]
                    if key in group_means.index:
                        if pd.isna(r['latitude']):
                            r['latitude'] = group_means.loc[key, 'latitude']
                        if pd.isna(r['longitude']):
                            r['longitude'] = group_means.loc[key, 'longitude']
                return r
            df = df.apply(_fill_row, axis=1)
            if df['latitude'].isna().sum() == 0 and df['longitude'].isna().sum() == 0:
                break
        return df

    def geo_groupwise_fill(self, df, target_col, group_cols):
        for col in group_cols:
            group_medians = df.dropna(subset=[target_col]).groupby(col)[target_col].median()
            def _fill_row(r):
                if pd.isna(r[target_col]):
                    key = r[col]
                    if key in group_medians.index:
                        r[target_col] = group_medians.loc[key]
                return r
            df = df.apply(_fill_row, axis=1)
            if df[target_col].isna().sum() == 0:
                break
        return df

    def geo_groupwise_fill_mode(self, df, target_col, group_cols):
        for col in group_cols:
            group_modes = df.dropna(subset=[target_col]).groupby(col)[target_col].agg(lambda x: x.mode().iloc[0])
            def _fill_row(r):
                if pd.isna(r[target_col]):
                    key = r[col]
                    if key in group_modes.index:
                        r[target_col] = group_modes.loc[key]
                return r
            df = df.apply(_fill_row, axis=1)
            if df[target_col].isna().sum() == 0:
                break
        return df
