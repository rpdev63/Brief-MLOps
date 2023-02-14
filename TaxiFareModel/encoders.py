# create a DistanceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized
import pandas as pd

class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """
    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        # A COMPPLETER
        self.start_lat=start_lat
        self.start_lon=start_lon
        self.end_lat=end_lat
        self.end_lon=end_lon
    
    def fit(self, X, y=None):
        # A COMPLETER 
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['distance'] = haversine_vectorized(X_, self.start_lat, self.start_lon, self.end_lat, self.end_lon)
        return X_[['distance']]



# create a TimeFeaturesEncoder
class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """
    def __init__(self, datetime_column ):
       # A COMPLETER
        self.datetime_column = datetime_column
        
    def fit(self, X, y=None):
        # A COMPLETER
        return self

    def transform(self, X, y=None):
        # A COMPLETER
        X_ = X.copy()
        X_[self.datetime_column] = pd.to_datetime(X_[self.datetime_column])
        X_['dow'] = X_[self.datetime_column].dt.dayofweek
        X_['hour'] = X_[self.datetime_column].dt.hour
        X_['month'] = X_[self.datetime_column].dt.month
        X_['year'] = X_[self.datetime_column].dt.year
        return X_[['dow', 'hour', 'month', 'year']]