import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class WalmartDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features_df, stores_df):
        self.features_df = features_df
        self.stores_df = stores_df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df['Date'] = pd.to_datetime(df['Date'])
        self.features_df['Date'] = pd.to_datetime(self.features_df['Date'])
        if 'Date' in self.stores_df.columns:
            self.stores_df['Date'] = pd.to_datetime(self.stores_df['Date'])

        df = df.merge(self.stores_df, on='Store', how='left')
        df = df.merge(self.features_df, on=['Store', 'Date'], how='left')

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week

        if 'IsHoliday_x' in df.columns:
            df['IsHoliday'] = df['IsHoliday_x'].astype(int)
        elif 'IsHoliday' in df.columns:
            df['IsHoliday'] = df['IsHoliday'].astype(int)

        df['Type'] = df['Type'].astype('category').cat.codes

        features_to_drop = ['Date', 'IsHoliday_x', 'IsHoliday_y']
        df = df.drop(columns=features_to_drop, errors='ignore')

        return df

class TimeSeriesSplitter:
    def __init__(self, split_date='2012-01-01', target_col='Weekly_Sales', date_col='Date'):
        self.split_date = pd.to_datetime(split_date)
        self.target_col = target_col
        self.date_col = date_col

    def split(self, df):
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        train_mask = df[self.date_col] < self.split_date
        val_mask = df[self.date_col] >= self.split_date

        X_train = df[train_mask].drop(columns=[self.target_col], errors='ignore')
        y_train = df[train_mask][self.target_col] if self.target_col in df.columns else None

        X_val = df[val_mask].drop(columns=[self.target_col], errors='ignore')
        y_val = df[val_mask][self.target_col] if self.target_col in df.columns else None

        return X_train, y_train, X_val, y_val

class WalmartDataTransformer_updatedFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, features_df, stores_df):
        self.features_df = features_df
        self.stores_df = stores_df
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        self.features_df['Date'] = pd.to_datetime(self.features_df['Date'])
        
        df = df.merge(self.stores_df, on='Store', how='left')
        df = df.merge(self.features_df, on=['Store', 'Date'], how='left')
        
        df['CPI'] = df['CPI'].fillna(df['CPI'].median())
        df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].median())
      
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
        
        df['Is_December'] = (df['Month'] == 12).astype(int)
        df['Is_November'] = (df['Month'] == 11).astype(int)
        
        if 'IsHoliday_x' in df.columns:
            df['IsHoliday'] = df['IsHoliday_x']
        elif 'IsHoliday' not in df.columns and 'IsHoliday_y' in df.columns:
            df['IsHoliday'] = df['IsHoliday_y']
            
        if 'IsHoliday' in df.columns:
            df['IsHoliday'] = df['IsHoliday'].astype(int)
        else:
            df['IsHoliday'] = 0
            
        # 5. კატეგორიული სვეტის კოდირება (Type)
        if 'Type' in df.columns:
            df['Type'] = df['Type'].astype('category').cat.codes
            
        # 6. ზედმეტი სვეტების წაშლა (სწორი სახელებით!)
        drop_cols = ['Date', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'IsHoliday_x', 'IsHoliday_y']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        return df

class WalmartLagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, full_df):
        # only this columns that are neseccary for lag 
        self.sales_history = full_df[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()
        self.sales_history['Date'] = pd.to_datetime(self.sales_history['Date'])
        
        self.sales_history = self.sales_history.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)
        
        # create lags
        self.sales_history['Lag_1'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
        self.sales_history['Lag_2'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(2)
        self.sales_history['Lag_4'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(4)
        self.sales_history['Rolling_Mean_4'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(window=4).mean()
        
        
        lag_cols = ['Lag_1', 'Lag_2', 'Lag_4', 'Rolling_Mean_4']
        self.sales_history[lag_cols] = self.sales_history[lag_cols].fillna(0)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        df['Date'] = pd.to_datetime(df['Date'])
 
        df = df.merge(
            self.sales_history[['Store', 'Dept', 'Date', 'Lag_1', 'Lag_2', 'Lag_4', 'Rolling_Mean_4']], 
            on=['Store', 'Dept', 'Date'], 
            how='left'
        )
   
        lag_cols = ['Lag_1', 'Lag_2', 'Lag_4', 'Rolling_Mean_4']
        df[lag_cols] = df[lag_cols].fillna(0)
        
        return df

class WalmartLagTransformer_v2(BaseEstimator, TransformerMixin):
    def __init__(self, full_df):
        self.sales_history = full_df[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()
        self.sales_history['Date'] = pd.to_datetime(self.sales_history['Date'])
        self.sales_history = self.sales_history.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)
        
        # ლაგების შექმნა
        self.sales_history['Lag_1'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
        self.sales_history['Lag_2'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(2)
        self.sales_history['Lag_4'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(4)
        self.sales_history['Lag_52'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(52) # <- წლიური ლაგი
        self.sales_history['Rolling_Mean_4'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(window=4).mean()
        
        # აქ ფილნას არ ვუკეთებთ სრულ ისტორიას, რათა მოდელმა ნამდვილი NaN დაინახოს საწყის კვირებში
        # და ჩვენ ისინი ქვემოთ, ფილტრაციის დროს მოვაშოროთ.
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        df = df.merge(
            self.sales_history[['Store', 'Dept', 'Date', 'Lag_1', 'Lag_2', 'Lag_4', 'Lag_52', 'Rolling_Mean_4']], 
            on=['Store', 'Dept', 'Date'], 
            how='left'
        )
        
        # თუ ტესტირებისას მაინც სადმე ახალი NaN შეგვხვდა, უსაფრთხოებისთვის შევავსოთ 0-ით
        lag_cols = ['Lag_1', 'Lag_2', 'Lag_4', 'Lag_52', 'Rolling_Mean_4']
        df[lag_cols] = df[lag_cols].fillna(0)
        return df

class WalmartLagTransformer_v2(BaseEstimator, TransformerMixin):
    def __init__(self, full_df):
        self.sales_history = full_df[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()
        self.sales_history['Date'] = pd.to_datetime(self.sales_history['Date'])
        self.sales_history = self.sales_history.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)

        # ლაგების შექმნა
        self.sales_history['Lag_1'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
        self.sales_history['Lag_2'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(2)
        self.sales_history['Lag_4'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(4)
        self.sales_history['Lag_52'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(52) # <- წლიური ლაგი
        self.sales_history['Rolling_Mean_4'] = self.sales_history.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(window=4).mean()

        # აქ ფილნას არ ვუკეთებთ სრულ ისტორიას, რათა მოდელმა ნამდვილი NaN დაინახოს საწყის კვირებში
        # და ჩვენ ისინი ქვემოთ, ფილტრაციის დროს მოვაშოროთ.

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['Date'] = pd.to_datetime(df['Date'])

        df = df.merge(
            self.sales_history[['Store', 'Dept', 'Date', 'Lag_1', 'Lag_2', 'Lag_4', 'Lag_52', 'Rolling_Mean_4']],
            on=['Store', 'Dept', 'Date'],
            how='left'
        )

        # თუ ტესტირებისას მაინც სადმე ახალი NaN შეგვხვდა, უსაფრთხოებისთვის შევავსოთ 0-ით
        lag_cols = ['Lag_1', 'Lag_2', 'Lag_4', 'Lag_52', 'Rolling_Mean_4']
        df[lag_cols] = df[lag_cols].fillna(0)
        return df
