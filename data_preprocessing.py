import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import numpy as np

def load_data(filename):
    return pd.read_csv(filename)

def remove_outliers(df):
    # Идентификация и удаление строк с очевидными выбросами в количестве проданных товаров
    df = df[(df['item_cnt_day'] > 0) & (df['item_cnt_day'] < df['item_cnt_day'].quantile(0.99))]
    df = df[df['item_price'] > df['item_price'].quantile(0.01)]  #  исключаем распродажи по очень низкой цене
    """
    df.sort_values(by='date', inplace=True)
    df['rolling_mean_price'] = df.groupby('item_id')['item_price'].transform(lambda x: x.rolling(window=42, min_periods=1).mean())
    df['price_change_pct'] = np.abs(df['item_price'] - df['rolling_mean_price']) / df['rolling_mean_price'] * 100
    less_than_5_pct_change = df[df['price_change_pct'] < 5]
    df = df.drop(less_than_5_pct_change.index)
    """
    return df

def scale_data(X_train, X_test):
    scaler = StandardScaler(with_mean=False)  # Обратите внимание на параметр with_mean=False
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def preprocess_data(df, train=True, preprocessor=None):
    # Преобразование столбца 'date' в числовые столбцы 'month', 'year', 'day'
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df = remove_outliers(df)
    
    # Удаление столбца 'date' после извлечения необходимой информации
    df = df.drop(columns=['date'])
    
    numeric_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in ['item_category_id', 'item_cnt_day']]
    categorical_features = ['item_category_id']

    if train or preprocessor is None:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        numeric_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'  # сохраняем остальные колонки без изменений
        )
        X = preprocessor.fit_transform(df.drop('item_cnt_day', axis=1))
    else:
        X = preprocessor.transform(df.drop('item_cnt_day', axis=1))

    y = df['item_cnt_day']
    return X, y, preprocessor

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data = load_data('sales_train.csv')
    data_items = load_data('items.csv')
    data = pd.merge(data, data_items, on='item_id')
    data = data.drop(columns='item_name')
    X, y, preprocessor = preprocess_data(data, train=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)