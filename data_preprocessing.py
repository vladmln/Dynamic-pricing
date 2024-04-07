import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    return pd.read_csv(filename)

def remove_outliers(df):
    """Удаление выбросов из данных."""
    # Идентификация и удаление строк с очевидными выбросами в количестве проданных товаров
    df = df[(df['item_cnt_day'] > 0) & (df['item_cnt_day'] < df['item_cnt_day'].quantile(0.99))]
    df = df[df['item_price'] > df['item_price'].quantile(0.01)]  #  исключаем распродажи по очень низкой цене
    return df

def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df['item_category_id'] = df['item_category_id'] / df['item_category_id'].sum()
    df = remove_outliers(df)
    # Дополнительная предобработка по необходимости
    return df

def split_data(df, test_size=0.2):
    features = df[['item_id', 'shop_id', 'item_category_id', 'item_price', 'month', 'year', 'day']]
    #features = features.reindex(columns=['item_id', 'shop_id', 'item_category_id', 'item_price', 'month', 'year', 'day'])
    target = df['item_cnt_day']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    data = load_data('sales_train.csv')
    data_items = load_data('items.csv')
    data = pd.merge(data, data_items, on=['item_id'])
    data = data.drop(columns=['item_name'])
    #data = data.reindex(columns=['date', 'date_block_num','item_cnt_day', 'item_id', 'shop_id', 'item_category_id', 'item_price', 'month', 'year', 'day'])
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(processed_data)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)