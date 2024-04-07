import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    return pd.read_csv(filename)

def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    # Дополнительная предобработка по необходимости
    return df

def split_data(df, test_size=0.2):
    features = df[['date_block_num', 'shop_id', 'item_id', 'item_price', 'month', 'year', 'day']]
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
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(processed_data)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
