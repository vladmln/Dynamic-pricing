import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from model import PricePredictionModel
from data_preprocessing import load_data, preprocess_data, split_data, scale_data
from torch.utils.data import DataLoader, TensorDataset

def evaluate_model(X_test, y_test, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        # Преобразуем X_test в плотный формат, если он разреженный
        if isinstance(X_test, np.ndarray):
            X_test_tensor = torch.FloatTensor(X_test)
        else:  # Это scipy.sparse matrix
            X_test_tensor = torch.FloatTensor(X_test.toarray())  # Преобразование в плотный формат
        
        X_test_tensor = X_test_tensor.to(device)
        
        # Создаем DataLoader для управляемого перебора данных
        test_dataset = TensorDataset(X_test_tensor, torch.FloatTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            predictions.extend(outputs.cpu().numpy())  # Собираем предсказания
        
    # Расчет метрик
    predictions = np.array(predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Создание таблицы метрик
    metrics = pd.DataFrame({
        'Metric': ['R^2 Score', 'MAE', 'MSE', 'RMSE'],
        'Value': [r2, mae, mse, rmse]
    })

    return metrics


import matplotlib.pyplot as plt

def plot_metrics(metrics):
    fig, ax = plt.subplots()
    ax.barh(metrics['Metric'], metrics['Value'], color='skyblue')
    ax.set_xlabel('Value')
    ax.set_title('Model Performance Metrics')
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')
    data = load_data('sales_train.csv')
    data_items = load_data('items.csv')
    data = pd.merge(data, data_items, on=['item_id'])
    data = data.drop(columns=['item_name'])
    X, y, preprocessor = preprocess_data(data, train=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    model = PricePredictionModel(input_size=X_train_scaled.shape[1]).to(device)

    metrics_table = evaluate_model(X_test_scaled, y_test.values, model, device)
    print(metrics_table)

    # Визуализация метрик
    plot_metrics(metrics_table)
