import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from model import PricePredictionModel
from data_preprocessing import load_data, preprocess_data, split_data, scale_data

def evaluate_model(X_test, y_test, model):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in torch.FloatTensor(X_test):
            outputs = model(inputs.unsqueeze(0)).squeeze().item()
            predictions.append(outputs)
    predictions = np.array(predictions)

    # Расчет метрик
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

if __name__ == "__main__":
    data = load_data('sales_train.csv')
    data_items = load_data('items.csv')
    data = pd.merge(data, data_items, on=['item_id'])
    data = data.drop(columns=['item_name'])
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(processed_data)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    model = PricePredictionModel(input_size=X_train_scaled.shape[1])

    # Предположим, что модель уже обучена и сохранена, загружаем ее
    #model.load_state_dict(torch.load('model.pth'))

    metrics_table = evaluate_model(X_test_scaled, y_test.values, model)
    print(metrics_table)
