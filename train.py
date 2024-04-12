import torch
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import PricePredictionModel
from data_preprocessing import load_data, preprocess_data, split_data, scale_data
import numpy as np

def train_model(X_train, y_train, model, criterion, optimizer, epochs=5, batch_size=16):
    # Проверяем, является ли X_train разреженной матрицей и преобразуем её в плотный формат
    if isinstance(X_train, np.ndarray):
        X_train_tensor = torch.FloatTensor(X_train)
    else:  # Если это scipy.sparse matrix
        X_train_tensor = torch.FloatTensor(X_train.toarray())  # Преобразуем в плотный формат
    
    y_train_tensor = torch.FloatTensor(y_train)

    # Перемещаем тензоры на тот же девайс, что и модель
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)

    # Создаем DataLoader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Цикл обучения
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for features, labels in loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()  # Добавляем squeeze для устранения лишнего измерения
            loss = criterion(outputs, labels)  # Теперь размеры совпадают
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Avg Loss: {running_loss / len(loader)}')
    model.apply(reset_weights)

def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(X_train_scaled, y_train.values, model, criterion, optimizer)
