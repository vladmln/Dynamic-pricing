import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import PricePredictionModel
from data_preprocessing import load_data, preprocess_data, split_data, scale_data

def train_model(X_train, y_train, model, criterion, optimizer, epochs=5, batch_size=64):
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

if __name__ == "__main__":
    data = load_data('sales_train.csv')
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(processed_data)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    model = PricePredictionModel(input_size=X_train_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(X_train_scaled, y_train.values, model, criterion, optimizer)
