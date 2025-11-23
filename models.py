import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin

class TorchRegressor(BaseEstimator, RegressorMixin):
    '''
    Define neural network model using PyTorch
    '''

    def __init__(self, lr=1e-3, hidden=64, epochs=20, dropout=0.2, device=None):
        self.lr = lr
        self.hidden = hidden
        self.epochs = epochs
        self.model = None
        self.dropout = dropout

        # 自动选择设备
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loss_curve_ = []

    
    def _build_model(self, input_dim, output_dim=1):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, output_dim)
        ).to(self.device)
    

    def fit(self, X, Y):
        # numpy → tensor，并放到 GPU
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Y, dtype=torch.float32).view(-1, Y.shape[1]).to(self.device)

        self.model = self._build_model(X.shape[1], output_dim=Y.shape[1])
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()

            self.train_loss_curve_.append(loss.item())
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X).cpu().numpy() # 🔥 输出转回 CPU numpy
        return pred