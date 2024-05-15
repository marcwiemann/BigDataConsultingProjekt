import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

df = pd.read_csv('creditcard.csv')

x = df.drop('Class', axis=1)
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train.values, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)
y_test = torch.tensor(y_test.values, dtype=torch.float)
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(x_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    
if torch.backends.mps.is_available():
    device = torch.device("mps")
    
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")
print(device)

x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

model = Model().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
writer = SummaryWriter('runs/creditcard_experiment')

def calculate_accuracy(y_pred, y_true):
    predicted = y_pred.round()  
    correct = (predicted == y_true).float()  
    accuracy = correct.sum() / len(correct)  
    return accuracy


epochs = 500
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_acc = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.view(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += calculate_accuracy(output, target.view(-1, 1))

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)

    print(f'Epoch {epoch+1}, Loss: {train_loss:.6f}, Accuracy: {train_acc:.6f}')

    
    if (epoch+1) % 100 == 0:
        torch.save(model.state_dict(), f'models/creditcard_{epoch+1}.pt')

model.eval()
with torch.no_grad():
    test_preds = model(x_test.to(device))
    test_accuracy = calculate_accuracy(test_preds, y_test.to(device))
    writer.add_scalar('Accuracy/test', test_accuracy, epoch)
    print(f'Test Accuracy: {test_accuracy:.4f}')

torch.save(model.state_dict(), f'models/creditcard_{pd.Timestamp.now()}.pt')
writer.close()