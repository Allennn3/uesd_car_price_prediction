# author:Liu Yu
# time:2024/9/28 12:18
# author:Liu Yu
# time:2024/9/27 15:15
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings("ignore")

# tensorboard
writer = SummaryWriter("../logs")

train_data = pd.read_csv("../Data/used_car_train_20200313.csv", sep=' ')
test_data = pd.read_csv("../Data/used_car_testB_20200421.csv", sep=' ')

train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
test_data['notRepairedDamage'].replace('-', np.nan, inplace=True)

train_data.drop(train_data[train_data.isnull().any(axis=1)].index, axis=0, inplace=True)
test_data.drop(test_data[test_data.isnull().any(axis=1)].index, axis=0, inplace=True)

# 计算车龄
def calculate_age(regDate):
    regDate = str(regDate)
    day = int(regDate[4:])
    year = int(regDate[:4])
    current = datetime.now()
    current_year = current.year
    current_day_str = str(current.month) + str(current.day)
    current_day = int(current_day_str)
    if ((current_day - day) > 0):
        gap = 0
    else:
        gap = 1

    return current_year - year - gap


train_data['regDate'] = train_data['regDate'].apply(calculate_age)
test_data['regDate'] = test_data['regDate'].apply(calculate_age)
train_data['age'] = train_data['regDate']
test_data['age'] = test_data['regDate']

#删除不需要的标签
label = ['SaleID', 'name', 'creatDate', 'regDate']
train_data.drop(label, axis=1, inplace=True)
test_data.drop(label, axis=1, inplace=True)


X = train_data.drop('price', axis=1)
y = train_data['price']

scaler = StandardScaler()
X = scaler.fit_transform(X)
test_data = scaler.fit_transform(test_data)

#制作dataset
class UsedCarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#划分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = X_train.astype(float), X_val.astype(float), y_train.astype(float), y_val.astype(float)

train_dataset = UsedCarDataset(X_train, y_train)
val_dataset = UsedCarDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class MLPModel(nn.Module):
    def __init__(self, num_features):
        super(MLPModel, self).__init__()
        self.hidden1 = nn.Linear(num_features, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.hidden3 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(p=0.5)  # 添加 dropout 层
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.batchnorm3 = nn.BatchNorm1d(16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.hidden1(x)))
        x = self.relu(self.batchnorm2(self.hidden2(x)))
        x = self.relu(self.batchnorm3(self.hidden3(x)))
        x = self.dropout(x)
        return self.output(x)

model = MLPModel(X_train.shape[1])

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

# 学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

num_epochs = 100
train_loss_list, val_loss_list = [], []

# 记录训练的次数
total_train_step, total_val_step = 0, 0

# 记录测试的次数
total_test_step = 0

#初始化一个dataframe用于记录和比较验证集
val_results = pd.DataFrame(columns=['Actual', 'Predicted'])
loss_records = pd.DataFrame(columns=['Epochs', 'train_loss', 'val_loss'])

for epoch in range(num_epochs):
    total_train_loss = 0
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() / len(inputs)
        total_train_step = total_train_step + 1
        if total_train_step % 300 == 0:
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("total_train_loss", loss.item(), total_train_step)

    print(f'Epoch {epoch+1}/{num_epochs}, total_train_Loss: {total_train_loss}')
    train_loss_list.append(total_train_loss)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_val_loss = total_val_loss + loss.item() / len(inputs)
            batch_results = pd.DataFrame({'Actual': targets.cpu().numpy().flatten(),
                                          'Predicted': outputs.cpu().numpy().flatten()})
            val_results = pd.concat([val_results, batch_results], ignore_index=True)
    total_val_step = total_val_step + 1
    writer.add_scalar("total_val_loss", total_val_loss, total_val_step)
    print(f"Epoch {epoch+1}/{num_epochs}, total_val_Loss: {total_val_loss}")
    val_loss_list.append(total_val_loss)
    # 更新学习率
    scheduler.step(total_val_loss)

    batch_loss = pd.DataFrame({'Epochs': epoch+1,
                               'train_loss': total_train_loss,
                               'val_loss': total_val_loss}, index=[0])
    loss_records = pd.concat([loss_records, batch_loss], ignore_index=True)

torch.save(model, '../model/MLP_model.pth')

val_results.to_csv('../Results/MLP_predictions.csv', index=False)
loss_records.to_csv('../Results/MLP_loss_records.csv', index=False)


my_list = list(range(1, num_epochs+1))
loss_pd = pd.DataFrame({'Epochs': my_list,
                       'train_loss': train_loss_list,
                       'val_loss': val_loss_list})
loss_pd.to_csv('../Results/MLP_loss_pd.csv', index=False)

fig, axs = plt.subplots(2, 1)
axs[0].plot(my_list, train_loss_list)
axs[0].set_title('Training Loss')
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('train_loss')

axs[1].plot(my_list, val_loss_list)
axs[1].set_title('val Loss')
axs[1].set_xlabel('epochs')
axs[1].set_ylabel('val_loss')

plt.tight_layout()

plt.savefig('../Results/MLP_loss.png')
plt.show()
writer.close()

