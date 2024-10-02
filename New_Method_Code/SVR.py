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
import warnings
from sklearn.svm import SVR


warnings.filterwarnings("ignore")



train_data = pd.read_csv("../Data/new_trainData.csv")
test_data = pd.read_csv("../Data/used_car_testB_20200421.csv", sep=' ')

# train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
# test_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
#
# train_data.drop(train_data[train_data.isnull().any(axis=1)].index, axis=0, inplace=True)
# test_data.drop(test_data[test_data.isnull().any(axis=1)].index, axis=0, inplace=True)
"""

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

"""
y = train_data['price']
X = train_data.drop('price', axis=1)


scaler = StandardScaler()
X = scaler.fit_transform(X)


#划分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = X_train.astype(float), X_val.astype(float), y_train.astype(float), y_val.astype(float)

# 定义 SVR 模型
svr = SVR(kernel='rbf', C=1, epsilon=0.1)

# 训练模型
svr.fit(X_train, y_train)

print("========训练完成========")
# 预测

y_pred_val = svr.predict(X_val)

# 评估模型

mse_val = sklearn.metrics.mean_squared_error(y_val, y_pred_val)

print(f"Validation Mean Squared Error: {mse_val}")

# 可视化结果

val_results = pd.DataFrame({'Actual': y_val, 'Predicted': y_pred_val})

# 保存结果

val_results.to_csv('../New_Results/SVR_val_predictions.csv', index=False)

# 绘制损失曲线
epochs = list(range(1, 101))  # 假设有100个“epoch”
loss_pd = pd.DataFrame({'Epochs': epochs,
                       'val_loss': [mse_val] * len(epochs)})
loss_pd.to_csv('../New_Results/SVR_loss_pd.csv', index=False)

plt.plot(epochs, [mse_val] * len(epochs))
plt.xlabel('epochs')
plt.ylabel('mse_val')
plt.title('validation MSE')


plt.tight_layout()

plt.savefig('../New_Results/SVR_loss.png')
plt.show()

