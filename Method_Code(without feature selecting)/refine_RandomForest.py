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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore")

train_data = pd.read_csv("../Data/used_car_train_20200313.csv", sep=' ')
test_data = pd.read_csv("../Data/used_car_testB_20200421.csv", sep=' ')

train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
test_data['notRepairedDamage'].replace('-', np.nan, inplace=True)

train_data.drop(train_data[train_data.isnull().any(axis=1)].index, axis=0, inplace=True)
test_data.drop(test_data[test_data.isnull().any(axis=1)].index, axis=0, inplace=True)

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

#标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_data = scaler.fit_transform(test_data)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建管道
pipeline = Pipeline(steps=[
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True))
])


# 参数调优
param_grid = {
    'regressor__n_estimators': [50, 100, 150],
    'regressor__max_depth': [10, 20],
    'regressor__min_samples_split': [5, 10],
    'regressor__min_samples_leaf': [5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 最佳参数
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# 使用最佳参数重新训练模型
best_rf = grid_search.best_estimator_
rf = best_rf

# 预测

y_pred_val = rf.predict(X_val)

# 评估模型

mse_val = sklearn.metrics.mean_squared_error(y_val, y_pred_val)


print(f"Validation Mean Squared Error: {mse_val}")

# 可视化结果

val_results = pd.DataFrame({'Actual': y_val, 'Predicted': y_pred_val})

# 保存结果

val_results.to_csv('../Results/refine_RF_val_predictions.csv', index=False)

# 绘制损失曲线
epochs = list(range(1, 101))  # 假设有100个“epoch”


plt.plot(epochs, [mse_val] * len(epochs))
plt.title('Validation MSE')
plt.xlabel('epochs')
plt.ylabel('val_mse')

plt.tight_layout()

plt.savefig('../Results/refine_RF_loss.png')
plt.show()

