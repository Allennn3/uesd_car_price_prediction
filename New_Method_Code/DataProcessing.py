# author:Zhuoying Li
# time:2024/10/2 16:31
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from lightgbm import log_evaluation, early_stopping
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)

Train_data = pd.read_csv("../Data/used_car_train_20200313.csv", sep=' ')
Test_data = pd.read_csv("../Data/used_car_testB_20200421.csv", sep=' ')

#用众数填补空缺值
Train_data['bodyType'] = Train_data['bodyType'].fillna(0)
Train_data['fuelType'] = Train_data['fuelType'].fillna(0)
Train_data['gearbox'] = Train_data['gearbox'].fillna(0)
Train_data['model'] = Train_data['model'].fillna(0)

#将seller其中的异常值1改为0
Train_data['seller'] = Train_data['seller'][Train_data['seller']==1]=0

Train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
Train_data['notRepairedDamage'] = Train_data['notRepairedDamage'].astype(float)

#异常值截断
Train_data['power'][Train_data['power'] > 600] = 600
Train_data['power'][Train_data['power'] < 1] = 1
Train_data['v_13'][Train_data['v_13'] > 6] = 6
Train_data['v_14'][Train_data['v_14'] > 4] = 4

#删除取值没有变化的列
Train_data = Train_data.drop(['seller', 'offerType'], axis=1)

# 计算某品牌的销售统计量
Train_gb = Train_data.groupby("brand")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data.price.max()
    info['brand_price_median'] = kind_data.price.median()
    info['brand_price_min'] = kind_data.price.min()
    info['brand_price_sum'] = kind_data.price.sum()
    info['brand_price_std'] = kind_data.price.std()
    info['brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2)
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})
Train_data = Train_data.merge(brand_fe, how='left', on='brand')

# 使用时间：
# 数据里有时间出错的格式，errors='coerce'，遇到不能转换的数据赋值为nan
Train_data['used_time'] = (pd.to_datetime(Train_data['creatDate'], format='%Y%m%d', errors='coerce') -
                            pd.to_datetime(Train_data['regDate'], format='%Y%m%d', errors='coerce')).dt.days
Train_data['used_time'].mean()

#用平均数或众数填充缺失值
Train_data['used_time'].fillna(round(Train_data['used_time'].mean()), inplace=True)

#对连续型数据进行分桶
#对power进行分桶,分成31个桶
bin = [i*10 for i in range(31)]
Train_data['power_bin'] = pd.cut(Train_data['power'], bin, labels=False)

#删除缺失值的行
Train_data.drop(Train_data[Train_data.isnull().any(axis=1)].index, axis=0, inplace=True)

# 删除不需要的数据
y_train = Train_data['price']
x_train = Train_data.drop(['name','SaleID', 'regionCode', 'price', 'regDate', 'creatDate'], axis=1)


features = pd.get_dummies(x_train)
feature_names = list(features.columns)
features = np.array(features)

labels = np.array(y_train).reshape((-1, ))
feature_importance_values = np.zeros(len(feature_names))
task = 'regression'
early_stopping = True
eval_metric = 'l2'
n_iterations = 10
# callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=100)]

for _ in range(n_iterations):
    if task == 'classification':
        model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1)
    if task =='regression':
        model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1)
    else:
        raise ValueError('Task must be either "classification" or "regression"')
    #提前终止训练，需要验证集
    if early_stopping:
        train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels, test_size = 0.15)
  # Train the model with early stopping
        model.fit(train_features, train_labels, eval_metric = eval_metric,eval_set = [(valid_features, valid_labels)])
        # gc.enable()
        del train_features, train_labels, valid_features, valid_labels
        # gc.collect()

    else:
        model.fit(features, labels)
  # Record the feature importances
    feature_importance_values += model.feature_importances_ / n_iterations
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
#按照重要性大小对特征进行排序
feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)

#计算特征的相对重要性，全部特征的相对重要性之和为1
feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()

#计算特征的累计重要性
#cutsum :返回给定 axis 上的累计和
feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

#选取累计重要性大于0.99的特征，删除
drop_columns=list(feature_importances.query('cumulative_importance>0.99')['feature'])
print(drop_columns)

Train_data = Train_data.drop(drop_columns, axis=1)
print(Train_data.shape)

Train_data.to_csv("../Data/new_trainData.csv", index=0)



