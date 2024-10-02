# author:Liu Yu
# time:2024/10/2 18:14
import pandas as pd
import matplotlib.pyplot as plt

def plot_contrast_graph(path1, path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    x1 = df1['Epochs']
    y1 = df1['val_loss']
    x2 = df2['Epochs']
    y2 = df2['val_loss']
    plt.plot(x1, y1, label='before')
    plt.plot(x2, y2, label='after')

    plt.xlabel('epochs')  # 替换为X轴的标签
    plt.ylabel('val_loss')  # 替换为Y轴的标签
    plt.legend()
    plt.grid(True)
    plt.show()

path1 = '../Results/RF_loss_pd.csv'
path2 = '../New_Results/RF_loss_pd.csv'
plot_contrast_graph(path1, path2)