import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_values_heatmap(dataset):
    plt.figure(figsize=(10,5))
    ax = sns.heatmap(dataset.isna(), cbar=False)
# 获取索引
    dates = dataset.index

# 每1024条数据显示一个标签（假设你的数据是每小时一次）
    ax.set_yticks(range(0, len(dates), 1024))
    ax.set_yticklabels([d.strftime('%y-%m-%d') for d in dates[::1024]], fontsize=7)
    plt.title("Missing Values Heatmap (-200 replaced by NaN)")
    plt.show()

def plot_scatter_each_column_rolling(dataset, window=1):
    for col in dataset.columns:
        plt.figure(figsize=(10, 4))

    # 计算滑动平均
        rolling_values = dataset[col].rolling(window=window, min_periods=1).mean()
        plt.scatter(dataset.index, rolling_values, s=5)
        plt.title(f"Rolling Mean Scatter Plot ({window}) for {col}")
        plt.xlabel("Datetime")
        plt.ylabel(f"{col} (Rolling Mean)")
        plt.tight_layout()
        plt.show()

def plot_scatter_each_column_highlight_missing(dataset, imputed_dataset):
    for col in dataset.columns:
        # 原始的 NaN mask（True = 原来是缺失）
        nan_mask = dataset[col].isna()

        plt.figure(figsize=(12,5))

        # 蓝色点：原本非缺失位置
        plt.scatter(dataset.index[~nan_mask],
                    imputed_dataset[col][~nan_mask],
                    s=12, label='Original values', alpha=0.7)

        # 红色点：原本是缺失 → 用填充值显示
        plt.scatter(dataset.index[nan_mask],
                    imputed_dataset[col][nan_mask],
                    s=12, color='red', label='Imputed values')

        plt.title(f"Imputed values highlighted in red: {col}")
        plt.legend()
        plt.show()