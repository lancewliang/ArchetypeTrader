import numpy as np
import pandas as pd
import os
import pickle
import logging
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool, cpu_count
from functools import partial

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def smooth_data(data):
    #使用低通巴特沃斯滤波器对时间序列数据进行平滑处理,去除高频噪声。
    N, Wn = 1, 0.05
    b, a = butter(N, Wn, btype='low')
    return filtfilt(b, a, data)

def get_slope(smoothed_data):
    #通过线性回归计算时间序列的趋势斜率,用于量化价格走势的方向和强度。
    X = np.arange(len(smoothed_data)).reshape(-1, 1)
    model = LinearRegression().fit(X, smoothed_data)
    return model.coef_[0]
N, Wn = 1, 0.05
bb, aa = butter(N, Wn, btype='low')
def get_slope_window(window):

    y = filtfilt(bb, aa, window.values)
    X = np.arange(len(y)).reshape(-1, 1)   
    model = LinearRegression().fit(X, y)
    return model.coef_[0]

def chunk(df_train, df_val, df_test):
    chunk_size = 4320
    for i in range(int(len(df_train) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        df_chunk = df_train[start:end].reset_index(drop=True)
        df_chunk.to_feather('./data/AL/train/df_{}.feather'.format(i))

    for i in range(int(len(df_val) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        df_chunk = df_val[start:end].reset_index(drop=True)
        df_chunk.to_feather('./data/AL/val/df_{}.feather'.format(i))

    for i in range(int(len(df_test) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        df_chunk = df_test[start:end].reset_index(drop=True)
        df_chunk.to_feather('./data/AL/test/df_{}.feather'.format(i))

def label_slope(df_train, df_val, df_test):
    """
    趋势标签生成
    为训练集、验证集和测试集的时间序列数据添加趋势斜率标签
    
    参数:
        df_train (pd.DataFrame): 训练集数据,包含'close'列
        df_val (pd.DataFrame): 验证集数据,包含'close'列
        df_test (pd.DataFrame): 测试集数据,包含'close'列
    """
    chunk_size = 4320  # 每个时间窗口的大小(单位:数据点)
    
    # 存储各数据集的斜率值
    slopes_train, slopes_val, slopes_test = [], [], []
    
    # 分块计算训练集斜率
    for i in range(0, int(len(df_train) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_train['close'][start:end].values
        smoothed_chunk = smooth_data(chunk)  # 平滑处理
        slope = get_slope(smoothed_chunk)    # 计算斜率
        slopes_train.append(slope)

    # 分块计算验证集斜率(逻辑同上)
    for i in range(0, int(len(df_val) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_val['close'][start:end].values
        smoothed_chunk = smooth_data(chunk)
        slope = get_slope(smoothed_chunk)
        slopes_val.append(slope)

    # 分块计算测试集斜率(逻辑同上)
    for i in range(0, int(len(df_test) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_test['close'][start:end].values
        smoothed_chunk = smooth_data(chunk)
        slope = get_slope(smoothed_chunk)
        slopes_test.append(slope)

    # 使用分位数将斜率分为5个类别(0-4)
    quantiles = [0, 0.05, 0.35, 0.65, 0.95, 1]
    slope_labels_train, bins = pd.qcut(slopes_train, q=quantiles, retbins=True, labels=False)

    # 初始化索引存储结构
    train_indices = [[] for _ in range(5)]
    val_indices = [[] for _ in range(5)]
    test_indices = [[] for _ in range(5)]
    
    # 保存训练集标签
    for index, label in enumerate(slope_labels_train):
        train_indices[label].append(index)
    with open('./data/AL/train/slope_labels.pkl', 'wb') as file:
        pickle.dump(train_indices, file)

    # 调整边界值防止溢出
    bins[0] = -100
    bins[-1] = 100
    
    # 处理验证集标签并调整极端类别
    slope_labels_val = pd.cut(slopes_val, bins=bins, labels=False, include_lowest=True)
    slope_labels_val = [1 if element == 0 else element for element in slope_labels_val]
    slope_labels_val = [3 if element == 4 else element for element in slope_labels_val]
    
    # 处理测试集标签并调整极端类别(逻辑同上)
    slope_labels_test = pd.cut(slopes_test, bins=bins, labels=False, include_lowest=True)
    slope_labels_test = [1 if element == 0 else element for element in slope_labels_test]
    slope_labels_test = [3 if element == 4 else element for element in slope_labels_test]

    # 保存验证集和测试集标签
    for index, label in enumerate(slope_labels_val):
        val_indices[label].append(index)
    with open('./data/AL/val/slope_labels.pkl', 'wb') as file:
        pickle.dump(val_indices, file)
    for index, label in enumerate(slope_labels_test):
        test_indices[label].append(index)
    with open('./data/AL/test/slope_labels.pkl', 'wb') as file:
        pickle.dump(test_indices, file)

def label_volatility(df_train, df_val, df_test):
    """
    为训练集、验证集和测试集的时间序列数据添加波动率标签
    
    参数:
        df_train (pd.DataFrame): 训练集数据
        df_val (pd.DataFrame): 验证集数据
        df_test (pd.DataFrame): 测试集数据
    """
    chunk_size = 4320  # 时间窗口大小
    
    # 存储波动率值
    volatilities_train, volatilities_val, volatilities_test = [], [], []
    
    # 分块计算波动率(与label_slope结构类似)
    for i in range(0, int(len(df_train)/chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_train[start:end]
        chunk['return'] = chunk['close'].pct_change().fillna(0)  # 计算收益率
        volatility = chunk['return'].std()  # 计算标准差作为波动率
        volatilities_train.append(volatility)

    # 验证集处理(逻辑同上)
    for i in range(0, int(len(df_val)/chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_val[start:end]
        chunk['return'] = chunk['close'].pct_change().fillna(0)
        volatility = chunk['return'].std()
        volatilities_val.append(volatility)
    
    # 测试集处理(逻辑同上)
    for i in range(0, int(len(df_test)/chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_test[start:end]
        chunk['return'] = chunk['close'].pct_change().fillna(0)
        volatility = chunk['return'].std()
        volatilities_test.append(volatility)

    # 波动率分位数分类
    quantiles = [0, 0.05, 0.35, 0.65, 0.95, 1]
    vol_labels_train, bins = pd.qcut(volatilities_train, q=quantiles, retbins=True, labels=False)

    # 初始化索引存储结构
    train_indices = [[] for _ in range(5)]
    val_indices = [[] for _ in range(5)]
    test_indices = [[] for _ in range(5)]
    
    # 保存训练集标签
    for index, label in enumerate(vol_labels_train):
        train_indices[label].append(index)
    with open('./data/AL/train/vol_labels.pkl', 'wb') as file:
        pickle.dump(train_indices, file)

    # 调整边界值并处理极端类别
    bins[0] = 0
    bins[-1] = 1
    
    # 处理验证集和测试集标签
    vol_labels_val = pd.cut(volatilities_val, bins=bins, labels=False, include_lowest=True)
    vol_labels_val = [1 if element == 0 else element for element in vol_labels_val]
    vol_labels_val = [3 if element == 4 else element for element in vol_labels_val]
    
    vol_labels_test = pd.cut(volatilities_test, bins=bins, labels=False, include_lowest=True)
    vol_labels_test = [1 if element == 0 else element for element in vol_labels_test]
    vol_labels_test = [3 if element == 4 else element for element in vol_labels_test]

    # 保存验证集和测试集标签
    for index, label in enumerate(vol_labels_val):
        val_indices[label].append(index)
    with open('./data/AL/val/vol_labels.pkl', 'wb') as file:
        pickle.dump(val_indices, file)
    for index, label in enumerate(vol_labels_test):
        test_indices[label].append(index)
    with open('./data/AL/test/vol_labels.pkl', 'wb') as file:
        pickle.dump(test_indices, file)


def label_whole(df, dataset_name='dataset'):
    """
    对整个数据集添加滚动窗口特征
    为每个数据点生成基于滚动窗口的历史特征,增强模型对时序模式的感知能力。
    参数:
        df (pd.DataFrame): 输入数据集,包含'close'列
        dataset_name (str): 数据集名称,用于日志输出
    返回:
        pd.DataFrame: 添加了slope和vol特征的新数据集
    """
    logger.info(f"开始处理 {dataset_name}, shape: {df.shape}")

    window_size_list = [180,360]  # 窗口尺寸列表

    # 先计算一次return,避免重复计算
    df['return'] = df['close'].pct_change().fillna(0)

    for i in range(len(window_size_list)):
        window_size = window_size_list[i]
        logger.info(f"{dataset_name}: 计算窗口大小 {window_size} 的特征...")

        # 添加滚动窗口斜率特征
        df['slope_{}'.format(window_size)] = df['close'].rolling(window=window_size).apply(get_slope_window)

        # 添加滚动窗口波动率特征
        df['vol_{}'.format(window_size)] = df['return'].rolling(window=window_size).std()

    logger.info(f"{dataset_name} 处理完成")
    return df


def process_single_dataset(args):
    """
    处理单个数据集的包装函数,用于多进程并行
    参数:
        args: (df, dataset_name) 元组
    返回:
        处理后的DataFrame
    """
    df, dataset_name = args
    result = label_whole(df, dataset_name)
    result = result.dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    return dataset_name, result

if __name__ == "__main__":

    # 加载数据 -> 创建目录 -> 分块存储 -> 生成标签 -> 添加特征 -> 保存结果
    df_train = pd.read_feather('./data/AL/df_train.feather')
    df_val = pd.read_feather('./data/AL/df_val.feather')
    df_test = pd.read_feather('./data/AL/df_test.feather')
    logger.info(f"data shape: {df_train.shape}")
    logger.info(f"columns: {df_train.columns.tolist()}")
    logger.info(f"head:\n{df_train.head()}")
    logger.info(f"tail:\n{df_train.tail()}")
    os.makedirs('./data//train', exist_ok=True)
    os.makedirs('./data/AL/val', exist_ok=True)
    os.makedirs('./data/AL/test', exist_ok=True)
    os.makedirs('./data/AL/whole', exist_ok=True)

    # chunk(df_train, df_val, df_test)
    # label_slope(df_train, df_val, df_test)
    # label_volatility(df_train, df_val, df_test)

    # 使用多进程并行处理三个数据集
    logger.info("=" * 50)
    logger.info("开始并行处理数据集特征提取...")
    logger.info(f"可用CPU核心数: {cpu_count()}")
    logger.info("=" * 50)

    # 准备数据集列表
    datasets = [
        (df_train.copy(), 'df_train'),
        (df_val.copy(), 'df_val'),
        (df_test.copy(), 'df_test')
    ]

    # 使用3个进程并行处理(每个数据集一个进程)
    num_processes = min(3, cpu_count())
    logger.info(f"使用 {num_processes} 个进程并行处理")

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_dataset, datasets)

    # 收集结果
    result_dict = {name: df for name, df in results}
    df_train = result_dict['df_train']
    df_val = result_dict['df_val']
    df_test = result_dict['df_test']

    logger.info("=" * 50)
    logger.info("所有数据集处理完成,开始保存文件...")
    logger.info("=" * 50)

    df_train.to_feather('./data/AL/whole/train.feather')
    logger.info("df_train 已保存")

    df_val.to_feather('./data/AL/whole/val.feather')
    logger.info("df_val 已保存")

    df_test.to_feather('./data/AL/whole/test.feather')
    logger.info("df_test 已保存")

    logger.info("=" * 50)
    logger.info("全部流程完成!")
    logger.info("=" * 50)


