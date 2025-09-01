"""
read_Bio.py
将biovid PartA的filtered生理数据（ecg+gsr）读取并保存每个subject的指定生理信号与标签为npz文件
Preprocess bioVid data to processed; per processed file included all the data of one subject (patient)
"""
import argparse # 命令行参数解析
import csv
import os
import shutil
import glob

import numpy as np

# Column names
colNames = ['coltime', 'gsr', 'ecg', 'emg_trapezius', 'emg_corrugator', 'emg_zygomaticus']

# Label dicts
pain_labels = {
    'BL1': 0,
    'PA1': 1,
    'PA2': 2,
    'PA3': 3,
    'PA4': 4
}

# 命令行参数解析函数
def _parse_args():
    """
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='read_Bio.py')
    parser.add_argument('--data_dir', type=str, default=r'D:\EmoData\BioVid\PartA\biosignals_filtered') # 原始数据目录路径
    parser.add_argument('--output_dir', type=str, default=r'D:\EmoData\BioVid\processed\ECG') # 处理后的输出目录路径
    parser.add_argument('--select_signal', type=int, default = 2, help='1 as GSR(ECG) signal, see colNames above') # 选择要处理的生理信号
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
    return args


def process_bioVid(data_dir, output_dir, select_signal=1):
    """
    :param select_signal: physiological signal to be selected, 1 as GSR(ECG) signal, see colNames above
    :param output_dir: processed data directory
    :param data_dir: data directory
    :return: Save BioVid csv files to processed
    """
    # data_dir_files contain 87 subjects' pain signals,
    # Each subject has 5*20 (pain intensity level: 5; replicates: 20) samples
    data_dir_files = os.listdir(data_dir)
    i = 0
    # Dimension of files: 5*20
    for files in data_dir_files:
        parent_path = os.path.join(data_dir, files)
        path = glob.glob(parent_path + '/*.csv')

        labels_per = []
        signals_per = []
        # One single csv file per iteration included one pain intensity level and its signals
        for csv_file in path:
            # 提取疼痛级别标签
            key = csv_file.split('-')[-2]
            label = pain_labels[key]
            # 提取指定信号数据
            signal = []
            with open(csv_file, newline = '') as f:
                next(f) # 跳过标题行
                reader = csv.reader(f, delimiter = '\t')

                for row in reader:
                    values = [float(num) for num in row]
                    # Select gsr: 1;ecg: 2
                    signal.append(values[select_signal])
            signals_per.append(signal)
            labels_per.append(label)
        # 转换为numpy数组
        x = np.asarray(signals_per)
        y = np.asarray(labels_per)
        print('Data Processed: {}/{}'.format(i+1, len(data_dir_files)))
        print('X, y shape:{}, {}'.format(x.shape, y.shape))
        # 保存为npz文件
        filename = os.path.join(output_dir, data_dir_files[i] + '.processed')
        signals_dict = {
            "x": x,
            "y": y
        }
        np.savez(filename, **signals_dict)
        i += 1

    print("\n===================Done===================\n")


if __name__ == '__main__':
    args = _parse_args()
    process_bioVid(args.data_dir, args.output_dir, args.select_signal)