import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
__all__ = ['MMDataLoader']
logger = logging.getLogger('EMOE')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        # 根据数据集名称调用对应的初始化方法
        DATASET_MAP = {
            'biovid':self.__init_biovid,
        }
        DATASET_MAP[args['dataset_name']]()
    def __init_biovid(self): # ECG+vision+GSR
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.gsr = data[self.mode]['gsr'].astype(np.float32)
        self.ecg = data[self.mode]['ecg'].astype(np.float32)
        self.ids = data[self.mode]['id']

        # 指定额外ECG特征文件
        if self.args['feature_ECG'] != "":
            with open(self.args['feature_ECG'], 'rb') as f:
                data_ECG = pickle.load(f)
            self.ecg = data_ECG[self.mode]['ecg'].astype(np.float32)
            self.args['feature_dims'][0] = self.ecg.shape[2]
        # 指定额外GSR特征文件
        if self.args['feature_GSR'] != "":
            with open(self.args['feature_GSR'], 'rb') as f:
                data_GSR = pickle.load(f)
            self.gsr = data_GSR[self.mode]['gsr'].astype(np.float32)
            self.args['feature_dims'][1] = self.gsr.shape[2]
        # 指定额外视觉特征文件
        if self.args['feature_V'] != "":
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2]

        # 标签处理
        if self.args['train_mode'] == "classification":
            self.labels = {
                'M': np.array(data[self.mode]['classification_labels']).astype(np.int64)
            }
        else:
            self.labels = {
                'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
            }

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        # 非对齐数据记录实际seq_len
        if not self.args['need_data_aligned']:
            if self.args['feature_ECG'] != "":
                self.ecg_lengths = list(data_ECG[self.mode]['ecg_lengths'])
            else:
                self.ecg_lengths = data[self.mode]['ecg_lengths']
            if self.args['feature_GSR'] != "":
                self.gsr_lengths = list(data_GSR[self.mode]['gsr_lengths'])
            else:
                self.gsr_lengths = data[self.mode]['gsr_lengths']
            if self.args['feature_V'] != "":
                self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
            else:
                self.vision_lengths = data[self.mode]['vision_lengths']
        # 处理无穷大值
        self.ecg[self.ecg == -np.inf] = 0
        self.gsr[self.gsr == -np.inf] = 0
        # 归一化
        if 'need_normalized' in self.args and self.args['need_normalized']:
            self.__normalize()

    # 截断方法
    def __truncate(self):
        # 将长序列截断为固定长度
        def do_truncate(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature

        ecg_length, gsr_length, video_length = self.args['seq_lens']
        self.vision = do_truncate(self.vision, video_length)
        self.ecg = do_truncate(self.ecg, ecg_length)
        self.gsr = do_truncate(self.gsr, gsr_length)

    # 归一化方法
    def __normalize(self):
        # 转置
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.ecg = np.transpose(self.ecg, (1, 0, 2))
        self.gsr = np.transpose(self.gsr, (1, 0, 2))
        # 求均值
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.ecg = np.mean(self.ecg, axis=0, keepdims=True)
        self.gsr = np.mean(self.gsr, axis=0, keepdims=True)

        self.vision[self.vision != self.vision] = 0
        self.ecg[self.ecg != self.ecg] = 0
        self.gsr[self.gsr != self.gsr] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.ecg = np.transpose(self.ecg, (1, 0, 2))
        self.gsr = np.transpose(self.gsr, (1, 0, 2))

    # 获取样本数量
    def __len__(self):
        return len(self.labels['M'])

    # 获取序列长度
    def get_seq_len(self):
        return (self.ecg.shape[1], self.gsr.shape[1], self.vision.shape[1])

    # 获取特征维度
    def get_feature_dim(self):
        return self.ecg.shape[1], self.gsr.shape[1], self.vision.shape[1]
    # 获取单个样本
    def __getitem__(self, index):
        sample = {
            'ecg': self.ecg[index],
            'gsr': self.gsr[index],
            'vision': self.vision[index],
            'index': index,
            'id': self.ids[index],
            # 分类任务label整数；回归任务label浮点
            'labels': {k: torch.Tensor(v[index].reshape(-1), dtype=torch.long) for k, v in self.labels.items()}
                if self.args['train_mode'] == "classification" else {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
        }
        if not self.args['need_data_aligned']:
            sample['ecg_lengths'] = self.ecg_lengths[index]
            sample['gsr_lengths'] = self.gsr_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        return sample

def MMDataLoader(args, num_workers):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True) # 训练时打乱数据
        for ds in datasets.keys()
    }
    
    return dataLoader

if __name__ == '__main__':
    with open(r"D:\Code\python\Reshow\EMOE\dataset\MOSI\unaligned_50.pkl", 'rb') as f:
        data = pickle.load(f)
        print(data['train']['audio'].shape)
        print(data['train']['vision'].shape)
        print(type(data['train']['id'][0]))
        # for i in range(50):
        #     print(f"max:{max(data['train']['audio'][2][i])},min:{min(data['train']['audio'][2][i])}")
        # print("====vision====")
        # for i in range(50):
        #     print(f"max:{max(data['train']['vision'][2][i])},min:{min(data['train']['vision'][2][i])}")
