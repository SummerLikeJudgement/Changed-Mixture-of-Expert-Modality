import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self, train_mode):
        if train_mode == "regression":
            self.metrics_dict = {
                'BIOVID': self.__eval_biovid_regression,
            }
        else:
            self.metrics_dict = {
                'BIOVID': self.__eval_biovid_classification,
            }
    def __eval_biovid_classification(self, y_pred, y_true):
        """
        {
            "pain":0,1,2,3,4
        }
        """
        # 转化为numpy数组
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        y_pred_class = np.argmax(y_pred, axis=1)
        # 五分类评估
        Mult_acc_5 = accuracy_score(y_true, y_pred_class) # 5分类准确率
        F1_score_5 = f1_score(y_true, y_pred_class, average='weighted') # 加权F1分数
        # 二分类评估
        ## p0 vs p4
        p0p4_mask = (y_true == 0) | (y_true == 4)
        # 筛选概率并重新计算预测
        y_pred_p0p4 = y_pred[p0p4_mask]
        y_true_p0p4 = y_true[p0p4_mask]
        y_pred_p0p4_bi = np.array([[v[0], v[4]] for v in y_pred_p0p4])
        y_pred_p0p4_class = np.argmax(y_pred_p0p4_bi, axis=1)
        # 标签映射p0->0,p4->1
        y_true_p0p4_bi = np.where(y_true_p0p4 == 4, 1, 0)

        ## todo:有疼痛vs无疼痛

        acc_bi = accuracy_score(y_true_p0p4_bi, y_pred_p0p4_class)
        F1_bi = f1_score(y_true_p0p4_bi, y_pred_p0p4_class, average='weighted')

        eval_results = {
            # 5分类
            "Acc_5":round(Mult_acc_5, 4),
            "F1_5":round(F1_score_5, 4),
            # 2分类
            "Acc_2":round(acc_bi, 4),
            "F1_2":round(F1_bi, 4),
        }
        return eval_results

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_biovid_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        # 将预测值和真实值裁剪到指定范围
        test_preds_a5 = np.clip(test_preds, a_min=0., a_max=4.)
        test_truth_a5 = np.clip(test_truth, a_min=0., a_max=4.)
        test_preds_a3 = np.clip(test_preds, a_min=1., a_max=3.)
        test_truth_a3 = np.clip(test_truth, a_min=1., a_max=3.)
        # 计算平均绝对误差
        mae = np.mean(np.absolute(test_preds - test_truth)).astype(
            np.float64)  # Average L1 distance between preds and truths
        # 计算Pearson相关系数
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        # 计算不同裁剪范围的准确度
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        # 筛选非零值，转化为二元标签
        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)
        # 基于非零标签计算准确度和 F1 分数
        non_zeros_acc2 = accuracy_score(non_zeros_binary_truth, non_zeros_binary_preds)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        eval_results = {
            "Acc_2": round(non_zeros_acc2, 4),
            "F1_score": round(non_zeros_f1_score, 4),
            "Acc_5": round(mult_a5, 4),
            "MAE": round(mae, 4),
        }
        return eval_results

    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]