from asyncio import constants
from typing import Any
import torch
from unet import *
from dataset import *
from visualize import *
from anomaly_map import *
from metrics import *
from feature_extractor import *
from reconstruction import *
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

class DDAD:
    def __init__(self, unet, config) -> None:
        self.test_dataset = Dataset_maker(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=False,
        )
        
        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size= config.data.test_batch_size,
            shuffle=False,
            num_workers= config.model.num_workers,
            drop_last=False,
        )
        self.unet = unet
        self.config = config
        self.reconstruction = Reconstruction(self.unet, self.config)
        self.transform = transforms.Compose([
                            transforms.CenterCrop((224)), 
                        ])
    ##加的
    def display_confusion_matrix_at_thresholds(self, labels_list, predictions, thresholds):
        """
        計算並顯示不同閾值下的混淆矩陣。
        :param labels_list: 真實標籤 (0 或 1) 的列表
        :param predictions: 模型預測分數的列表
        :param thresholds: 閾值列表
        """
        for threshold in thresholds:
            binary_predictions = [1 if pred > threshold else 0 for pred in predictions]
            
            # 獲取唯一的標籤類別
            unique_classes = unique_labels(labels_list, binary_predictions)
            
            # 確保輸出混淆矩陣為完整格式
            cm = confusion_matrix(labels_list, binary_predictions, labels=unique_classes)
            
            if cm.shape != (2, 2):
                # 自動補全混淆矩陣以確保符合 TN, FP, FN, TP 格式
                complete_cm = [[0, 0], [0, 0]]
                for i, row_class in enumerate(unique_classes):
                    for j, col_class in enumerate(unique_classes):
                        complete_cm[row_class][col_class] = cm[i][j]
                cm = complete_cm

            tp, fn, fp, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            print(f"\nThreshold: {threshold:.2f}")
            print(f"Confusion Matrix:\n[[TN: {tn} FP: {fp}]\n [FN: {fn} TP: {tp}]]")
            fig, ax = plt.subplots(figsize=(6, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive","Negative"])
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            ax.set_title(f"Confusion Matrix at Threshold {threshold:.4f}")
            plt.show()
            # print(f"Optimal threshold: {self.threshold}, Min FPR: {min_fpr}, TPR at Max TNR: {tpr[max_tpr_at_max_tnr_index]}")
    ##加的

    def __call__(self) -> Any:
        feature_extractor = domain_adaptation(self.unet, self.config, fine_tune=False)
        feature_extractor.eval()
###############################################
        v_list=[0,0.4,0.8,1.2,1.6,2,5,10]#
        # v_list=[3,5,7,10,100]
        # v_list=[1.2]
        for v_value in v_list:#
            labels_list = []
            predictions= []
            anomaly_map_list = []
            gt_list = []
            reconstructed_list = []
            forward_list = []
            with torch.no_grad():
                for input, gt, labels in self.testloader:
                    # input = input.unsqueeze(0).to(self.config.model.device)  # 增加 batch 維度
                    input = input.to(self.config.model.device)
                    x0 = self.reconstruction(input, input, self.config.model.w)[-1]
                    anomaly_map = heat_map(x0, input, feature_extractor, self.config,v_value)

                    anomaly_map = self.transform(anomaly_map)
                    gt = self.transform(gt)

                    forward_list.append(input)
                    anomaly_map_list.append(anomaly_map)


                    gt_list.append(gt)
                    reconstructed_list.append(x0)
                    for pred, label in zip(anomaly_map, labels):
                        labels_list.append(0 if label == 'good' else 1)
                        predictions.append(torch.max(pred).item())

                print(predictions)        
                metric = Metric(labels_list, predictions, anomaly_map_list, gt_list, self.config)
                metric.optimal_threshold(v_value)
                if self.config.metrics.auroc:
                    print('AUROC: ({:.1f},{:.1f})'.format(metric.image_auroc() * 100, metric.pixel_auroc() * 100))
                if self.config.metrics.pro:
                    print('PRO: {:.1f}'.format(metric.pixel_pro() * 100))
                if self.config.metrics.misclassifications:
                    forward_list = torch.cat(forward_list, dim=0)
                    metric.misclassified(forward_list)
            
    # #################
        reconstructed_list = torch.cat(reconstructed_list, dim=0)
        
        anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
        pred_mask = (anomaly_map_list > metric.threshold).float()
        gt_list = torch.cat(gt_list, dim=0)
        if not os.path.exists('results'):
                os.mkdir('results')
        if self.config.metrics.visualisation:
            visualize(forward_list, reconstructed_list, gt_list, pred_mask, anomaly_map_list, self.config.data.category)
        # 顯示特定閥值的混淆矩陣

        thresholds = [metric.threshold]
        self.display_confusion_matrix_at_thresholds(labels_list, predictions, thresholds)