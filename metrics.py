import torch
from torchmetrics import ROC, AUROC, F1Score
import os
from torchvision.transforms import transforms
from skimage import measure
import pandas as pd
from statistics import mean
import numpy as np
from sklearn.metrics import auc
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve,confusion_matrix
from visualize import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Metric:
    def __init__(self,labels_list, predictions, anomaly_map_list, gt_list, config) -> None:
        self.labels_list = labels_list
        self.predictions = predictions
        self.anomaly_map_list = anomaly_map_list
        self.gt_list = gt_list
        self.config = config
        self.threshold = 0.5
        self.precision_list=[]
        self.recall_list=[]
    
    def image_auroc(self):
        auroc_image = roc_auc_score(self.labels_list, self.predictions)
        return auroc_image
    
    def pixel_auroc(self):
        resutls_embeddings = self.anomaly_map_list[0]
        for feature in self.anomaly_map_list[1:]:
            resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
        resutls_embeddings =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 

        gt_embeddings = self.gt_list[0]
        for feature in self.gt_list[1:]:
            gt_embeddings = torch.cat((gt_embeddings, feature), 0)

        resutls_embeddings = resutls_embeddings.clone().detach().requires_grad_(False)
        gt_embeddings = gt_embeddings.clone().detach().requires_grad_(False)

        auroc_p = AUROC(task="binary")
        
        gt_embeddings = torch.flatten(gt_embeddings).type(torch.bool).cpu().detach()
        resutls_embeddings = torch.flatten(resutls_embeddings).cpu().detach()
        auroc_pixel = auroc_p(resutls_embeddings, gt_embeddings)
        return auroc_pixel
    
    # def optimal_threshold(self):
    #     fpr, tpr, thresholds = roc_curve(self.labels_list, self.predictions)

    #     # Calculate Youden's J statistic for each threshold
    #     youden_j = tpr - fpr

    #     # Find the optimal threshold that maximizes Youden's J statistic
    #     optimal_threshold_index = np.argmax(youden_j)
    #     optimal_threshold = thresholds[optimal_threshold_index]
    #     self.threshold = optimal_threshold
    #     return optimal_threshold
    def recall_precise(self):
        tpr, fpr, thresholds = roc_curve(self.labels_list, self.predictions)
        thresholds_list=[np.quantile(self.predictions,0.5),np.quantile(self.predictions,0.75),np.quantile(self.predictions,0.8),np.quantile(self.predictions,0.9),np.quantile(self.predictions,0.95),np.quantile(self.predictions,0.75)+1.5*(np.quantile(self.predictions,0.75)-np.quantile(self.predictions,0.25)),np.quantile(self.predictions,0.5)+np.std(self.predictions),np.quantile(self.predictions,0.5)+2*np.std(self.predictions),np.quantile(self.predictions,0.5)+3*np.std(self.predictions),np.quantile(self.predictions,0.5)+4*np.std(self.predictions),np.quantile(self.predictions,0.5)+5*np.std(self.predictions),np.quantile(self.predictions,0.5)+6*np.std(self.predictions)]
        precison=[]
        recall=[]
        for threshold in thresholds_list:
        # 二元分類的預測
            binary_predictions = [1 if pred > threshold else 0 for pred in self.predictions]

            # 計算混淆矩陣
            tp, fn, fp, tn = confusion_matrix(self.labels_list, binary_predictions).ravel()

            # 計算 recall 和 precison
            precison = tp / (fp + tp) if (fp + tp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precison.append(precison)
            recall.append(recall)

        print("Thresholds List:", thresholds_list)
        print("precison:", precison)
        print("recall:", recall)

        # 計算 TNR（True Negative Rate）
        # tnr = 1 - fpr#min fpr

        # 找到 TNR 最大的索引（可能有多個）
        max_precison = np.max(precison)
        max_precison_indices = np.where(np.array(precison) == max_precison)[0]
        print("Indices of Min FPR:", max_precison_indices)

        # 在 TNR 最大的閾值中，選擇對應 TPR 最大的閾值
        if len(max_precison_indices) >= 1:
            # tpr=np.argmax(tpr[min_fpr_indices])
            # print(tpr)
            # max_tpr = np.max(tpr)
            # print(max_tpr)
            # max_tpr_at_max_tnr_index = np.where(tpr == max_tpr)[0][0]
            max_precison_index = max_precison_indices[np.argmax(np.array(tpr)[max_precison_indices])]
        else:
            max_precisonr_index = max_precison_indices[0]

        # 找到最終最佳閾值
        print(max_precison_index)
        optimal_threshold = thresholds_list[max_precison_index]
        # optimal_threshold=0.006
        self.threshold = optimal_threshold
        return optimal_threshold,precison,recall

    def optimal_threshold(self,v_value): 
    # 計算 FPR（False Positive Rate）、TPR（True Positive Rate）和對應的 thresholds
        # fpr, tpr, thresholds = roc_curve(self.labels_list, self.predictions)
        print(self.labels_list)
        print(self.predictions)
        tpr, fpr, thresholds = roc_curve(self.labels_list, self.predictions)
        thresholds_list=[np.quantile(self.predictions,0.5),np.quantile(self.predictions,0.75),np.quantile(self.predictions,0.75)+1.5*(np.quantile(self.predictions,0.75)-np.quantile(self.predictions,0.25)),np.quantile(self.predictions,0.5)+np.std(self.predictions),np.quantile(self.predictions,0.5)+2*np.std(self.predictions),np.quantile(self.predictions,0.5)+3*np.std(self.predictions),np.quantile(self.predictions,0.5)+4*np.std(self.predictions),np.quantile(self.predictions,0.5)+5*np.std(self.predictions),np.quantile(self.predictions,0.5)+6*np.std(self.predictions)]
        # thresholds_list=[0.4]
        fpr=[]
        tpr=[]
        for threshold in thresholds_list:
        # 二元分類的預測
            binary_predictions = [1 if pred > threshold else 0 for pred in self.predictions]

            # 計算混淆矩陣
            tp, fn, fp, tn = confusion_matrix(self.labels_list, binary_predictions).ravel()

            # 計算 FPR 和 TPR
            fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr_value = tp / (tp + fn) if (tp + fn) > 0 else 0

            fpr.append(fpr_value)
            tpr.append(tpr_value)

        print("Thresholds List:", thresholds_list)
        print("FPR:", fpr)
        print("TPR:", tpr)

        # 計算 TNR（True Negative Rate）
        # tnr = 1 - fpr#min fpr

        # 找到 TNR 最大的索引（可能有多個）
        min_fpr = np.min(fpr)
        min_fpr_indices = np.where(np.array(fpr) == min_fpr)[0]
        print("Indices of Min FPR:", min_fpr_indices)

        # 在 TNR 最大的閾值中，選擇對應 TPR 最大的閾值
        if len(min_fpr_indices) >= 1:
            # tpr=np.argmax(tpr[min_fpr_indices])
            # print(tpr)
            # max_tpr = np.max(tpr)
            # print(max_tpr)
            # max_tpr_at_max_tnr_index = np.where(tpr == max_tpr)[0][0]
            max_tpr_at_max_tnr_index = min_fpr_indices[np.argmax(np.array(tpr)[min_fpr_indices])]
        else:
            max_tpr_at_max_tnr_index = min_fpr_indices[0]

        # 找到最終最佳閾值
        print(max_tpr_at_max_tnr_index)
        optimal_threshold = thresholds_list[max_tpr_at_max_tnr_index]
        self.threshold = optimal_threshold

         # ======= 🔥 繪製直方圖 =======
        plt.figure(figsize=(8, 6))

        # 根據 labels_list 分類 predictions
        predictions_0 = [pred for pred, label in zip(self.predictions, self.labels_list) if label == 0]
        predictions_1 = [pred for pred, label in zip(self.predictions, self.labels_list) if label == 1]

        # 繪製長條圖
        # 計算共同的 bins 範圍
        all_predictions = np.concatenate([predictions_0, predictions_1])  # 合併所有預測值
        bins = np.linspace(all_predictions.min(), all_predictions.max(), 100)  # 設定統一 bins
        plt.hist(predictions_0, bins=bins, alpha=0.7, color="blue", label="Label 0")  # 藍色: Label 0
        plt.hist(predictions_1, bins=bins, alpha=0.7, color="orange", label="Label 1")  # 橘色: Label 1

        # 加入標題與標籤
        plt.axvline(optimal_threshold, color="red", linestyle="dashed", linewidth=2, label=f"Threshold: {optimal_threshold:.4f}")
        plt.xlabel("Prediction Values")
        plt.ylabel("Count")
        plt.title(f"Histogram of Predictions score v={v_value}")
        plt.legend()

        # 顯示圖表
        plt.show()
        return optimal_threshold


    def pixel_pro(self):
        #https://github.com/hq-deng/RD4AD/blob/main/test.py#L337
        def _compute_pro(masks, amaps, num_th = 200):
            resutls_embeddings = amaps[0]
            for feature in amaps[1:]:
                resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
            amaps =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 
            amaps = amaps.squeeze(1)
            amaps = amaps.cpu().detach().numpy()
            gt_embeddings = masks[0]
            for feature in masks[1:]:
                gt_embeddings = torch.cat((gt_embeddings, feature), 0)
            masks = gt_embeddings.squeeze(1).cpu().detach().numpy()
            min_th = amaps.min()
            max_th = amaps.max()
            delta = (max_th - min_th) / num_th
            binary_amaps = np.zeros_like(amaps)
            df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])

            for th in np.arange(min_th, max_th, delta):
                binary_amaps[amaps <= th] = 0
                binary_amaps[amaps > th] = 1

                pros = []
                for binary_amap, mask in zip(binary_amaps, masks):
                    for region in measure.regionprops(measure.label(mask)):
                        axes0_ids = region.coords[:, 0]
                        axes1_ids = region.coords[:, 1]
                        tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                        pros.append(tp_pixels / region.area)

                inverse_masks = 1 - masks
                fp_pixels = np.logical_and(inverse_masks , binary_amaps).sum()
                fpr = fp_pixels / inverse_masks.sum()
                # print(f"Threshold: {th}, FPR: {fpr}, PRO: {mean(pros)}")
                #加的
                if not pros:  # 如果 pros 是空列表
                    pros = [0]  # 或者選擇一個合理的預設值
                #加的
                df = pd.concat([df, pd.DataFrame({"pro": mean(pros), "fpr": fpr, "threshold": th}, index=[0])], ignore_index=True)
                # df = df.concat({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

            # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
            df = df[df["fpr"] < 0.3]
            df["fpr"] = df["fpr"] / df["fpr"].max()

            pro_auc = auc(df["fpr"], df["pro"])
            return pro_auc
        
        pro = _compute_pro(self.gt_list, self.anomaly_map_list, num_th = 200)
        return pro
    
    def misclassified(self,input):
        predictions = torch.tensor(self.predictions)
        labels_list = torch.tensor(self.labels_list)
        # Pass_predictions = predictions(labels_list==0)
        # NG_predictions = predictions(labels_list==1)

        predictions0_1 = (predictions > self.threshold).int()

        misclassified_samples = []  # 用來存錯誤分類的樣本索引
        # print(len(input))
        for i, (l, p) in enumerate(zip(labels_list, predictions0_1)):
            if l != p:
                print(f'Sample {i}: predicted as {p.item()}, label is {l.item()}')
                misclassified_samples.append(i)

        # 🔥 顯示分類錯誤的圖片
        num_samples = len(misclassified_samples)
        if num_samples > 0:
            plt.figure(figsize=(10, 5))
            cols = min(num_samples, 5)  # 每行最多顯示 5 張
            rows = (num_samples // cols) + 1

            for idx, sample_idx in enumerate(misclassified_samples[:cols * rows]):
                plt.subplot(rows, cols, idx + 1)
                image = input[sample_idx].squeeze()  # 假設圖片為 2D 灰階
                plt.imshow(show_tensor_image(image))
                plt.title(f"Pred: {predictions0_1[sample_idx].item()} | Label: {labels_list[sample_idx].item()}")
                plt.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            print("No misclassified samples found.")
       

    # def misclassified(self):
    #     predictions = torch.tensor(self.predictions)
    #     labels_list = torch.tensor(self.labels_list)
    #     predictions0_1 = (predictions > self.threshold).int()
    #     for i,(l,p) in enumerate(zip(labels_list, predictions0_1)):
    #         print('Sample : ', i, ' predicted as: ',p.item() ,' label is: ',l.item(),'\n' ) if l != p else None

