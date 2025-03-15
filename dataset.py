import os
from glob import glob
from pathlib import Path
import shutil
import numpy as np
# import csv
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2


def scale_transform(t):
    return (t * 2) - 1

class Dataset_maker(torch.utils.data.Dataset): 
    def __init__(self, root, category, config, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(),
                transforms.Lambda(scale_transform),  # Scale between [-1, 1]
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(),  # Scales data into [0, 1]
            ]
        )
        self.config = config
        self.is_train = is_train
        self.is_validation =config.metrics.is_validation
    
        
        # 驗證資料
        train_path = os.path.join(root, "train", category, "input", "PASS", "*", "*.jpg")
        print(f"Searching for training images in: {train_path}")
        all_images = glob(train_path)

        # 如果 `is_train=True`，拆分 80% 訓練 / 20% 驗證
        train_files, val_files = train_test_split(all_images, test_size=0.2, random_state=config.model.seed)

        if is_train:
            self.image_files = train_files
            print(f"Training set size: {len(self.image_files)}")
       
        else:
            if self.is_validation:
                # self.image_files = val_files
                # print(f"Validation set size: {len(self.image_files)}")
                val_path = os.path.join(root, "train", category, "input", "NG", "*", "*.jpg")
                print(f"Searching for training images in: {val_path}")
                self.image_files = val_files+glob(val_path)
                print(f"Validation set size: {len(self.image_files)}")

            else:#測試資料
                pass_path = os.path.join(root, "val", category, "input", "PASS", "*", "*.jpg")
                ng_path = os.path.join(root, "val", category, "input", "NG", "*", "*.jpg")
                print(f"Searching for validation images in: {pass_path} and {ng_path}")
                self.image_files = glob(pass_path) + glob(ng_path)
        
        # 檢查是否有檔案
        if len(self.image_files) == 0:
            raise ValueError(
                f"No images found in the specified path. "
                f"Check your dataset structure and ensure the files are located at: "
                f"{train_path if is_train else pass_path + ' or ' + ng_path}"
            )

    def __getitem__(self, index):

        image_file = self.image_files[index]
        image = Image.open(image_file).convert("RGB")  # 確保影像是 RGB 格式
        # image = self.image_transform(image)
        # 切圖片
        # cropped_image = self.detect_and_crop(image, show=True)

        image = self.preprocess_image(image)  # 應用前處理
        image_tensor = self.image_transform(image)
        # print("cropped_image",np.shape(cropped_image))

        # 轉換成 Tensor
        # image_tensor = self.image_transform(cropped_image)
        # print("image_tensor",np.shape(image_tensor))
        # if True:  # 顯示裁剪後的圖片
        #         plt.figure(figsize=(4, 4))
        #         plt.imshow(image_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) 
        #         # plt.imshow(image)
        #         plt.axis("off")
        #         plt.show()


        if self.is_train:
            label = 'good'  # 訓練資料的標籤固定為 'good'
            return image_tensor, label
        else:
            print(os.path.dirname(image_file))
            if (os.path.dirname(image_file).endswith("PASS\\SolderLight") or os.path.dirname(image_file).endswith("PASS\\UniformLight") or os.path.dirname(image_file).endswith("PASS\\LowAngleLight") or os.path.dirname(image_file).endswith("PASS\\WhiteLight")):
                #更改多重條件
                # target = torch.zeros([1, image.shape[-2], image.shape[-1]])  # 沒有異常遮罩
                target = torch.zeros([1, image_tensor.shape[-2], image_tensor.shape[-1]])  # 沒有異常遮罩
                label = 'good'
                print("0")
            else:  # NG 類別
                # target = torch.zeros([1, image.shape[-2], image.shape[-1]])  # 預設遮罩（可自訂）
                target = torch.zeros([1, image_tensor.shape[-2], image_tensor.shape[-1]])  # 預設遮罩（可自訂）
                label = 'defective'
                print("1")
            return image_tensor, target, label

    def __len__(self):
        return len(self.image_files)
    

    def preprocess_image(self, image):
        """ 前處理：裁剪物件、旋轉對齊、統一尺寸 """
        if  self.config.data.category=="(FIDUCIALMARK)-S":
            image = np.array(image)  # 轉換為 NumPy 陣列
            image = self.resize_object(image)  # 裁剪並調整尺寸
            image = Image.fromarray(image)  # 轉回 PIL 格式
            # image = self.image_transform(image)  # 應用標準轉換
            if image.shape[0] == 1:
                image = image.expand(3, image.shape[1], image.shape[2])
        elif  self.config.data.category=="9CM410437X11" or self.config.data.category=="9CM4001U-Z11":
            image = np.array(image)  # 轉換為 NumPy 陣列
            image = self.rotate_if_needed(image)
            image = Image.fromarray(image)  # 轉回 PIL 格式
            # image = self.image_transform(image)  # 應用標準轉換

            # if self.image_transform(image).shape[0] == 1:
            #     image = image.expand(3, image.shape[1], image.shape[2])
        else:
            return image
        return image

    def resize_object(self, image, target_size=(128, 128)):
        """ 使用 Bounding Box 裁剪物件，並 Resize 到固定大小，將所有步驟顯示於同一張圖 """
        
        # fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 建立 2x3 的子圖 (共 6 張圖)
        # axes = axes.flatten()  # 攤平成一維陣列，方便索引
        
        # 1️⃣ **顯示原始圖片**
        # axes[0].imshow(image)
        # axes[0].set_title("original")
        # axes[0].axis("off")

        # 2️⃣ **轉換成灰階**
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # axes[1].imshow(gray, cmap="gray")
        # axes[1].set_title("gray")
        # axes[1].axis("off")

        # 3️⃣ **二值化處理**
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_TOZERO_INV) # 如果大於 127 等於 0，反之數值不變。
        _, thresh = cv2.threshold(thresh, 70, 255, cv2.THRESH_BINARY)# 如果大於 60 等於 0，反之0。
        # axes[2].imshow(thresh, cmap="gray")
        # axes[2].set_title("BINARY")
        # axes[2].axis("off")

        # 4️⃣ **尋找輪廓**
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5️⃣ **畫出輪廓 (如果有)**
        if contours:
            contour_image = image.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

            # axes[3].imshow(contour_image)
            # axes[3].set_title("Contours")
            # axes[3].axis("off")

            # 6️⃣ **計算最大輪廓的 Bounding Box 並裁剪**
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cropped = image[y:y+h, x:x+w]

            # axes[4].imshow(cropped)
            # axes[4].set_title("cropped")
            # axes[4].axis("off")

            # 7️⃣ **縮放到目標大小**
            resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
            # axes[5].imshow(resized)
            # axes[5].set_title("Resize")
            # axes[5].axis("off")
            # plt.tight_layout()
            # plt.show()
            
            return resized

        # **如果沒有輪廓，直接 Resize 整張圖片**
        resized = cv2.resize(image, target_size)

        # axes[3].imshow(resized)
        # axes[3].set_title(f"無輪廓，直接 Resize 到 {target_size}")
        # axes[3].axis("off")

        # plt.tight_layout()
        # plt.show()
        
        return resized


    def rotate_if_needed(self,image):
        """ 
        偵測圖片大小，若長邊大於短邊則旋轉 90 度，否則保持原樣。
        """
        height, width = image.shape[:2]  # 取得圖片的高與寬
        
        # 建立 Matplotlib 圖片顯示
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 建立 1x2 圖片排列
        
        # 顯示原始圖片
        # axes[0].imshow(image)
        # axes[0].set_title(f"before ({width}x{height})")
        # axes[0].axis("off")

        # 若長邊大於短邊，則旋轉 90 度
        if width < height:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            # axes[1].imshow(rotated_image)
            # axes[1].set_title(f"after ({rotated_image.shape[1]}x{rotated_image.shape[0]})")
        else:
            rotated_image = image
            # axes[1].imshow(rotated_image)
            # axes[1].set_title("after")

        # axes[1].axis("off")
        # plt.tight_layout()
        # plt.show()

        return rotated_image  # 回傳最終圖片
