import torch
import os
import torch.nn as nn
from dataset import *
import matplotlib.pyplot as plt
from dataset import *
from loss import *


def trainer(model, category, config):
    '''
    Training the UNet model
    :param model: the UNet model
    :param category: the category of the dataset
    '''
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=config.model.learning_rate)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay
    )
    # 讓能訪問模型的參數(weight)
    train_dataset = Dataset_maker(
        root= config.data.data_dir,
        category=category,
        config = config,
        is_train=True,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=False,
    )
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(config.model.checkpoint_dir):
        os.mkdir(config.model.checkpoint_dir)

    loss_list=[]
    mean_loss=[]
    diff_loss=[]
    if config.model.still_train==True:
        epoch_times=config.model.epochs-config.model.still_train_load
    else:
        epoch_times=config.model.epochs

    for epoch in range(epoch_times):
        print("epoch=",epoch)
        inter_loss=[]
        for step, batch in enumerate(trainloader):
            optimizer.zero_grad()
            # optimizer.zero_grad() 負責清空梯度，避免前一次的梯度影響這次更新。
            t = torch.randint(0, config.model.trajectory_steps, (batch[0].shape[0],), device=config.model.device).long()
            #torch.randint(0, config.model.trajectory_steps, ...) 會 在 [0, trajectory_steps) 之間隨機選取數值（代表擴散過程中的某個時間步）。
            # batch[0].shape[0]：表示這個 batch 中的樣本數量，確保每個樣本都有對應的 t。
            # .long()：確保 t 是 整數型別 long（因為時間步 t 需要是整數）
            loss = get_loss(model, batch[0], t, config) 
            # model：擴散模型。
            # batch[0]：取出當前 batch 的 圖片數據（通常是 x_0）。
            # t：隨機選定的時間步（代表這張圖片的擴散階段）。
            # config：模型設定。
            loss.backward()
            # 根據 loss 計算每個參數的梯度（即 dL/dW，表示損失對權重的偏導數）。
            # backward() 會沿著 模型的計算圖（computation graph），對所有參數執行 反向傳播（Backpropagation）。
            optimizer.step()
            # 梯度下降（Gradient Descent）W=W−η⋅∇L，其中 η 是學習率（learning rate）。
            torch.cuda.empty_cache()#加的
            # if (epoch+1) % 25 == 0 and step == 0:
            # loss_list.append("loss.item()")
            print(f"Epoch {epoch+1} | Loss: {loss.item()}")
            if (epoch>=100):
                inter_loss.append(loss.item())
            if (epoch+1) %50 == 0 and epoch>0 and step ==0:
                if config.model.save_model:
                    if config.model.DDADS==True:
                        torch.save(model.state_dict(), os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.epochs),"smallUnet","save.pt"))
                    else:
                        torch.save(model.state_dict(), os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.epochs),"bigUnet","save.pt"))
                    # model_save_dir = os.path.join(config.model.checkpoint_dir, category,config.model.epochs)
                    # if not os.path.exists(model_save_dir):
                    #     os.mkdir(model_save_dir)
                    # # torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch+1))) 
                    #     # torch.save(model.state_dict(),os.path.join(os.path.join(config.data.data_dir, config.data.category),"save.pt"))
                    #     torch.save(model.state_dict(), os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.epochs),"save.pt"))
        if (epoch>=100):
            mean_loss.append(sum(inter_loss)/len(inter_loss))
            diff_loss.append(max(inter_loss)-min(inter_loss))
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(mean_loss) + 1), mean_loss, marker='o', linestyle='-', color='b', label='mean Loss')
    plt.plot(range(1, len(diff_loss) + 1), diff_loss, marker='o', linestyle='-', color='r', label='diff Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    # plt.savefig('/content/drive/MyDrive/Colab Notebooks/loss.png')

    # if config.model.DDADS==True:
    #     torch.save(model.state_dict(), os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.epochs),"smallUnet","save.pt"))
    # else:
    #     torch.save(model.state_dict(), os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.epochs),"bigUnet","save.pt"))#加的
