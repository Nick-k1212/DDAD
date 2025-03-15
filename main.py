import torch
import numpy as np
import os
import argparse
from unet import *
from omegaconf import OmegaConf
from train import trainer
from feature_extractor import * 
from ddad import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

def build_model(config):
    if config.model.DDADS:
        unet = UNetModel(config.data.image_size, 32, dropout=0.3, n_heads=2 ,in_channels=config.data.input_channel)
    else:
        unet = UNetModel(config.data.image_size, 64, dropout=0.0, n_heads=4 ,in_channels=config.data.input_channel)
    return unet

def train(config):
    torch.manual_seed(42)
    np.random.seed(42)
    unet = build_model(config)
    print(" Num params: ", sum(p.numel() for p in unet.parameters())) 
    unet.to(config.model.device)
    unet.train()
    unet = torch.nn.DataParallel(unet)
    # checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'1000'))
    # unet.load_state_dict(checkpoint) 
    if config.model.still_train==True:
        checkpoint =torch.load(os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.still_train_load),"bigUnet","save.pt"))
        unet.load_state_dict(checkpoint)
    trainer(unet, config.data.category, config)#config.data.category, 


def detection(config):
    unet = build_model(config)
    # checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.load_chp)))
    if config.model.DDADS==True:
        checkpoint =torch.load(os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.epochs),"smallUnet","save.pt"))
    else:
        checkpoint =torch.load(os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.epochs),"bigUnet","save.pt"))
    # checkpoint =torch.load(os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.epochs),"save.pt"))
    unet = torch.nn.DataParallel(unet)
    unet.load_state_dict(checkpoint)    
    unet.to(config.model.device)
    # checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.load_chp)))
    # checkpoint =torch.load(os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.epochs),"save.pt"))
    if config.model.DDADS==True:
        checkpoint =torch.load(os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.epochs),"smallUnet","save.pt"))
    else:
        checkpoint =torch.load(os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.epochs),"bigUnet","save.pt"))
    unet.eval()
    ddad = DDAD(unet, config)
    ddad()  
    
def finetuning(config):
    unet = build_model(config)
    # checkpoint = torch.load(os.path.join(os.getcwd(), config.model.c]heckpoint_dir, config.data.category, str(config.model.load_chp)))
    # torch.save(unet.state_dict(),"C:\dataset\MVTec\screw\save.pt")#改的
    # checkpoint = torch.load(os.path.join(os.path.join(config.data.data_dir, "train",config.data.category),str(config.model.epochs),"save.pt"))#改的
    # checkpoint =torch.load(os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.epochs),"save.pt"))
    if config.model.DDADS==True:
        checkpoint =torch.load(os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.epochs),"smallUnet","save.pt"))
    else:
        checkpoint =torch.load(os.path.join(os.path.join(config.data.data_dir,"train", config.data.category),str(config.model.w),str(config.model.epochs),"bigUnet","save.pt"))
    unet = torch.nn.DataParallel(unet)
    # unet.load_state_dict(checkpoint)
    unet.load_state_dict(checkpoint, strict=False)    
    unet.to(config.model.device)
    unet.eval()
    domain_adaptation(unet, config, fine_tune=True)




def parse_args():
    cmdline_parser = argparse.ArgumentParser('DDAD')    
    cmdline_parser.add_argument('-cfg', '--config', 
                                default= os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), 
                                help='config file')
    cmdline_parser.add_argument('--train', 
                                default= True, #False
                                help='Train the diffusion model') 
    cmdline_parser.add_argument('--detection', 
                                default= True, #False
                                help='Detection anomalies')
    cmdline_parser.add_argument('--domain_adaptation', 
                                default= True, #False
                                help='Domain adaptation')
    args, unknowns = cmdline_parser.parse_known_args()
    return args


###加的
# def save_untrained_model(checkpoint_dir,config):
#     """
#     建立未經訓練的模型並存儲到指定資料夾。
#     """
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
    
#     model = build_model(config)
#     model_path = os.path.join(checkpoint_dir, "untrained_model.pth")
#     torch.save(model.state_dict(), model_path)
#     print(f"Untrained model saved to {model_path}")

###加的
    
if __name__ == "__main__":
    # checkpoint_path = os.path.join("C:\dataset\MVTec\screw")
    # checkpoint = torch.load(checkpoint_path)

    # if isinstance(checkpoint, dict):
    #     print(f"Checkpoint keys: {list(checkpoint.keys())[:10]}")  # 打印前10個鍵

    torch.cuda.empty_cache()
    args = parse_args()
    config = OmegaConf.load(args.config)
#==============================================================C:\Users\高韻堯\OneDrive\桌面\專題_test\DDAD\DDAD\checkpoints\MVTec
    # checkpoint_path = os.path.join("C:\dataset\MVTec\screw")  #, "model.pth"
    # if not os.path.isfile(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
#==========================================

    print("Class: ",config.data.category, "   w:", config.model.w, "   v:", config.model.v, "   load_chp:", config.model.load_chp,   "   feature extractor:", config.model.feature_extractor,"         w_DA: ",config.model.w_DA,"         DLlambda: ",config.model.DLlambda)
    print(f'{config.model.test_trajectoy_steps=} , {config.data.test_batch_size=}')
    torch.manual_seed(42)
    np.random.seed(42)
    print("1")
    if torch.cuda.is_available():
        print("1.5")
        torch.cuda.manual_seed_all(42)
    if args.train:
        print("2")
        print('Training...')
        train(config)
    if args.domain_adaptation:
        print("3")
        print('Domain Adaptation...')
        # save_untrained_model("C:\dataset\MVTec\screw",config)
        finetuning(config)
    if args.detection:
        print("4")
        print('Detecting Anomalies...')
        detection(config)
