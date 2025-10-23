# -- coding: utf-8 --
import sys
sys.path.append(r"/data/Desktop/Semi-NC/") 

import torch.nn as nn
import torch.optim as optim
import random
import torch
from config import *
args = Params() #all files only have one args，all files can change it，but only this one！
from utils import *
from Loss_utils import *
from Train_Student.Train_S import train_S 
from Train_Student.Val_S import val_S         
from Train_Student.Test_S import test_S 
from Train_Student.dataloader_S import SemiSupervisedDataLoaderFactory  
import warnings 
import time
warnings.filterwarnings("ignore")
import ssl  
ssl._create_default_https_context = ssl._create_unverified_context

from Network.Pro_Extract.Net_T import Net_T
from Network.Pro_Extract.Net_S import Net_S
 
seed = 7170 
torch.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 
np.random.seed(seed) 
random.seed(seed)   

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device("cuda:{}".format(args.gpu_ids) if torch.cuda.is_available() else "cpu")
print("============================{}===========================".format(device))
T_model = Net_T(args.input_dim,args.num_classes).to(device)
S_model = Net_S(args.input_dim,args.num_classes).to(device) 
optimizer_S = optim.Adam(S_model.parameters(), lr = args.init_lr_S)
scheduler_S = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_S, mode="min", factor=0.99, patience=5, verbose=True, threshold=0, threshold_mode="rel", cooldown=0, min_lr=1e-6, eps=1e-08)
criterion  = {"DiceLoss":DiceLoss(args.num_classes).to(device),"CEloss":nn.CrossEntropyLoss().to(device),"BCEloss":nn.BCEWithLogitsLoss().to(device)}
  
data_loader = SemiSupervisedDataLoaderFactory() 
val_loader = data_loader.load_val_data(args,batch_size = args.val_batch)
test_loader = data_loader.load_test_data(args,batch_size = args.test_batch)
train_loader = data_loader.load_train_data(args,batch_size = args.train_batch) 
# ========================================== Train Model ==============================================

start = time.time() 
 
# 载入教师模型训练权重以执行Pro-Refine
# ==================== CORN-1 ===========================
# T_model.load_state_dict(torch.load(r"/data/Desktop/Semi-NC/Best_model_for_publish/Teacher_Weights/CORN-1_5/best_model_157.pkl",map_location='cuda:0'))
# T_model.load_state_dict(torch.load(r"/data/Desktop/Semi-NC/Best_model_for_publish/Teacher_Weights/CORN-1_20/best_model_113.pkl",map_location='cuda:0'))
# T_model.load_state_dict(torch.load(r"/data/Desktop/Semi-NC/Best_model_for_publish/Teacher_Weights/CORN-1_50/best_model_42.pkl",map_location='cuda:0'))
# ==================== CORN-Pro ===========================
# T_model.load_state_dict(torch.load(r"/data/Desktop/Semi-NC/Best_model_for_publish/Teacher_Weights/CORN-Pro_5/best_model_155.pkl",map_location='cuda:0'))
# T_model.load_state_dict(torch.load(r"/data/Desktop/Semi-NC/Best_model_for_publish/Teacher_Weights/CORN-Pro_20/best_model_56.pkl",map_location='cuda:0'))
# T_model.load_state_dict(torch.load(r"/data/Desktop/Semi-NC/Best_model_for_publish/Teacher_Weights/CORN-Pro_50/best_model_35.pkl",map_location='cuda:0'))
# # ==================== CORN-Complex ===========================
# T_model.load_state_dict(torch.load(r"/data/Desktop/Semi-NC/Best_model_for_publish/Teacher_Weights/CORN-Complex677_1/best_model_226.pkl",map_location='cuda:0')) 
# T_model.load_state_dict(torch.load(r"/data/Desktop/Semi-NC/Best_model_for_publish/Teacher_Weights/CORN-Complex677_5/best_model_85.pkl",map_location='cuda:0'))
T_model.load_state_dict(torch.load(r"/data/Desktop/Semi-NC/Best_model_for_publish/Teacher_Weights/CORN-Complex677_10/best_model_24.pkl",map_location='cuda:0'))
print("Start training Student!")

step = 0 
start_epoch = 0
start_epoch = setup_training_resume(args, S_model, optimizer_S)
# S_model.load_state_dict(torch.load(r"/data/Desktop/Semi-NC/Best_model_for_publish/TrainComplex6770_TestPro_10/best_model_58.pkl",map_location='cuda:0'))
args.enhance_mode_S = "train"

for epoch in range(start_epoch, args.epochs_S): 
    args.epoch_S = epoch    
    if args.epoch_S > args.switch_epoch:
        args.lambda_proto = 0.2
    else:  
        args.lambda_proto = 0  
    
    step = train_S(args, device, train_loader, S_model, T_model, optimizer_S, criterion, epoch, step=step)
    # loss_S, _ = val_S(args, device, val_loader, S_model, criterion, epoch)
    dice_pred = test_S(args, device, test_loader, S_model, criterion, epoch)
     
    if dice_pred > args.best_dice:  
        args.best_dice = dice_pred
        best_model_path = os.path.join(args.KD_S_Bestmodel_path, f"best_model_{epoch}.pkl")
        torch.save(S_model.state_dict(), best_model_path)
        delete_previous_models(args.KD_S_Bestmodel_path)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': S_model.state_dict(),
            'optimizer_state_dict': optimizer_S.state_dict(),  
            'best_dice': args.best_dice,
        }
        checkpoint_path = os.path.join(args.S_checkpoint_dir_path, f"best_checkpoint_{epoch}.pkl")
        torch.save(checkpoint, checkpoint_path)
        delete_previous_checkpoints(args.S_checkpoint_dir_path, keep_current=checkpoint_path)
     