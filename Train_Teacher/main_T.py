# -- coding: utf-8 --
import sys
sys.path.append(r"/data/Desktop/Semi-NC") 
import torch.nn as nn
import torch.optim as optim
import random
import torch
from config import *
args = Params()
from utils import *
from Loss_utils import *
from Train_Teacher.Train_T import train_T 
from Train_Teacher.Val_T import val_T         
from Train_Teacher.Test_T import test_T  
from Train_Teacher.dataloader_T import Data_loader
import warnings
import time
warnings.filterwarnings("ignore")
import ssl  
ssl._create_default_https_context = ssl._create_unverified_context

from Network.Pro_Extract.Net_T import Net_T

##========================================================== 模型参数设置 =========================================================
seed = 7170 
torch.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 
np.random.seed(seed) 
random.seed(seed)   

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device("cuda:{}".format(args.gpu_ids) if torch.cuda.is_available() else "cpu")
print("============================{}===========================".format(device))
T_model = Net_T(args.input_dim,args.num_classes).to(device)   
optimizer_T = optim.Adam(T_model.parameters(), lr = args.init_lr_T)
scheduler_T = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_T, mode="min", factor=0.99, patience=5, verbose=True, threshold=0, threshold_mode="rel", cooldown=0, min_lr=1e-6, eps=1e-08)
criterion  = {"DiceLoss":DiceLoss(args.num_classes).to(device),"CEloss":nn.CrossEntropyLoss().to(device),"BCEloss":nn.BCEWithLogitsLoss().to(device)}
  
data_loader = Data_loader()
val_loader = data_loader.load_val_data(args,batch_size = args.val_batch)
test_loader = data_loader.load_test_data(args,batch_size = args.test_batch)
train_loader = data_loader.load_train_data(args,batch_size = args.train_batch) 
# ========================================== Train Model ==============================================
start = time.time() 
print("Start training Teacher!")
start_epoch = 0

start_epoch = setup_training_resume(args, T_model, optimizer_T)

args.enhance_mode_T = "train"

for epoch in range(start_epoch, args.epochs_T): 
    args.epoch_T = epoch       
  
    train_T(args, device, train_loader, T_model, T_model, optimizer_T, criterion, epoch)
    # loss_T = val_T(args, device, val_loader, T_model, criterion, epoch)
    dice_pred = test_T(args, device, test_loader, T_model, criterion, epoch)
     
    if dice_pred > args.best_dice:  
        args.best_dice = dice_pred
      
        best_model_path = os.path.join(args.KD_T_Bestmodel_path, f"best_model_{epoch}.pkl")
        torch.save(T_model.state_dict(), best_model_path)
        delete_previous_models(args.KD_T_Bestmodel_path)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': T_model.state_dict(),
            'optimizer_state_dict': optimizer_T.state_dict(), 
            'best_dice': args.best_dice,
        }
        checkpoint_path = os.path.join(args.T_checkpoint_dir_path, f"best_checkpoint_{epoch}.pkl")
        torch.save(checkpoint, checkpoint_path)
        delete_previous_checkpoints(args.T_checkpoint_dir_path, keep_current=checkpoint_path)
    
 