# -- coding: utf-8 --
"""
@Model:Semi-Net
@Time:2025/5/1
@Author:lihongshuo
"""
import os
from openpyxl import Workbook
import importlib
import numpy as np
from utils import move_file
from Dataset_List import DatasetParameters

class Params():
    def __init__(self):

        self.root_path = r"/data/Desktop/Semi-NC/"
# ==================================================== Change ================================================
        self.dataset = "CORN-Complex677" #  CORN-Complex677  CORN-Pro  CORN-1  CORN-Complex6770  TrainComplex677_Test1  TrainComplex677_TestPro
        self.data_path_selection = "teacher" # ratio( student teacher )  complete( origin general )
        # if self.data_path_selection in ["teacher","student"]:
        self.labeled_ratio = "10" # 5 20 50  1 5 10  
        self.model_name = "Dicon" # SemiNC Dicon NerveFormer
        self.train_SDF = False # True False # Train teacher with SDF or original images.
        self.marker = "FULLY"
        self.resume_training = False
        if self.resume_training == True: 
            self.S_checkpoint_dir_path_last = r"/data/Desktop/Semi-NC/checkpoints/S/S_student_SemiNC_TrainComplex6770_TestPro_10_XU"
# ==================================================== Change ================================================
        self.Course_inference_folder_path = r"/data/Desktop/Semi-NC/Dataset/Train_Origin/CORN-1/train/nerve_label"
        self.lambda_proto = 0.2
        self.lambda_KD = 0.3
        self.switch_epoch = 100  

        if self.data_path_selection == "teacher":
            # [CORN-Complex677 CORN-Pro CORN-1] # Train teacher!
            self.data_path = r"/data/Desktop/Semi-NC/Dataset/Train_Teacher" + r"/" + self.dataset + r"/" + self.labeled_ratio  
        elif self.data_path_selection == "student": 
            # [CORN-Complex677 CORN-Complex6770 CORN-Pro CORN-1] # Train student!   
            self.data_path = r"/data/Desktop/Semi-NC/Dataset/Train_Student/" + self.dataset + r"/" + self.labeled_ratio  
        elif self.data_path_selection == "origin":  
            # [CORN-Complex677 CORN-Pro CORN-1]  # Train Original image! 
            self.data_path = r"/data/Desktop/Semi-NC/Dataset/Train_Origin/" + self.dataset 
        else: 
            # [TrainComplex677_Test1   TrainComplex677_TestPro]  # Train Generalization image! 
            self.data_path = r"/data/Desktop/Semi-NC/Dataset/Train_Generalization/" + self.dataset 
  
        self.S_mark_name = "S_" + self.data_path_selection + "_" + self.model_name + "_" + self.dataset + "_" + self.labeled_ratio + "_" + self.marker
        self.env_name_S = "EA2025MIA_" + self.S_mark_name
        self.T_mark_name = "T_" + self.data_path_selection + "_" + self.model_name + "_" + self.dataset + "_" + self.labeled_ratio + "_" + self.marker
        self.env_name_T = "EA2025MIA_" + self.T_mark_name

        self.device_ids = [0]      
        self.gpu_ids = "0"        
        self.sync_bn = False    
        self.zoom_factor = 8 
        self.shot = 1 
        self.train_iter = 100  
        self.eval_iter = 100
        self.pyramid = False
        self.beta = 3 # For teacher model inference!
        self.confidence_threshold = 0.99 
            
        dataset_params = DatasetParameters(self.dataset)
        self.mode_metric = dataset_params.parameters["mode_metric"]
        self.save_mode = dataset_params.parameters["save_mode"]
        self.lower_limit = dataset_params.parameters["lower_limit"]
        self.upper_limit = dataset_params.parameters["upper_limit"]
        self.image_folder = dataset_params.parameters["image_folder"]
        self.label_folder = dataset_params.parameters["label_folder"]
        self.roi_size = [dataset_params.parameters["roi_size"][0],dataset_params.parameters["roi_size"][1]]
        self.dialated_pixels_list = dataset_params.parameters["dialated_pixels_list"]
        self.nii = dataset_params.parameters["nii"]
        self.train_label_path = dataset_params.parameters["train_label_path"]
        self.val_label_path = dataset_params.parameters["val_label_path"]
        self.sen_thed = dataset_params.parameters["sen_thed"]
        self.thed_lr_list = dataset_params.parameters["thed_lr_list"]
        self.thed_lr = dataset_params.parameters["thed_lr_list"][0]
        self.palette = dataset_params.parameters["palette"]
        self.input_dim = dataset_params.parameters["input_dim"]
        self.num_classes = dataset_params.parameters["num_classes"]
        self.train_name = dataset_params.parameters["train_name"]
        self.test_name = dataset_params.parameters["test_name"]
        self.crop = dataset_params.parameters["crop"]
        self.resize = dataset_params.parameters["resize"]
        self.mapping = dataset_params.parameters["mapping"]
      
        dataset_params = DatasetParameters(self.dataset)
        print(dataset_params.parameters)

        # ps aux | grep 3417782 
        
        self.enhance_mode_S = None
        self.enhance_mode_T = None
        self.epochs_T = 300
        self.epochs_S = 600 
        self.epoch_T = 0
        self.epoch_S = 0
         

        # nohup /home/imed/anaconda3/envs/LHS/bin/python /data/Desktop/Semi-NC/main.py > S_NCNet_CORN3.log 2>&1 &
        # nohup python -m visdom.server -port 9000 > visdom.log 2>&1 &
        self.adjust_lr = False
        self.seed_random =777
        self.checkpoint = False

        if self.crop == True:
            self.train_batch = 2
            self.val_batch = 4
            self.test_batch = 4
        elif self.data_path_selection == "teacher":
            self.train_batch = 2 
            self.val_batch = 4
            self.test_batch = 1                
        else:
            self.train_batch = 2
            self.val_batch = 4
            self.test_batch = 4   
                  
        self.init_lr_T = 1e-4 #0.01
        self.init_lr_S = 1e-4
        self.init_lr = 1e-4 
        self.lr_count = 0
        self.save_all_pkl = False
        self.open_lr_adjust = False
        self.open_OTUS = True
    
        # Ablation----All are based on 90 epoch! We make different intervals, and we make different initial 90 epoch pkls! 
        self.label = False
        self.empty_folder = False
        self.mark_epoch = 0
        self.mark_S_epoch = 0 
        self.T_val_loss = [] 
        self.mark_T_epoch = 0
        # ============================================================================================
        self.best_loss = np.inf 
       
        self.checkpoint_epoch = 150
        self.power = 0.9   
        self.lr = 1e-4
         
# ==================================================== 其他变量值 ================================================
        self.best_dice = 0
        self.best_sen = 0
        self.epoch = 0
        
        self.vis_port = 9000
        self.name = "lhs"
        self.best_epoch = 150
        self.train_loss = []
        self.test_loss = []
        self.val_loss = [] 
        self.output = self.env_name_S + "_output.log"

        self.KD_S_Bestmodel_path = r"/data/Desktop/Semi-NC/Model_Weights/best_model/S/" + self.S_mark_name 
        self.KD_T_Bestmodel_path = r"/data/Desktop/Semi-NC/Model_Weights/best_model/T/" + self.T_mark_name
        os.makedirs(self.KD_S_Bestmodel_path, exist_ok=True)   
        os.makedirs(self.KD_T_Bestmodel_path, exist_ok=True)
        self.S_checkpoint_dir_path = r"/data/Desktop/Semi-NC/checkpoints/S/" + self.S_mark_name  
        self.T_checkpoint_dir_path = r"/data/Desktop/Semi-NC/checkpoints/T/" + self.T_mark_name 
        os.makedirs(self.S_checkpoint_dir_path, exist_ok=True)   
        os.makedirs(self.T_checkpoint_dir_path, exist_ok=True)

        self.True_vision_train_path = r"/data/Desktop/Semi-NC/True_vision/train/" + self.S_mark_name
        self.True_vision_test_path = r"/data/Desktop/Semi-NC/True_vision/test/" + self.S_mark_name
        os.makedirs(self.True_vision_test_path, exist_ok=True)
        os.makedirs(self.True_vision_train_path, exist_ok=True)

        self.nerve_pseudo_folder = os.path.join(self.data_path, 'train', 'nerve_pseudo_label')
        self.cell_pseudo_folder = os.path.join(self.data_path, 'train', 'cell_pseudo_label')
        os.makedirs(self.nerve_pseudo_folder, exist_ok=True)
        os.makedirs(self.cell_pseudo_folder, exist_ok=True) 
        
        self.paper_save_nerve = r"/data/Desktop/Semi-NC/Prediction/CNs/" + self.S_mark_name
        self.paper_save_cell = r"/data/Desktop/Semi-NC/Prediction/LCs/" + self.S_mark_name 
        os.makedirs(self.paper_save_nerve, exist_ok=True)
        os.makedirs(self.paper_save_cell, exist_ok=True)            
         
        self.metric_test_nerve = { 
            "total_sen_nerve": [],  "total_dice_nerve": [], "total_pre_nerve": [], "total_fdr_nerve":[], "total_MHD_nerve": [], 
            "total_sen_nerve_std": [],  "total_dice_nerve_std": [], "total_pre_nerve_std": [], "total_fdr_nerve_std":[], "total_MHD_nerve_std": []}
        self.metrics_dict_test_nerve = {
            "Sen_nerve": {'total': self.metric_test_nerve["total_sen_nerve"]},
            "Dice_nerve": {'total': self.metric_test_nerve["total_dice_nerve"]},
            "pre_nerve": {'total': self.metric_test_nerve["total_pre_nerve"]},
            "FDR_nerve": {'total': self.metric_test_nerve["total_fdr_nerve"]},
            "MHD_nerve": {'total': self.metric_test_nerve["total_MHD_nerve"]}
        }

        self.metric_test_cell = { 
            "total_sen_cell": [],  "total_dice_cell": [], "total_pre_cell": [], "total_fdr_cell":[], "total_MHD_cell": [], 
            "total_sen_cell_std": [],  "total_dice_cell_std": [], "total_pre_cell_std": [], "total_fdr_cell_std":[], "total_MHD_cell_std": []}
        self.metrics_dict_test_cell = {
            "Sen_cell": {'total': self.metric_test_cell["total_sen_cell"]},
            "Dice_cell": {'total': self.metric_test_cell["total_dice_cell"]},
            "pre_cell": {'total': self.metric_test_cell["total_pre_cell"]},
            "FDR_cell": {'total': self.metric_test_cell["total_fdr_cell"]},
            "MHD_cell": {'total': self.metric_test_cell["total_MHD_cell"]}
        }

        self.metric_test_nerve_T = {
            "total_sen_nerve_T": [],  "total_dice_nerve_T": [], "total_pre_nerve_T": [], "total_fdr_nerve_T":[], "total_MHD_nerve_T": [], 
            "total_sen_nerve_std_T": [],  "total_dice_nerve_std_T": [], "total_pre_nerve_std_T": [], "total_fdr_nerve_std_T":[], "total_MHD_nerve_std_T": []}
        self.metrics_dict_test_nerve_T = {
            "Sen_nerve_T": {'total': self.metric_test_nerve_T["total_sen_nerve_T"]},
            "Dice_nerve_T": {'total': self.metric_test_nerve_T["total_dice_nerve_T"]},
            "pre_nerve_T": {'total': self.metric_test_nerve_T["total_pre_nerve_T"]},
            "FDR_nerve_T": {'total': self.metric_test_nerve_T["total_fdr_nerve_T"]},
            "MHD_nerve_T": {'total': self.metric_test_nerve_T["total_MHD_nerve_T"]}
        }

        self.metric_test_cell_T = { 
            "total_sen_cell_T": [],  "total_dice_cell_T": [], "total_pre_cell_T": [], "total_fdr_cell_T":[], "total_MHD_cell_T": [], 
            "total_sen_cell_std_T": [],  "total_dice_cell_std_T": [], "total_pre_cell_std_T": [], "total_fdr_cell_std_T":[], "total_MHD_cell_std_T": []}
        self.metrics_dict_test_cell_T = {
            "Sen_cell_T": {'total': self.metric_test_cell_T["total_sen_cell_T"]},
            "Dice_cell_T": {'total': self.metric_test_cell_T["total_dice_cell_T"]},
            "pre_cell_T": {'total': self.metric_test_cell_T["total_pre_cell_T"]},
            "FDR_cell_T": {'total': self.metric_test_cell_T["total_fdr_cell_T"]},
            "MHD_cell_T": {'total': self.metric_test_cell_T["total_MHD_cell_T"]}
        }