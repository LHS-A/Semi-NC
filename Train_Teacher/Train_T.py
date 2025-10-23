# -- coding: utf-8 --
import torch
import torch.nn.functional as F
from contrastive_loss import ProAlignLoss
from utils import delete_previous_models
from visualizer import Visualizer

def train_T(args, device, train_loader, T_model, TT_model, optimizer, criterion, epoch):
    vis = Visualizer(env=args.env_name_T, port=args.vis_port) 
    T_model.train()
    print("================================ Train Teacher Epoch:{} =====================================".format(epoch))
    for image_lst,SDF_image_lst,nerve_label_lst,cell_label_lst,image_name_lst in train_loader:
        image = image_lst.float().to(device)
        SDF_image = SDF_image_lst.float().to(device)
        nerve_label = nerve_label_lst.float().to(device)
        cell_label = cell_label_lst.float().to(device)

        optimizer.zero_grad() 
        if args.train_SDF == True:
            pred_N,pred_C = T_model(SDF_image) #(B,C,H.W),pred is score!
        else: 
            pred_N,pred_C = T_model(image) #(B,C,H.W),pred is score! 

        pred_BCE = 0.5 * criterion["BCEloss"](pred_N, nerve_label) + 0.5 * criterion["BCEloss"](pred_C, cell_label)
        pred_dice = 0.5 * criterion["DiceLoss"](pred_N, nerve_label) + 0.5 * criterion["DiceLoss"](pred_C, cell_label)
        loss = pred_BCE + pred_dice

        args.train_loss.append(loss.item()) 
        vis.plot(win="train_loss", y=loss.item(), con_point=len(args.train_loss),
                    opts=dict(title="训练损失", xlabel="batch", ylabel="train_loss"))

        loss.backward(retain_graph=False)
        optimizer.step()

    print("=============================================================================================================== ")
