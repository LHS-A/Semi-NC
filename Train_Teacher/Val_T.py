# -- coding: utf-8 --
import torch
from metric import *
from utils import move_file,calculate_mean_and_std
from copy import deepcopy
from visualizer import Visualizer

def val_T(args,device,val_loader,T_model,criterion,epoch):
    print("================================== Valid Teacher Epoch:{} ====================================".format(epoch))
    vis = Visualizer(env=args.env_name_T, port=args.vis_port)
    batch_sen_nerve,batch_dice_nerve,batch_pre_nerve,batch_fdr_nerve,batch_MHD_nerve = [],[],[],[],[]
    batch_sen_cell,batch_dice_cell,batch_pre_cell,batch_fdr_cell,batch_MHD_cell = [],[],[],[],[]
    T_model.eval()
    with torch.no_grad():
        for image_lst,SDF_image_lst,nerve_label_lst,cell_label_lst,image_name_lst in val_loader:
            image = image_lst.float().to(device)
            SDF_image = SDF_image_lst.float().to(device)
            nerve_label = nerve_label_lst.float().to(device)
            cell_label = cell_label_lst.float().to(device)

            if args.train_SDF == True:
                pred_N,pred_C = T_model(SDF_image) #(B,C,H.W),pred is score!
            else: 
                pred_N,pred_C = T_model(image) #(B,C,H.W),pred is score! 

            pred_BCE = 0.5 * criterion["BCEloss"](pred_N, nerve_label) + 0.5 * criterion["BCEloss"](pred_C, cell_label)
            pred_dice = 0.5 * criterion["DiceLoss"](pred_N, nerve_label) + 0.5 * criterion["DiceLoss"](pred_C, cell_label)
            loss = pred_BCE + pred_dice

            args.val_loss.append(loss.item())
            # print("[{0:d}] --- val_loss:{1:.10f}".format(len(args.val_loss), loss.item()))
            vis.plot(win="val_loss", y=loss.item(), con_point=len(args.val_loss),
                    opts=dict(title="验证损失", xlabel="val_numbers", ylabel="val_loss"))

            nerve_sen,nerve_dice, nerve_pre, nerve_FDR, nerve_MHD = batch_metrics_pred(args,vis,image,torch.sigmoid(pred_N),nerve_label, "nerve", image_name_lst, args.val_batch)
            cell_sen,cell_dice, cell_pre, cell_FDR, cell_MHD = batch_metrics_pred(args,vis,image,torch.sigmoid(pred_C),cell_label, "cell", image_name_lst, args.val_batch)

            batch_sen_nerve.append(nerve_sen);batch_dice_nerve.append(nerve_dice);batch_pre_nerve.append(nerve_pre);batch_fdr_nerve.append(nerve_FDR);batch_MHD_nerve.append(nerve_MHD)
            batch_sen_cell.append(cell_sen);batch_dice_cell.append(cell_dice);batch_pre_cell.append(cell_pre);batch_fdr_cell.append(cell_FDR);batch_MHD_cell.append(cell_MHD)

        sen_cell, sen_cell_std, sen_percls_mean_cell,sen_percls_std_cell = calculate_mean_and_std(batch_sen_cell)
        dice_cell, dice_cell_std, dice_percls_mean_cell,dice_percls_std_cell = calculate_mean_and_std(batch_dice_cell)
        pre_cell, pre_cell_std, pre_percls_mean_cell,pre_percls_std_cell = calculate_mean_and_std(batch_pre_cell)
        fdr_cell, fdr_cell_std, fdr_percls_mean_cell,fdr_percls_std_cell = calculate_mean_and_std(batch_fdr_cell)
        MHD_cell, MHD_cell_std, MHD_percls_mean_cell,MHD_percls_std_cell = calculate_mean_and_std(batch_MHD_cell)

        sen_nerve, sen_nerve_std, sen_percls_mean_nerve,sen_percls_std_nerve = calculate_mean_and_std(batch_sen_nerve)
        dice_nerve, dice_nerve_std, dice_percls_mean_nerve,dice_percls_std_nerve = calculate_mean_and_std(batch_dice_nerve)
        pre_nerve, pre_nerve_std, pre_percls_mean_nerve,pre_percls_std_nerve = calculate_mean_and_std(batch_pre_nerve)
        fdr_nerve, fdr_nerve_std, fdr_percls_mean_nerve,fdr_percls_std_nerve = calculate_mean_and_std(batch_fdr_nerve)
        MHD_nerve, MHD_nerve_std, MHD_percls_mean_nerve,MHD_percls_std_nerve = calculate_mean_and_std(batch_MHD_nerve)

        print("================================ Epoch:{} Val Metric =====================================".format(epoch))
        print("sen_PerCls_nerve: {}±{}, dice_PerCls_nerve: {}±{}, pre_PerCls_nerve: {}±{}, fdr_PerCls_nerve: {}±{}, MHD_PerCls_nerve: {}±{}".format(sen_percls_mean_nerve,sen_percls_std_nerve,dice_percls_mean_nerve,dice_percls_std_nerve,pre_percls_mean_nerve,pre_percls_std_nerve,fdr_percls_mean_nerve,fdr_percls_std_nerve,MHD_percls_mean_nerve,MHD_percls_std_nerve))
        print("sen_PerCls_cell: {}±{}, dice_PerCls_cell: {}±{}, pre_PerCls_cell: {}±{}, fdr_PerCls_cell: {}±{}, MHD_PerCls_cell: {}±{}".format(sen_percls_mean_cell,sen_percls_std_cell,dice_percls_mean_cell,dice_percls_std_cell,pre_percls_mean_cell,pre_percls_std_cell,fdr_percls_mean_cell,fdr_percls_std_cell,MHD_percls_mean_cell,MHD_percls_std_cell))

        return loss, (dice_nerve + dice_cell) / 2