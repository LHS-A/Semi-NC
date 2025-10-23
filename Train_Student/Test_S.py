# -- coding: utf-8 --

from metric import *
import torch
from visualizer import Visualizer
from Loss_utils import DiceLoss
from copy import deepcopy
import torch.nn as nn
from config import *
args = Params() 
from utils import calculate_mean_and_std

def test_S(args, device, test_loader, S_model, criterion, epoch):
    vis = Visualizer(env=args.env_name_S, port=args.vis_port) 
    print("================================== Test Student Epoch:{} ====================================".format(epoch))
    S_model.eval()

    batch_sen_nerve, batch_dice_nerve, batch_pre_nerve, batch_fdr_nerve, batch_MHD_nerve = [], [], [], [], []
    batch_sen_cell, batch_dice_cell, batch_pre_cell, batch_fdr_cell, batch_MHD_cell = [], [], [], [], []
 
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            image = batch['image'].float().to(device)
            SDF_image = batch['SDF_image'].float().to(device)
            nerve_label = batch['nerve_label'].float().to(device)
            cell_label = batch['cell_label'].float().to(device)
            image_name_lst = batch['image_name']
            is_labeled = batch['is_labeled']  

            pred_N, pred_C = S_model(image)

            pred_BCE = 0.5 * criterion["BCEloss"](pred_N, nerve_label) + 0.5 * criterion["BCEloss"](pred_C, cell_label)
            pred_dice = 0.5 * criterion["DiceLoss"](pred_N, nerve_label) + 0.5 * criterion["DiceLoss"](pred_C, cell_label)
            loss = pred_BCE + pred_dice

            args.test_loss.append(loss.item())
            vis.plot(win="test_loss", y=loss.item(), con_point=len(args.test_loss),
                    opts=dict(title="Test Loss", xlabel="test_numbers", ylabel="test_loss"))
            
            nerve_sen, nerve_dice, nerve_pre, nerve_FDR, nerve_MHD = batch_metrics_pred(
                args, vis, image, torch.sigmoid(pred_N), nerve_label, "nerve", image_name_lst, args.test_batch)
            cell_sen, cell_dice, cell_pre, cell_FDR, cell_MHD = batch_metrics_pred( 
                args, vis, image, torch.sigmoid(pred_C), cell_label, "cell", image_name_lst, args.test_batch)
 
            batch_sen_nerve.append(nerve_sen); batch_dice_nerve.append(nerve_dice)
            batch_pre_nerve.append(nerve_pre); batch_fdr_nerve.append(nerve_FDR); batch_MHD_nerve.append(nerve_MHD)
            batch_sen_cell.append(cell_sen); batch_dice_cell.append(cell_dice)
            batch_pre_cell.append(cell_pre); batch_fdr_cell.append(cell_FDR); batch_MHD_cell.append(cell_MHD)
 
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

        # print(f"Epoch {epoch} | nerve dice: {dice_nerve:.4f}, cell dice: {dice_cell:.4f}")

        args.metric_test_nerve["total_sen_nerve"].append(sen_nerve)
        args.metric_test_nerve["total_dice_nerve"].append(dice_nerve)
        args.metric_test_nerve["total_pre_nerve"].append(pre_nerve)
        args.metric_test_nerve["total_fdr_nerve"].append(fdr_nerve)
        args.metric_test_nerve["total_MHD_nerve"].append(MHD_nerve)

        args.metric_test_cell["total_sen_cell"].append(sen_cell)
        args.metric_test_cell["total_dice_cell"].append(dice_cell)
        args.metric_test_cell["total_pre_cell"].append(pre_cell)
        args.metric_test_cell["total_fdr_cell"].append(fdr_cell)
        args.metric_test_cell["total_MHD_cell"].append(MHD_cell)

        vis.plot_metrics_total(args.metrics_dict_test_nerve)
        vis.plot_metrics_total(args.metrics_dict_test_cell)

        print("================================ Epoch:{} Student Test Metric =====================================".format(epoch))
        print("sen_nerve: {}±{}, dice_nerve: {}±{}, pre_nerve: {}±{}, fdr_nerve: {}±{}, MHD_nerve: {}±{}".format(sen_percls_mean_nerve,sen_percls_std_nerve,dice_percls_mean_nerve,dice_percls_std_nerve,pre_percls_mean_nerve,pre_percls_std_nerve,fdr_percls_mean_nerve,fdr_percls_std_nerve,MHD_percls_mean_nerve,MHD_percls_std_nerve))
        print("sen_cell: {}±{}, dice_cell: {}±{}, pre_cell: {}±{}, fdr_cell: {}±{}, MHD_cell: {}±{}".format(sen_percls_mean_cell,sen_percls_std_cell,dice_percls_mean_cell,dice_percls_std_cell,pre_percls_mean_cell,pre_percls_std_cell,fdr_percls_mean_cell,fdr_percls_std_cell,MHD_percls_mean_cell,MHD_percls_std_cell))

        return (dice_nerve + dice_cell) / 2
        
        
