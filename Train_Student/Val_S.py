# -- coding: utf-8 --
import torch
from metric import *
from utils import move_file,calculate_mean_and_std
from copy import deepcopy
from visualizer import Visualizer

def val_S(args, device, val_loader, S_model, criterion, epoch):
    vis = Visualizer(env=args.env_name_S, port=args.vis_port)  
    print("================================== Valid Student Epoch:{} ====================================".format(epoch))
    S_model.eval()

    batch_sen_nerve, batch_dice_nerve, batch_pre_nerve, batch_fdr_nerve, batch_MHD_nerve = [], [], [], [], []
    batch_sen_cell, batch_dice_cell, batch_pre_cell, batch_fdr_cell, batch_MHD_cell = [], [], [], [], []
 
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
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

            args.val_loss.append(loss.item())
            vis.plot(win="val_loss", y=loss.item(), con_point=len(args.val_loss),
                    opts=dict(title="val_loss", xlabel="val_numbers", ylabel="val_loss"))
            
            nerve_sen, nerve_dice, nerve_pre, nerve_FDR, nerve_MHD = batch_metrics_pred(
                args, vis, image, torch.sigmoid(pred_N), nerve_label, "nerve", image_name_lst, args.val_batch)
            cell_sen, cell_dice, cell_pre, cell_FDR, cell_MHD = batch_metrics_pred(
                args, vis, image, torch.sigmoid(pred_C), cell_label, "cell", image_name_lst, args.val_batch)

            batch_sen_nerve.append(nerve_sen); batch_dice_nerve.append(nerve_dice)
            batch_pre_nerve.append(nerve_pre); batch_fdr_nerve.append(nerve_FDR); batch_MHD_nerve.append(nerve_MHD)
            batch_sen_cell.append(cell_sen); batch_dice_cell.append(cell_dice)
            batch_pre_cell.append(cell_pre); batch_fdr_cell.append(cell_FDR); batch_MHD_cell.append(cell_MHD)

        sen_cell, _, _, _ = calculate_mean_and_std(batch_sen_cell)
        dice_cell, _, _, _ = calculate_mean_and_std(batch_dice_cell)
        pre_cell, _, _, _ = calculate_mean_and_std(batch_pre_cell)
        fdr_cell, _, _, _ = calculate_mean_and_std(batch_fdr_cell)
        MHD_cell, _, _, _ = calculate_mean_and_std(batch_MHD_cell)

        sen_nerve, _, _, _ = calculate_mean_and_std(batch_sen_nerve)
        dice_nerve, _, _, _ = calculate_mean_and_std(batch_dice_nerve)
        pre_nerve, _, _, _ = calculate_mean_and_std(batch_pre_nerve)
        fdr_nerve, _, _, _ = calculate_mean_and_std(batch_fdr_nerve)
        MHD_nerve, _, _, _ = calculate_mean_and_std(batch_MHD_nerve)

        print(f"Epoch {epoch} | nerve dice: {dice_nerve:.4f}, cell dice: {dice_cell:.4f}")
        return loss, (dice_nerve + dice_cell) / 2
