# -- coding: utf-8 -- 
import sys 
sys.path.append(r"/data/Desktop/Semi-NC/") 
import torch
import torch.nn.functional as F
from visualizer import Visualizer
from contrastive_loss import ProAlignLoss
from sacnet import build_sac_prototypes, Weighted_GAP, sp_center_iter, build_prototypes_unlabeled, PrototypePool
from Loss_utils import Compute_KDloss 
from Train_Student.dataloader_S import create_unlabeled_loader
from utils import *

def train_S(args, device, train_loader, S_model, T_model, optimizer, criterion, epoch, proto_pool=None, step=0):

    vis = Visualizer(env=args.env_name_S, port=args.vis_port)  
    print(f"================================ Train Student Epoch:{epoch} =====================================") 
    
    confidence_threshold = max(0.7, 0.999 - 0.001 * epoch)

    valid_unlabeled_samples = 0
    
    if epoch % 30 == 0 and epoch != 0: 
        S_model.eval()               
        unlabeled_loader = create_unlabeled_loader(args, batch_size=4)          
        high_conf_count = generate_pseudo_labels(args, device, S_model, unlabeled_loader, epoch)

    S_model.train()
    T_model.eval()    
    
    if args.epoch_S > args.switch_epoch:    
        proto_pool_nerve = proto_pool or PrototypePool(device=device)
        proto_pool_cell = proto_pool or PrototypePool(device=device)
        proto_pool_bg = proto_pool or PrototypePool(device=device)
        def check_pool_health():
            nerve_health, nerve_msg = proto_pool_nerve.health_check()
            cell_health, cell_msg = proto_pool_cell.health_check()
            bg_health, bg_msg = proto_pool_bg.health_check()
            
            return nerve_health and cell_health and bg_health

    for batch_idx, batch in enumerate(train_loader):
        image = batch['image'].float().to(device)
        SDF_image = batch['SDF_image'].float().to(device)
        nerve_label = batch['nerve_label'].float().to(device)
        cell_label = batch['cell_label'].float().to(device)
        image_names = batch['image_name']
        is_labeled = batch['is_labeled'] 
        optimizer.zero_grad()

        is_labeled_batch = is_labeled.any()
        is_unlabeled_batch = not is_labeled_batch
        
        if is_unlabeled_batch and args.epoch_S <= args.switch_epoch:
            with torch.no_grad():
                teacher_pred_N, teacher_pred_C = T_model(SDF_image)
                confidence_N = torch.sigmoid(teacher_pred_N).flatten(1).mean(dim=1)
                confidence_C = torch.sigmoid(teacher_pred_C).flatten(1).mean(dim=1)
                sample_confidences = (confidence_N + confidence_C) / 2
                
                has_valid_samples = (sample_confidences > confidence_threshold).any()
                
                if not has_valid_samples:
                    step += 1
                    continue  

        with torch.no_grad():
            teacher_pred_N, teacher_pred_C = T_model(SDF_image) 
            Feas_teacher = T_model.combined_features

        if is_unlabeled_batch:
            with torch.no_grad():
                pred_N, pred_C = S_model(image)
                Feas_student = S_model.combined_features
        else:
            pred_N, pred_C = S_model(image)
            Feas_student = S_model.combined_features

        if is_labeled_batch:        
            pred_BCE = 0.5 * criterion["BCEloss"](pred_N, nerve_label) + 0.5 * criterion["BCEloss"](pred_C, cell_label)
            pred_dice = 0.5 * criterion["DiceLoss"](pred_N, nerve_label) + 0.5 * criterion["DiceLoss"](pred_C, cell_label)
            Seg_loss = pred_BCE + pred_dice
        else:
            Seg_loss = torch.tensor(0.0, device=device)   

        KD_Loss = torch.tensor(0.0, device=device)
        valid_kd_samples = 0 
        
        if is_labeled_batch:
            KD_Loss = Compute_KDloss(Feas_teacher, Feas_student)      
        else:
            confidence_N = torch.sigmoid(teacher_pred_N).flatten(1).mean(dim=1)
            confidence_C = torch.sigmoid(teacher_pred_C).flatten(1).mean(dim=1)
            sample_confidences = (confidence_N + confidence_C) / 2
            
            for i in range(image.size(0)):
                sample_confidence = sample_confidences[i].item()
                
                if sample_confidence > confidence_threshold:
                    sample_kd_loss = Compute_KDloss(
                        [feat[i:i+1] for feat in Feas_teacher], 
                        [feat[i:i+1] for feat in Feas_student]
                    )
                    KD_Loss = KD_Loss + sample_kd_loss
                    valid_kd_samples += 1
                    valid_unlabeled_samples += 1
                else:
                    if batch_idx == 0 and i == 0:
                        pass
            
            if valid_kd_samples > 0:
                KD_Loss = KD_Loss / valid_kd_samples
            else:
                KD_Loss = torch.tensor(0.0, device=device)

        skip_gradient_update = False
        
        if is_unlabeled_batch:
            if valid_kd_samples == 0:
                if args.epoch_S <= args.switch_epoch:
                    skip_gradient_update = True
                else:
                    skip_gradient_update = True
            else:
                if args.epoch_S > args.switch_epoch:
                    skip_gradient_update = True
        
        if skip_gradient_update:
            if args.epoch_S > args.switch_epoch and is_unlabeled_batch:
                with torch.no_grad():
                    feat_up14 = S_model.combined_features[8]
                    feat_up24 = S_model.combined_features[13]
                    
                    nerve_label_down = F.interpolate(
                        nerve_label, size=(feat_up14.size(2), feat_up14.size(3)), mode='nearest'
                    ).squeeze(1)
                    cell_label_down = F.interpolate(
                        cell_label, size=(feat_up24.size(2), feat_up24.size(3)), mode='nearest'
                    ).squeeze(1)

                    for b in range(image.size(0)):
                        if nerve_label[b].sum() > 0:
                            protos_nerve = build_sac_prototypes(
                                feat_up14[b], nerve_label_down[b:b+1], sp_center_iter, num_proto=3, n_iter=10
                            )
                            if protos_nerve is not None:
                                proto_pool_nerve.add_prototypes_safe(protos_nerve)
                        else:
                            logits_nerve_down = F.interpolate(
                                pred_N[b:b+1], size=(feat_up14.size(2), feat_up14.size(3)), mode='bilinear'
                            )
                            protos_unlabeled = build_prototypes_unlabeled(
                                feat_up14[b], logits_nerve_down, sp_center_iter,
                                step=step, num_proto_per_class=3, use_sac_on_high_conf=True
                            )
                            if protos_unlabeled:
                                for vec, label, conf in protos_unlabeled:
                                    if label == 0:
                                        proto_pool_bg.add_prototypes_safe(vec.unsqueeze(1))
                                    elif label == 1:
                                        proto_pool_nerve.add_prototypes_safe(vec.unsqueeze(1))

                        if cell_label[b].sum() > 0:
                            protos_cell = build_sac_prototypes(
                                feat_up24[b], cell_label_down[b:b+1], sp_center_iter, num_proto=3, n_iter=10
                            )
                            if protos_cell is not None:
                                proto_pool_cell.add_prototypes_safe(protos_cell)
                        else:
                            logits_cell_down = F.interpolate(
                                pred_C[b:b+1], size=(feat_up24.size(2), feat_up24.size(3)), mode='bilinear'
                            )
                            protos_unlabeled = build_prototypes_unlabeled(
                                feat_up24[b], logits_cell_down, sp_center_iter,
                                step=step, num_proto_per_class=3, use_sac_on_high_conf=True
                            )
                            if protos_unlabeled:
                                for vec, label, conf in protos_unlabeled:
                                    if label == 0:
                                        proto_pool_bg.add_prototypes_safe(vec.unsqueeze(1))
                                    elif label == 2:
                                        proto_pool_cell.add_prototypes_safe(vec.unsqueeze(1))

                        if nerve_label[b].sum() > 0 or cell_label[b].sum() > 0:
                            bg_mask = ((nerve_label_down[b] + cell_label_down[b]) == 0).float().unsqueeze(0)
                            if bg_mask.sum() > 0:
                                bg_proto = Weighted_GAP((feat_up14[b:b + 1] + feat_up24[b:b + 1]) / 2, bg_mask)
                                bg_proto = bg_proto.squeeze(-1).squeeze(-1).squeeze(0).unsqueeze(1)
                                proto_pool_bg.add_prototypes_safe(bg_proto)

                    if proto_pool_nerve is not None:
                        proto_pool_nerve.merge_to_capacity(capacity=256)
                    if proto_pool_cell is not None:
                        proto_pool_cell.merge_to_capacity(capacity=256)
                    if proto_pool_bg is not None:
                        proto_pool_bg.merge_to_capacity(capacity=128)
                
            del image, SDF_image, nerve_label, cell_label, pred_N, pred_C, Feas_teacher, Feas_student
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            step += 1
            continue  

        if not skip_gradient_update:
            loss = Seg_loss + args.lambda_KD * KD_Loss

            pcl_active = False
            if args.epoch_S > args.switch_epoch: 
                feat_up14 = S_model.combined_features[8]
                feat_up24 = S_model.combined_features[13]
                
                nerve_label_down = F.interpolate(
                    nerve_label, size=(feat_up14.size(2), feat_up14.size(3)), mode='nearest'
                ).squeeze(1)
                cell_label_down = F.interpolate(
                    cell_label, size=(feat_up24.size(2), feat_up24.size(3)), mode='nearest'
                ).squeeze(1)

                for b in range(image.size(0)):
                    if nerve_label[b].sum() > 0:
                        protos_nerve = build_sac_prototypes(
                            feat_up14[b], nerve_label_down[b:b+1], sp_center_iter, num_proto=3, n_iter=10
                        )
                        if protos_nerve is not None:
                            proto_pool_nerve.add_prototypes_safe(protos_nerve)
                    else:
                        logits_nerve_down = F.interpolate(
                            pred_N[b:b+1], size=(feat_up14.size(2), feat_up14.size(3)), mode='bilinear'
                        )
                        protos_unlabeled = build_prototypes_unlabeled(
                            feat_up14[b], logits_nerve_down, sp_center_iter,
                            step=step, num_proto_per_class=3, use_sac_on_high_conf=True
                        )
                        if protos_unlabeled:
                            for vec, label, conf in protos_unlabeled:
                                if label == 0:
                                    proto_pool_bg.add_prototypes_safe(vec.unsqueeze(1))
                                elif label == 1:
                                    proto_pool_nerve.add_prototypes_safe(vec.unsqueeze(1))

                    if cell_label[b].sum() > 0:
                        protos_cell = build_sac_prototypes(
                            feat_up24[b], cell_label_down[b:b+1], sp_center_iter, num_proto=3, n_iter=10
                        )
                        if protos_cell is not None:
                            proto_pool_cell.add_prototypes_safe(protos_cell)
                    else:
                        logits_cell_down = F.interpolate(
                            pred_C[b:b+1], size=(feat_up24.size(2), feat_up24.size(3)), mode='bilinear'
                        )
                        protos_unlabeled = build_prototypes_unlabeled(
                            feat_up24[b], logits_cell_down, sp_center_iter,
                            step=step, num_proto_per_class=3, use_sac_on_high_conf=True
                        )
                        if protos_unlabeled:
                            for vec, label, conf in protos_unlabeled:
                                if label == 0:
                                    proto_pool_bg.add_prototypes_safe(vec.unsqueeze(1))
                                elif label == 2:
                                    proto_pool_cell.add_prototypes_safe(vec.unsqueeze(1))

                    if nerve_label[b].sum() > 0 or cell_label[b].sum() > 0:
                        bg_mask = ((nerve_label_down[b] + cell_label_down[b]) == 0).float().unsqueeze(0)
                        if bg_mask.sum() > 0:
                            bg_proto = Weighted_GAP((feat_up14[b:b + 1] + feat_up24[b:b + 1]) / 2, bg_mask)
                            bg_proto = bg_proto.squeeze(-1).squeeze(-1).squeeze(0).unsqueeze(1)
                            proto_pool_bg.add_prototypes_safe(bg_proto)

                if proto_pool_nerve is not None:
                    proto_pool_nerve.merge_to_capacity(capacity=256)
                if proto_pool_cell is not None:
                    proto_pool_cell.merge_to_capacity(capacity=256)
                if proto_pool_bg is not None:
                    proto_pool_bg.merge_to_capacity(capacity=128)

                proalign_criterion = ProAlignLoss(tau=0.07, M=64, topPpos=3, device=device)
                pools_healthy = check_pool_health()
                
                if pools_healthy and proto_pool_nerve and proto_pool_cell is not None:
                    proto_loss_N = proalign_criterion(
                        query_feat=feat_up14, labels_map=nerve_label_down, logits=pred_N,
                        pool_nerve=proto_pool_nerve, pool_cell=proto_pool_cell, pool_bg=proto_pool_bg, fenzhi='nerve'
                    )
                    proto_loss_C = proalign_criterion(
                        query_feat=feat_up24, labels_map=cell_label_down, logits=pred_C,
                        pool_nerve=proto_pool_nerve, pool_cell=proto_pool_cell, pool_bg=proto_pool_bg, fenzhi='cell'
                    )
                    
                    PCL_loss = torch.tensor(proto_loss_N + proto_loss_C)
                    if not torch.isnan(PCL_loss) and not torch.isinf(PCL_loss):
                        loss = loss + args.lambda_proto * PCL_loss
                        pcl_active = True

            args.train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            vis.plot(win="train_loss", y=Seg_loss.item(), con_point=len(args.train_loss),
                    opts=dict(title="train_loss", xlabel="batch", ylabel="train_loss"), name="Seg_loss")
            vis.plot(win="train_loss", y=args.lambda_KD * KD_Loss.item(), con_point=len(args.train_loss),
                    opts=dict(title="train_loss", xlabel="batch", ylabel="train_loss"), name="KD_Loss")
            if args.epoch_S > args.switch_epoch and pcl_active:
                vis.plot(win="train_loss", y=args.lambda_proto * PCL_loss.item(), con_point=len(args.train_loss),
                        opts=dict(title="train_loss", xlabel="batch", ylabel="train_loss"), name="PCL_Loss")

        step += 1 

    print("=============================================================================================================== ")
    return step