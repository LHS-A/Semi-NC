import torch
import torch.nn.functional as F
from torch import nn
import random

class ProAlignLoss(nn.Module):
    def __init__(self, tau=0.07, M=64, topPpos=3, device='cuda', 
                 neg_lower_bound=3, neg_upper_bound=9, num_negatives=7):
        super().__init__()
        self.tau = tau
        self.M = M          # queries per class (per batch)
        self.topPpos = topPpos
        self.device = device
        self.neg_lower_bound = neg_lower_bound  # r_l
        self.neg_upper_bound = neg_upper_bound  # r_h  
        self.num_negatives = num_negatives      # N

    def sample_queries(self, feature_map, gt_mask=None, logits=None, M=64):
        """
        feature_map: C x H x W (query features)
        gt_mask: Bool mask for labeled queries (H x W), True = foreground
        logits: when unlabeled, network output logits used to select top M features
        M: int, maximum number of sampled pixels

        return: Tensor K x C of sampled query vectors 
        """
        C, H, W = feature_map.shape
        feat_flat = feature_map.view(C, -1).permute(1, 0)  # N x C
        N = H * W

        if gt_mask is not None:
            # labeled: select foreground pixels
            idxs = torch.nonzero(gt_mask.view(-1), as_tuple=False).squeeze(1)
        else:
            # unlabeled: pick top M pixels with highest feature magnitude
            if logits is None:
                raise ValueError("logits must be provided for unlabeled queries")
            # convert logits to single-channel score
            prob = torch.softmax(logits, dim=0)  # num_classes x H x W
            score_map = prob.max(dim=0)[0]  # H x W, max probability per pixel
            idxs = torch.topk(score_map.view(-1), k=min(M, N))[1]

        if idxs.numel() == 0:
            return None

        # randomly subsample to M
        if idxs.numel() > M:
            idxs = idxs[torch.randperm(len(idxs))[:M]]

        return feat_flat[idxs, :]  # K x C

    def sample_informative_negatives(self, q_pixels, class_probs, neg_pools, target_class):
        """
        Sample informative negatives following Section 2.5 strategy
        Avoid overly easy or overly hard negative samples
        
        Args:
            q_pixels: K x C query pixel features
            class_probs: K x num_classes class probability distribution for each query pixel
            neg_pools: negative prototype pool dictionary {class_id: PrototypePool}
            target_class: target class (positive class)
            
        Returns:
            neg_protos: K x N x C negative prototypes
        """
        K, C = q_pixels.shape
        device = q_pixels.device
        
        # store all negative prototypes
        all_neg_protos = []
        
        for k in range(K):
            # get class probability for k-th query pixel
            prob_k = class_probs[k]  # num_classes
            
            # sort class indices by probability descending
            sorted_classes = torch.argsort(prob_k, descending=True)
            
            # filter eligible negative classes
            valid_neg_classes = []
            for class_idx, class_id in enumerate(sorted_classes):
                # skip positive class
                if class_id == target_class:
                    continue
                    
                # check if within specified rank range
                if (class_idx >= self.neg_lower_bound and 
                    class_idx < self.neg_upper_bound and
                    class_id in neg_pools and 
                    len(neg_pools[class_id].pool) > 0):
                    valid_neg_classes.append(class_id)
            
            # sample prototypes from eligible negative classes
            neg_protos_k = []
            for neg_class in valid_neg_classes:
                if len(neg_protos_k) >= self.num_negatives:
                    break
                    
                # sample from class prototype pool
                pool = neg_pools[neg_class]
                if len(pool.pool) == 0:
                    continue
                
                # get all prototypes
                all_protos = pool.get_all()  # C x total_K
                if all_protos is None:
                    continue
                    
                # transpose to total_K x C
                pool_feat = all_protos.t().to(device)
                pool_feat = F.normalize(pool_feat, dim=1)
                
                # compute similarity with current query pixel
                q_pixel_norm = F.normalize(q_pixels[k:k+1], dim=1)  # 1 x C
                similarities = q_pixel_norm @ pool_feat.t()  # 1 x total_K
                
                # select medium similarity prototypes (avoid too easy or too hard negatives)
                total_protos = pool_feat.size(0)
                if total_protos > 0:
                    # sort similarities
                    sorted_sim, sorted_idx = torch.sort(similarities.squeeze(), descending=True)
                    
                    # pick middle range
                    start_idx = max(0, total_protos // 4)
                    end_idx = min(total_protos, 3 * total_protos // 4)
                    
                    if end_idx > start_idx:
                        # randomly select negatives from middle range
                        mid_range = sorted_idx[start_idx:end_idx]
                        if len(mid_range) > 0:
                            selected_idx = mid_range[torch.randperm(len(mid_range))[:1]]
                            neg_proto = pool_feat[selected_idx]  # 1 x C
                            neg_protos_k.append(neg_proto)
            
            # supplement with random samples if insufficient informative negatives
            while len(neg_protos_k) < self.num_negatives:
                # randomly pick a negative class
                available_classes = [c for c in neg_pools.keys() if c != target_class and len(neg_pools[c].pool) > 0]
                if not available_classes:
                    break
                    
                random_class = random.choice(available_classes)
                pool = neg_pools[random_class]
                all_protos = pool.get_all()
                if all_protos is not None:
                    pool_feat = all_protos.t().to(device)
                    if pool_feat.size(0) > 0:
                        random_idx = torch.randint(0, pool_feat.size(0), (1,))
                        neg_proto = pool_feat[random_idx]  # 1 x C
                        neg_protos_k.append(neg_proto)
            
            if neg_protos_k:
                # stack all negatives for current query pixel: N_k x C
                neg_protos_k = torch.cat(neg_protos_k, dim=0)
                # randomly select if exceeds required number
                if neg_protos_k.size(0) > self.num_negatives:
                    select_idx = torch.randperm(neg_protos_k.size(0))[:self.num_negatives]
                    neg_protos_k = neg_protos_k[select_idx]
            else:
                # zero pad if no negatives
                neg_protos_k = torch.zeros(self.num_negatives, C, device=device)
            
            all_neg_protos.append(neg_protos_k.unsqueeze(0))
        
        if all_neg_protos:
            # stack negatives for all query pixels: K x N x C
            return torch.cat(all_neg_protos, dim=0)
        else:
            return torch.zeros(K, self.num_negatives, C, device=device)

    def forward(self, query_feat, labels_map, logits, pool_nerve=None, pool_cell=None, pool_bg=None, fenzhi='nerve'):
        """
        query_feat: B x C x H x W
        labels_map: B x H x W
        logits: B x num_cls x H x W
        pool_nerve / pool_cell / pool_bg: PrototypePool
        fenzhi: 'nerve' or 'cell'
        """
        bs, C, H, W = query_feat.shape
        device = self.device
        tau = self.tau
        loss_total = 0.0
        valid_count = 0

        for b in range(bs):
            feat_b = query_feat[b]  # C x H x W
            logits_b = logits[b]    # num_cls x H_logits x W_logits

            # 1. select query pixels
            if labels_map is not None:
                mask_fg = labels_map[b] > 0  # foreground pixels
            else:
                # for unlabeled data, resize logits to feature map size
                logits_b_resized = F.interpolate(
                    logits_b.unsqueeze(0), 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)  # num_cls x H x W
                
                prob = torch.softmax(logits_b_resized, dim=0)
                score_map = prob.max(dim=0)[0]  # H x W
                scores_flat = score_map.view(-1)
                top_vals, top_idx = torch.topk(scores_flat, min(self.M, scores_flat.numel()))
                mask_fg = torch.zeros(H*W, dtype=torch.bool, device=device)
                mask_fg[top_idx] = True
                mask_fg = mask_fg.view(H, W)

            # resize logits to feature map size for probability computation
            logits_b_resized = F.interpolate(
                logits_b.unsqueeze(0), 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)  # num_cls x H x W

            q_pixels = self.sample_queries(feat_b, gt_mask=mask_fg, logits=logits_b_resized, M=self.M)  # K x C
            
            if q_pixels is None:
                continue

            # 2. set positive and negative pools based on fenzhi
            if fenzhi == 'nerve':
                pos_pool = pool_nerve
                target_class = 1  # nerve class
                neg_pools = {2: pool_cell, 0: pool_bg}  # cell and background as negatives
            elif fenzhi == 'cell':
                pos_pool = pool_cell  
                target_class = 2  # cell class
                neg_pools = {1: pool_nerve, 0: pool_bg}  # nerve and background as negatives
            else:
                raise ValueError("fenzhi must be 'nerve' or 'cell'")

            # 3. get class probability distribution for query pixels
            # compute class probabilities using resized logits
            class_probs = F.softmax(logits_b_resized, dim=0)  # num_classes x H x W
            class_probs_flat = class_probs.view(class_probs.size(0), -1).t()  # N x num_classes
            
            # get corresponding class probabilities for each query pixel
            if labels_map is not None:
                mask_flat = labels_map[b].view(-1) > 0
                if mask_flat.sum() > 0:
                    query_probs = class_probs_flat[mask_flat]
                    # ensure query_probs matches q_pixels count
                    if query_probs.size(0) > q_pixels.size(0):
                        query_probs = query_probs[:q_pixels.size(0)]
                    elif query_probs.size(0) < q_pixels.size(0):
                        # zero pad if insufficient foreground pixels
                        padding = torch.zeros(q_pixels.size(0) - query_probs.size(0), 
                                            query_probs.size(1), device=device)
                        query_probs = torch.cat([query_probs, padding], dim=0)
                else:
                    # zero pad if no foreground pixels
                    query_probs = torch.zeros(q_pixels.size(0), class_probs_flat.size(1), device=device)
            else:
                # for unlabeled data, use probabilities for all pixels
                query_probs = class_probs_flat[:q_pixels.size(0)]  # K x num_classes

            # 4. enhanced negative sampling
            neg_protos = self.sample_informative_negatives(
                q_pixels, query_probs, neg_pools, target_class
            )  # K x N x C
            
            # 5. select positive prototypes
            pos_proto = pos_pool.sample_prototypes_for_queries(q_pixels, topk=1)
            if pos_proto is None:
                continue
            pos_proto = F.normalize(pos_proto.to(device), dim=1)

            # skip batch if no valid negatives
            if neg_protos is None or (neg_protos.abs().max() < 1e-8 and neg_protos.numel() > 0):
                continue

            # normalize negative prototypes
            neg_protos = F.normalize(neg_protos, dim=2)  # K x N x C

            # === 6. compute contrastive loss ===
            # expand positive prototype dimension: K x 1 x C
            pos_proto = pos_proto.unsqueeze(1)  # K x 1 x C

            # concatenate positive and negative prototypes: K x (1+N) x C
            protos_all = torch.cat([pos_proto, neg_protos], dim=1)  # K x (1+N) x C

            # normalize query pixels
            q_norm = F.normalize(q_pixels, dim=1)  # K x C

            # compute similarities: K x (1+N)
            # use batch matrix multiplication
            sim = torch.sum(q_norm.unsqueeze(1) * protos_all, dim=2)  # K x (1+N)

            # InfoNCE loss
            logits_sim = sim / tau

            # positive prototype is in column 0
            labels = torch.zeros(q_pixels.size(0), dtype=torch.long, device=device)

            loss_pixel = F.cross_entropy(logits_sim, labels)

            loss_total += loss_pixel
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return loss_total / valid_count