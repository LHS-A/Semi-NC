# model/prototype_pool.py
import torch
import torch.nn.functional as F
import time

class PrototypePool:
    def __init__(self, device='cuda'):
        self.device = device
        self.pool = []  

    def add_prototypes_safe(self, protos):
        if protos is None:
            return

        if not isinstance(protos, torch.Tensor):
            raise TypeError("protos must be a torch.Tensor")

        protos = protos.detach().cpu()
    
        if torch.isnan(protos).any() or torch.isinf(protos).any():  
            return
        
        if protos.abs().max() < 1e-8:
            return

        if protos.dim() == 1:
            protos = protos.unsqueeze(1)  # (C,) -> (C, 1)
        elif protos.dim() == 3:
            if protos.shape[0] == 1:  # (1, C, K)
                protos = protos.squeeze(0)  # -> (C, K)
            else:
                return
        elif protos.dim() != 2:
            raise ValueError(f"2-D input {protos.dim()}-D, shape: {protos.shape}")

        self.pool.append(protos)

    def sample_prototypes_for_queries(self, q_pixels: torch.Tensor, topk=1):
        """
        q_pixels: K x C_query
        return: K x C_query selected prototypes
        """
        if len(self.pool) == 0:
            return None
     
        q_pixels = q_pixels.view(q_pixels.size(0), -1)  # K x C_query
        q_pixels = F.normalize(q_pixels, dim=1)         # K x C_query

        device = q_pixels.device
        
        all_protos = self.get_all()  
        if all_protos is None:
            return None
            
        pool_feat_flat = all_protos.t().to(device)  # total_K x C
        pool_feat_flat = F.normalize(pool_feat_flat, dim=1)
        
        if q_pixels.size(1) != pool_feat_flat.size(1):
            min_dim = min(q_pixels.size(1), pool_feat_flat.size(1))
            q_pixels = q_pixels[:, :min_dim]
            pool_feat_flat = pool_feat_flat[:, :min_dim]

        # cosine similarity
        sim = q_pixels @ pool_feat_flat.t()  # K x total_K

        topk_vals, topk_idx = torch.topk(sim, k=topk, dim=1)
        selected_protos = pool_feat_flat[topk_idx[:, 0], :]

        return selected_protos

    def get_all(self):
        if len(self.pool) == 0:
            return None
            
        first_dim = self.pool[0].shape[0]
        for i, proto in enumerate(self.pool):
            if proto.shape[0] != first_dim:

                if proto.shape[0] > first_dim:
                    self.pool[i] = proto[:first_dim, :]
                else:
                 
                    padding = torch.zeros(first_dim - proto.shape[0], proto.shape[1])
                    self.pool[i] = torch.cat([proto, padding], dim=0)
        
        result = torch.cat(self.pool, dim=1)  # C x total_K
        return result

    def merge_to_capacity(self, capacity=256):
        all_protos = self.get_all()  # C x total_K
        if all_protos is None:
            return

        total_protos = all_protos.shape[1]

        if total_protos <= capacity:
            return

        while all_protos.shape[1] > capacity:
            normed = F.normalize(all_protos, dim=0)  # C x total_K
            sim_matrix = torch.mm(normed.t(), normed)  # total_K x total_K
            sim_matrix.fill_diagonal_(-1.0)

            max_sim, max_idx = torch.max(sim_matrix.view(-1), dim=0)
            i = max_idx // sim_matrix.size(1)
            j = max_idx % sim_matrix.size(1)

            merged_proto = (all_protos[:, i] + all_protos[:, j]) / 2.0

            all_protos[:, i] = merged_proto
            mask = torch.ones(all_protos.shape[1], dtype=torch.bool)
            mask[j] = False
            all_protos = all_protos[:, mask]

        self.pool = [all_protos]

    def size(self):
        return len(self.pool)

    def health_check(self):
        if len(self.pool) == 0:
            return False, "empty"
        
        all_protos = self.get_all()
        if all_protos is None:
            return False, "None"
            
        if torch.isnan(all_protos).any() or torch.isinf(all_protos).any():
            return False, "NaN or Inf"
            
        if all_protos.abs().max() < 1e-8:
            return False, "too small"
            
        return True, f"health. include {len(self.pool)} prototypes, total {all_protos.shape[1]} feature."
    
# helpers in sacnet or new file model/sac_helpers.py

def init_farthest_points(masked_feat, K):
    """
    masked_feat: (D, N) tensor (already includes xy if desired)
    returns K indices into N using farthest point selection (deterministic)
    """
    D, N = masked_feat.shape
    if N == 0:
        return torch.LongTensor([]).to(masked_feat.device)
    # pick first as max-norm
    norms = torch.norm(masked_feat, dim=0)
    seeds = [int(torch.argmax(norms).item())]
    if K == 1:
        return torch.LongTensor(seeds)
    # iterative farthest selection
    dists = torch.full((N,), 1e9, device=masked_feat.device)
    for _ in range(1, K):
        last = masked_feat[:, seeds[-1]].unsqueeze(1)  # D x 1
        curd = torch.sum((masked_feat - last) ** 2, dim=0)
        dists = torch.minimum(dists, curd)
        seeds.append(int(torch.argmax(dists).item()))
    return torch.LongTensor(seeds).to(masked_feat.device)

def build_sac_prototypes(supp_feat, supp_mask, sp_center_iter_fn, num_proto=3, n_iter=10):

    C, H, W = supp_feat.shape

    mask_flat = (supp_mask.squeeze(0).view(-1) > 0.5)
    
    if mask_flat.sum() == 0:
        return None

    feat_flat = supp_feat.view(C, -1)  # C x N
    ys = torch.arange(H, device=supp_feat.device).view(H, 1).repeat(1, W).view(-1)
    xs = torch.arange(W, device=supp_feat.device).repeat(H)
    coords = torch.stack([ys, xs], dim=0).float()  # 2 x N
    
    feat_xy = torch.cat([feat_flat, coords], dim=0)  # (C+2) x N
    feat_xy_masked = feat_xy[:, mask_flat]  # (C+2) x num_roi

    num_roi = feat_xy_masked.shape[1]
    K = min(num_proto, num_roi)
    if K == 0:
        return None

    seeds_idx = init_farthest_points(feat_xy_masked, K)
    sp_init_center = feat_xy_masked[:, seeds_idx]  # (C+2) x K

    sp_center = sp_center_iter_fn(supp_feat, supp_mask, sp_init_center, n_iter=n_iter)
    
    if sp_center.size(0) != C:
        if sp_center.size(0) > C:
            sp_center = sp_center[:C, :]
    
    return sp_center  # C x K

def compute_pseudo_labels_and_confidence(logits):
    """
    logits: (bs, num_classes, h, w)
    returns pseudo_label (bs,h,w) and confidence (bs,h,w)
    """
    prob = F.softmax(logits, dim=1)
    conf, pseudo = torch.max(prob, dim=1)  # conf and pseudo are same device as logits
    return pseudo, conf


def gamma_schedule(step, gamma_start=0.99, gamma_end=0.6, decay_rate=0.001):

    gamma_t = gamma_start - decay_rate * step
    return max(gamma_end, gamma_t)


def build_prototypes_unlabeled(supp_feat, logits, sp_center_iter_fn,
                               step=0, num_proto_per_class=3, use_sac_on_high_conf=True):
    gamma_t = 0.99 - 0.001 * step
    gamma_t = max(0.6, gamma_t) 

    prob = F.softmax(logits, dim=1)
    conf, pseudo = torch.max(prob, dim=1)  # (H, W), (H, W)
    pseudo = pseudo[0] 
    conf_map = conf[0]
    num_classes = logits.shape[1]

    protos = []

    bg_mask = (pseudo == 0).unsqueeze(0).float()
    if bg_mask.sum() > 0:
        bg_proto = Weighted_GAP(supp_feat.unsqueeze(0), bg_mask)
        bg_proto = bg_proto.squeeze(0).squeeze(-1).squeeze(-1)  # C
        protos.append((bg_proto.cpu(), 0, float(conf_map[pseudo == 0].mean().item())))

    for cls in range(1, num_classes):
        mask_cls = (pseudo == cls)
        if mask_cls.sum() == 0:
            continue

        high_mask = mask_cls & (conf_map > gamma_t)
        low_mask = mask_cls & (conf_map <= gamma_t)

        if use_sac_on_high_conf and high_mask.sum() >= 3:
            mask_tensor = high_mask.float().unsqueeze(0)
            sp = build_sac_prototypes(supp_feat, mask_tensor, sp_center_iter_fn,
                                    num_proto=num_proto_per_class, n_iter=10)
            if sp is not None:
                for k in range(sp.shape[1]):
                    vec = sp[:, k].cpu()
                    protos.append((vec, cls, float(conf_map[high_mask].mean().item())))

        if low_mask.sum() > 0:
            mask_tensor = low_mask.float().unsqueeze(0)
            proto = Weighted_GAP(supp_feat.unsqueeze(0), mask_tensor)
            proto = proto.squeeze(0).squeeze(-1).squeeze(-1).cpu()
            protos.append((proto, cls, float(conf_map[low_mask].mean().item())))

    return protos

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def sp_center_iter(supp_feat, supp_mask, sp_init_center, n_iter):
    '''
    :param supp_feat: A Tensor of support feature, (C, H, W)
    :param supp_mask: A Tensor of support mask, (1, H, W)
    :param sp_init_center: A Tensor of initial sp center, (C + xy, num_sp)
    :param n_iter: The number of iterations
    :return: sp_center: The centroid of superpixels (prototypes) - C x num_sp
    '''
    c_xy, num_sp = sp_init_center.size()
    C, h, w = supp_feat.size()
    
    h_coords = torch.arange(h, device=supp_feat.device).view(h, 1).contiguous().repeat(1, w).unsqueeze(0).float()
    w_coords = torch.arange(w, device=supp_feat.device).repeat(h, 1).unsqueeze(0).float()
    supp_feat = torch.cat([supp_feat, h_coords, w_coords], 0)  # (C+2) x H x W
    supp_feat_roi = supp_feat[:, (supp_mask == 1).squeeze()]  # (C+2) x num_roi

    num_roi = supp_feat_roi.size(1)
    supp_feat_roi_rep = supp_feat_roi.unsqueeze(-1).repeat(1, 1, num_sp)  # (C+2) x num_roi x num_sp
    sp_center = torch.zeros_like(sp_init_center).to(supp_feat.device)  # (C+2) x num_sp

    for i in range(n_iter):
        # Compute association between each pixel in RoI and superpixel
        if i == 0:
            sp_center_rep = sp_init_center.unsqueeze(1).repeat(1, num_roi, 1)  # (C+2) x num_roi x num_sp
        else:
            sp_center_rep = sp_center.unsqueeze(1).repeat(1, num_roi, 1)
        
        assert supp_feat_roi_rep.shape == sp_center_rep.shape
        
        dist = torch.pow(supp_feat_roi_rep - sp_center_rep, 2.0)
        feat_dist = dist[:-2, :, :].sum(0) 
        spat_dist = dist[-2:, :, :].sum(0) 
        total_dist = torch.pow(feat_dist + spat_dist / 100, 0.5)
        p2sp_assoc = torch.neg(total_dist).exp()
        p2sp_assoc = p2sp_assoc / (p2sp_assoc.sum(0, keepdim=True))  # num_roi x num_sp

        sp_center = supp_feat_roi_rep * p2sp_assoc.unsqueeze(0)  # (C+2) x num_roi x num_sp
        sp_center = sp_center.sum(1)  # (C+2) x num_sp

    result = sp_center[:-2, :]  # C x num_sp
    
    return result