import torch
import torch.nn as nn
import torch.nn.functional as F

class CCLoss(nn.Module):
    """"Modified from CDDFuse: https://github.com/Zhaozixiang1228/MMIF-CDDFuse."""
    def __init__(self):
        super(CCLoss, self).__init__()
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, i, b):
        i = i.reshape(i.shape[0], -1)
        i = i - i.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        cc = torch.sum(i * b, dim=-1) / (self.eps + torch.sqrt(torch.sum(i ** 2, dim=-1)) * torch.sqrt(torch.sum(b**2, dim=-1)))
        cc = torch.clamp(cc, -1., 1.)
        return cc.mean()

class ComponentWiseSupConLoss(nn.Module):
    """Improved from Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07,
                 base_temperature=0.07, epsilon=1e-9,
                 disen_term_alpha=1.0, # Weight for the disentanglement term
                 disen_temperature=None):
        super(ComponentWiseSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.epsilon = epsilon
        self.disen_term_alpha = disen_term_alpha
        self.disen_temperature = disen_temperature if disen_temperature is not None else temperature
        self.contrast_mode = 'all'

    def _average_views(self, z1, z2):
        # Helper to average two views of features z1, z2 [bsz, D] -> [bsz, D]
        return (z1 + z2) / 2.0

    def forward(self, z_intrinsic, z_intrinsic_aug, z_bias, z_bias_aug, labels=None):
        """
        Args:
            z_intrinsic: features of shape [bsz, D].
            z_intrinsic_aug: augmented intrinsic features of shape [bsz, D].
            z_bias: features of shape [bsz, D].
            z_bias_aug: augmented bias features of shape [bsz, D].
            labels: ground truth of shape [bsz] for supervised contrast.
        Returns:
            A loss scalar.
        """
        device = z_intrinsic.device
        bsz = z_intrinsic.shape[0]

        z_intrinsic_flat = z_intrinsic.view(bsz, -1)
        z_intrinsic_aug_flat = z_intrinsic_aug.view(bsz, -1)
        z_bias_flat = z_bias.view(bsz, -1)
        z_bias_aug_flat = z_bias_aug.view(bsz, -1)

        # z_c-i'
        z_prime_intrinsic = self._average_views(z_intrinsic_flat, z_intrinsic_aug_flat)
        # z_c-a'
        z_prime_bias = self._average_views(z_bias_flat, z_bias_aug_flat)

        z_prime_intrinsic_norm = F.normalize(z_prime_intrinsic, dim=1)
        z_prime_bias_norm = F.normalize(z_prime_bias, dim=1)

        # z_c-i for sample k: concat(z_intrinsic[k], z_bias[k])
        # z_c-p for sample k: concat(z_intrinsic_aug[k], z_bias_aug[k])
        view1 = torch.cat([z_intrinsic_flat, z_bias_flat], dim=1) 
        view2 = torch.cat([z_intrinsic_aug_flat, z_bias_aug_flat], dim=1) 

        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)

        features_supcon = torch.stack([view1, view2], dim=1)

        if len(features_supcon.shape) < 3:
            raise ValueError('`features_supcon` needs to be [bsz, n_views, ...]')

        if labels is None:
            label_mask = torch.eye(bsz, dtype=torch.float32).to(device)
        else:
            labels_view = labels.contiguous().view(-1, 1)
            if labels_view.shape[0] != bsz:
                raise ValueError('Num of labels does not match num of features')
            label_mask = torch.eq(labels_view, labels_view.T).float().to(device)

        num_views = features_supcon.shape[1]

        contrast_feature = torch.cat(torch.unbind(features_supcon, dim=1), dim=0)

        anchor_feature = contrast_feature

        key_feature = contrast_feature

        # Compute logits for the main contrastive part: (anchor · key / τ)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, key_feature.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Part 1: sum_{a ∈ A(i)} exp(z_anchor · z_key_neg / τ)
        logits_mask_for_sum_denominator = torch.ones_like(logits).to(device)
        logits_mask_for_sum_denominator.scatter_(
            1,
            torch.arange(bsz * num_views).view(-1, 1).to(device),
            0
        )
        exp_logits_main_negatives = torch.exp(logits) * logits_mask_for_sum_denominator
        sum_exp_logits_main_negatives = exp_logits_main_negatives.sum(1, keepdim=True)

        # Part 2: alpha * exp(z_c-i' · z_c-a' / τ_disen)
        dot_product_i_prime_a_prime = (z_prime_intrinsic_norm * z_prime_bias_norm).sum(dim=1, keepdim=True)
        logits_i_prime_a_prime = dot_product_i_prime_a_prime / self.disen_temperature
        single_exp_disen_term_per_sample = torch.exp(logits_i_prime_a_prime)

        single_disen_exp_term_expanded = torch.tile(single_exp_disen_term_per_sample, (num_views, 1))
        
        denominator_sum_exp = sum_exp_logits_main_negatives + \
                              (self.disen_term_alpha * single_disen_exp_term_expanded)

        log_prob = logits - torch.log(denominator_sum_exp + self.epsilon)

        positives_mask_expanded = label_mask.repeat(num_views, num_views)

        final_supcon_mask_for_loss = positives_mask_expanded * logits_mask_for_sum_denominator

        num_positive_pairs = final_supcon_mask_for_loss.sum(1)
        num_positive_pairs_safe = torch.where(num_positive_pairs < 1e-6,
                                              torch.ones_like(num_positive_pairs),
                                              num_positive_pairs)
        
        mean_log_prob_pos = (final_supcon_mask_for_loss * log_prob).sum(1) / num_positive_pairs_safe

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
    