import torch
import torch.nn.functional as F
import ot
from scipy.spatial.distance import cdist

# pytorch adaptation of original implementation by authors of paper:
# https://github.com/bbdamodaran/deepJDOT

# squared L2 distance between samples in two matrices
def L2_dist(gs_batch, gt_batch):
    gs_sq = torch.sum(gs_batch ** 2, dim=1).view(-1, 1)
    gt_sq = torch.sum(gt_batch ** 2, dim=1).view(1, -1)
    dist = gs_sq + gt_sq
    dist -= 2.0 * (gs_batch @ gt_batch.T)
    return dist


class DeepJDot:
    def __init__(self, num_classes, alpha=10.0):
        self.num_classes = num_classes
        self.alpha = alpha

    def optimize_gamma(self, gs_batch, gt_batch, ys, ft_pred):
        C0 = cdist(gs_batch, gt_batch, metric='sqeuclidean')
        C1 = cdist(ys, ft_pred, metric='sqeuclidean')
        C = self.alpha * C0 + C1
        # distance maxtrix C is cost matrix
        # using optimal transport theory get the coupling matrix
        return ot.emd(ot.unif(gs_batch.shape[0]), ot.unif(gt_batch.shape[0]), C)

    def classifier_cat_loss(self, gamma, y_true_src, y_pred_tgt):
        # samples are not only aligned based on feature distance, but also on sample distance
        y_pred_tgt_log = torch.log(y_pred_tgt)
        loss = -y_true_src.float() @ y_pred_tgt_log.T
        return torch.sum(gamma * loss)

    def align_loss(self, gamma, gs_batch, gt_batch):
        gdist = L2_dist(gs_batch, gt_batch)
        return self.alpha * torch.sum(gamma * gdist)

    def jdot_loss(self, embeds, domain_labels, labels, predictions, da_info):
        assert len(embeds) == 1
        embeds = embeds[0]
        loss = torch.tensor(0., device=domain_labels.device)
        # split into source and target samples
        unique_dl = domain_labels.unique()
        # if a batch with samples of only one domain is encountered - return 0 as loss
        if len(unique_dl) == 1:
            return loss
        src_mask = domain_labels == unique_dl[0]
        tgt_mask = domain_labels == unique_dl[1]

        # normalize the predictions
        predictions = F.softmax(predictions, dim=1)

        labels = F.one_hot(labels, num_classes=self.num_classes)
        y_true_src = labels[src_mask]
        y_pred_tgt = predictions[tgt_mask]
        gs_batch = embeds[src_mask]
        gt_batch = embeds[tgt_mask]

        # calculate coupling matrix (determines which samples are aligned to each other, based on features and labels)
        gamma = torch.from_numpy(self.optimize_gamma(gs_batch.detach().cpu().numpy(), gt_batch.detach().cpu().numpy(),
                                                     y_true_src.cpu().numpy(), y_pred_tgt.detach().cpu().numpy())
                                 ).to(device=domain_labels.device)

        cat_loss = self.classifier_cat_loss(gamma, y_true_src, y_pred_tgt)
        align_loss = self.align_loss(gamma, gs_batch, gt_batch)
        loss = cat_loss + align_loss
        da_info['embed_losses'].append(loss.detach().cpu())
        da_info['jdot_cat_loss'] = cat_loss.detach().cpu()
        da_info['jdot_align_loss'] = align_loss.detach().cpu()
        return loss
