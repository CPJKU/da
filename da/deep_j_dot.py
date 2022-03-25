import torch
import torch.nn.functional as F
import ot
from scipy.spatial.distance import cdist

# pytorch adaptation of original implementation by authors of paper:
# https://github.com/bbdamodaran/deepJDOT


class DeepJDot:
    def __init__(self, num_classes, alpha=10.0):
        self.num_classes = num_classes
        self.alpha = alpha

    def optimize_gamma(self, gs_batch, gt_batch, ys, ft_pred):
        C0 = cdist(gs_batch, gt_batch, metric='sqeuclidean')
        C1 = cdist(ys, ft_pred, metric='sqeuclidean')  # official repo uses squared euclidean distance here
        C = self.alpha * C0 + C1
        # distance maxtrix C is cost matrix
        # using optimal transport theory get the coupling matrix
        return ot.emd(ot.unif(gs_batch.shape[0]), ot.unif(gt_batch.shape[0]), C)

    def classifier_cat_loss(self, gamma, y_true_src, y_pred_tgt):
        # samples are not only aligned based on feature distance, but also on label distance
        y_pred_tgt_log = torch.log(y_pred_tgt)
        loss = -y_true_src.float() @ y_pred_tgt_log.T
        return torch.sum(gamma * loss)

    def align_loss(self, gamma, gs_batch, gt_batch):
        # squared euclidean distance
        gdist = torch.cdist(gs_batch, gt_batch)**2
        return self.alpha * torch.sum(gamma * gdist)

    def jdot_loss(self, embeds, domain_labels, labels, predictions, da_info):
        loss = torch.tensor(0., device=domain_labels.device)
        # split into source and target samples
        unique_dl = domain_labels.unique()
        # if a batch with samples of only one domain is encountered - return 0 as loss
        if len(unique_dl) == 1:
            return loss
        da_info['jdot_cat_losses'] = []
        da_info['jdot_align_losses'] = []
        src_mask = domain_labels == unique_dl[0]
        tgt_mask = domain_labels == unique_dl[1]

        # normalize the predictions
        predictions = F.softmax(predictions, dim=1)

        labels = F.one_hot(labels, num_classes=self.num_classes)
        y_true_src = labels[src_mask]
        y_pred_tgt = predictions[tgt_mask]

        for embed in embeds:
            gs_batch = embed[src_mask]
            gt_batch = embed[tgt_mask]
            if torch.mean(torch.abs(gs_batch) + torch.abs(gt_batch)) <= 1e-7:
                print("Warning: feature representations tend towards zero. "
                      "Consider decreasing 'da_lambda' or using lambda schedule.")
            embed_loss, embed_cat_loss, embed_align_loss = self._jdot_loss(gs_batch, gt_batch, y_true_src, y_pred_tgt)
            da_info['embed_losses'].append(embed_loss.detach().cpu())
            da_info['jdot_cat_losses'].append(embed_cat_loss.detach().cpu())
            da_info['jdot_align_losses'].append(embed_align_loss.detach().cpu())
            loss += embed_loss
        return loss

    def _jdot_loss(self, gs_batch, gt_batch, y_true_src, y_pred_tgt):
        # calculate coupling matrix (determines which samples are aligned to each other, based on features and labels)
        gamma = torch.from_numpy(self.optimize_gamma(gs_batch.detach().cpu().numpy(), gt_batch.detach().cpu().numpy(),
                                                     y_true_src.cpu().numpy(), y_pred_tgt.detach().cpu().numpy())
                                 ).to(device=y_true_src.device)

        cat_loss = self.classifier_cat_loss(gamma, y_true_src, y_pred_tgt)
        align_loss = self.align_loss(gamma, gs_batch, gt_batch)
        loss = cat_loss + align_loss
        return loss, cat_loss, align_loss
