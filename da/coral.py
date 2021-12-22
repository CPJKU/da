import torch
import math

# implementation adapted from:
# https://github.com/SSARCandy/DeepCORAL/blob/200f7c8626236b6d04cab048670b85f14deaa17f/models.py#L13

def coral_loss(embeds, domain_labels, da_info):
    loss = torch.tensor(0., device=domain_labels.device)
    # split into source and target samples
    unique_dl = domain_labels.unique()
    # if a batch with samples of only one domain is encountered - return 0 as loss
    if len(unique_dl) == 1:
        return loss
    src_mask = domain_labels == unique_dl[0]
    tgt_mask = domain_labels == unique_dl[1]

    # for all embeds calculate the coral loss and sum them together
    for embed in embeds:
        src_embed = embed[src_mask]
        tgt_embed = embed[tgt_mask]
        embed_loss = coral(src_embed, tgt_embed)
        da_info['embed_losses'].append(embed_loss.detach().cpu())
        loss += embed_loss
    return loss


def comp_cov(x):
    xm = torch.mean(x, dim=0, keepdim=True) - x
    return xm.T @ xm


def coral(src_embed, tgt_embed):
    src_batch_size = src_embed.size(0)
    tgt_batch_size = tgt_embed.size(0)
    batch_size = src_batch_size + tgt_batch_size
    src_repeats = math.ceil(batch_size / src_batch_size)
    tgt_repeats = math.ceil(batch_size / tgt_batch_size)
    # handle case when source and target are not of same size
    src_embed_rep = torch.cat([src_embed] * src_repeats, dim=0)[:batch_size]
    tgt_embed_rep = torch.cat([tgt_embed] * tgt_repeats, dim=0)[:batch_size]
    d = src_embed_rep.size()[1]
    src_coral = comp_cov(src_embed_rep)
    tgt_coral = comp_cov(tgt_embed_rep)

    loss = torch.mean(torch.mul((src_coral - tgt_coral), (src_coral - tgt_coral)))
    loss = loss / (4 * d * d)
    return loss
