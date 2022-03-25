import torch
from torch.autograd import grad
import numpy as np
import math


def wdgrl_update(embeds, domain_labels, optimizer, critic_iter, gp_da_lambda, da_net, da_info):
    # critic tries to maximize Wasserstein distance
    total_gp_loss = torch.tensor(0., device=domain_labels.device)
    unique_dl = domain_labels.unique(sorted=True)
    # if a batch with samples of only one domain is encountered - return 0 as loss
    if len(unique_dl) == 1:
        return torch.tensor(0., device=domain_labels.device)
    src_mask = domain_labels == unique_dl[0]
    tgt_mask = domain_labels == unique_dl[1]

    # iterate through all embeddings and the according da_nets
    for i, embed in enumerate(embeds):
        # multiple iterations of training the critic
        for _ in range(critic_iter):
            # feature extractor is not updated here - detach src and target embeddings
            src_embed = embed[src_mask].detach()
            tgt_embed = embed[tgt_mask].detach()
            if torch.mean(torch.abs(src_embed) + torch.abs(tgt_embed)) <= 1e-7:
                print("Warning: feature representations tend towards zero. "
                      "Consider decreasing 'da_lambda' or using lambda schedule.")
            src_critic = da_net.nets[i](src_embed)
            tgt_critic = da_net.nets[i](tgt_embed)
            # estimation of wasserstein distance
            wd = src_critic.mean() - tgt_critic.mean()
            gp_loss = gradient_penalty(da_net.nets[i], src_embed, tgt_embed)
            total_gp_loss = total_gp_loss + gp_loss.detach()
            loss = -wd + gp_da_lambda * gp_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    da_info['gp_loss'] = total_gp_loss


def wdgrl_loss(embeds, domain_labels, da_net, da_info):
    wd_sim_list = []
    wd_meas_list = []
    loss = torch.tensor(0., device=domain_labels.device)
    unique_dl = domain_labels.unique()
    # if a batch with samples of only one domain is encountered - return 0 as loss
    if len(unique_dl) == 1:
        return loss
    src_mask = domain_labels == unique_dl[0]
    tgt_mask = domain_labels == unique_dl[1]

    # calculate wasserstein distance for all embeddings
    # critic is not updated by this function
    da_preds = da_net.forward(embeds)
    for pred in da_preds:
        src_pred = pred[src_mask]
        tgt_pred = pred[tgt_mask]
        wd = src_pred.mean() - tgt_pred.mean()
        loss += wd
        da_info['embed_losses'].append(wd.detach().cpu())
        wd_sim_list.append(src_pred.mean().item())
        wd_meas_list.append(tgt_pred.mean().item())
    da_info['wd_sim_mean'] = torch.tensor(np.sum(wd_sim_list))
    da_info['wd_meas_mean'] = torch.tensor(np.sum(wd_meas_list))
    return loss


def gradient_penalty(da_net, src_embed, tgt_embed):
    """Gradient Penalty is added to the loss as regulator based on gradient norm."""
    # make source and target embeddings the same size
    src_batch_size = src_embed.size(0)
    tgt_batch_size = tgt_embed.size(0)
    batch_size = src_batch_size + tgt_batch_size
    src_repeats = math.ceil(batch_size / src_batch_size)
    tgt_repeats = math.ceil(batch_size / tgt_batch_size)
    # handle case when source and target are not of same size
    src_embed_rep = torch.cat([src_embed] * src_repeats, dim=0)[:batch_size]
    tgt_embed_rep = torch.cat([tgt_embed] * tgt_repeats, dim=0)[:batch_size]

    alpha = torch.rand((batch_size, 1), device=src_embed_rep.device)
    differences = tgt_embed_rep - src_embed_rep
    interpolates = src_embed_rep + alpha * differences
    interpolates = torch.cat([interpolates, src_embed_rep, tgt_embed_rep]).requires_grad_()
    preds = da_net(interpolates)
    gradients = grad(outputs=preds, inputs=interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1)**2).mean()
    return gp
