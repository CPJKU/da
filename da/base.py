"""
Contains the base class 'DABase' as the entry point for creating a new alignment component.
"""

import torch.nn as nn
import torch
import os
import json
from typing import List

from da.helpers import get_activation, single_unit, build_units
from da import cmd, dann, wdgrl, mmd, swd, coral, deep_j_dot


class DABase:
    """
    This is the main class that sets up the alignment component of a specific domain adaptation type.

    Attributes
    ----------
        embeds_size : list
            dimension of all the embeddings (activations) of the base network that are passed to this class
        embeds_idx : list
            indices of embeddings/layer activations that should be used for calculating domain loss,
            layer 0 = first layer, layer -1 = last layer
        da_type : string
            specifies which type of alignment component is used
            options: 'mmd', 'cmd', 'swd', 'coral', 'dann', 'wdgrl', 'jdot'
        da_lambda : float
            trades of clf/reg loss and domain loss
            overall loss usually calculated as: loss = clf_loss + da_lambda * domain_loss
        lambda_auto_schedule: bool
            if schedule for lambda is applied (False => constant lambda)
        lambda_pretrain_steps: int
            steps until lambda starts to increase
            steps --> number of times get_da_loss(..) is called
            usually get_da_loss(..) is called once on every batch
        lambda_inc_steps : int
            steps until lambda reaches value specified in 'lambda_final'
        lambda_final : float
            final lambda value, if lambda_auto_schedule is 'True'
        num_domains : int
            number of different domains (usually 2 - source and target)
        num_classes : int
            number of classes of the underlying classification problem ('None'=regression problem)
        adv_config : dict
            contains parameters for configuring adversarial da net and optimizer (for wdgrl and dann)
            standard parameters are loaded from 'adv_conf.json' and updated with the provided dict
        da_spec_config : dict
            contains parameters specific to the domain adaptation methods
            standard parameters are loaded from 'da_spec_conf.json' and updated with the provided dict
        da_net : DANetworks
            in case the alignment component consists of a network (e.g. 'dann' or 'wdgrl') a DANetworks object is
            created that handles the networks
        da_optimizer : torch.optim.Adam
            in case the networks stored in da_net need to be optimized inside this package (e.g. in 'wdgrl')
        loss_steps : int
            counts the total number of times get_da_loss(..) has been called

    Methods
    -------
        _setup_da :
            Setting up the respective alignment compoenent
        get_da_params :
            Returns list containing all parameters of the da networks, can be used if da networks are optimized
            outside of this package
        get_da_nets :
            Returns list of all da networks as pytorch modules
        update_lambda :
            Set a new value for 'da_lambda'
        get_current_lambda :
            Returns value of 'da_lambda'
        _auto_lambda_scheduler :
            Automatically called, returns updated value of 'da_lambda'
    """

    def __init__(self,
                 embeds_size: list,
                 embeds_idx: tuple = (-1,),
                 da_type: str = 'dann',
                 da_lambda: float = 1.0,
                 lambda_auto_schedule: bool = False,
                 lambda_pretrain_steps: int = 10000,
                 lambda_inc_steps: int = 100000,
                 lambda_final: float = 1.0,
                 num_domains: int = 2,
                 num_classes: int = 10,
                 adv_config: dict = {},
                 da_spec_config: dict = {}
                 ):

        """
        Sets up ingredients needed for domain adaptation.

        Parameters
        ----------
            embeds_size : list
                dimension of all the embeddings (activations) of the base network that are passed to this class
            embeds_idx : list
                indices of layer embeddings that should be used for calculating domain loss,
                layer 0 = first layer, layer -1 = last layer
            da_type : string
                specifies which type of alignment component is used
                options: 'mmd', 'cmd', 'swd', 'coral', 'dann', 'wdgrl', 'jdot'
            da_lambda : float
                trades of clf/reg loss and domain loss
                overall loss usually calculated as: loss = clf_loss + da_lambda * domain_loss
            lambda_auto_schedule: bool
                if schedule for lambda is applied (False => constant lambda)
            lambda_pretrain_steps: int
                steps until lambda starts to increase
                steps => number of times get_da_loss(..) is called
                usually get_da_loss(..) is called once on every batch
            lambda_inc_steps : int
                total number of steps until lambda reaches value specified in 'lambda_final'
            lambda_final : float
                final lambda value, if lambda_auto_schedule is 'True'
            num_domains : int
                number of different domains (usually 2 - source and target)
            num_classes : int
                number of classes of the underlying classification problem ('None'=regression problem)
            adv_config : dict
                contains parameters for configuring adversarial da net and optimizer (for wdgrl and dann)
                standard parameters are loaded from 'adv_conf.json' and updated with the provided dict
            da_spec_config : dict
                contains parameters specific to the domain adaptation methods
                standard parameters are loaded from 'da_spec_conf.json' and updated with the provided dict
        """
        self.embeds_size = embeds_size
        self.embeds_idx = embeds_idx
        self.da_type = da_type
        self.da_lambda = da_lambda
        self.lambda_auto_schedule = lambda_auto_schedule
        self.lambda_pretrain_steps = lambda_pretrain_steps
        self.lambda_inc_steps = lambda_inc_steps
        self.final_lambda = lambda_final
        self.lambda_inc_val = self.final_lambda / (self.lambda_inc_steps - self.lambda_pretrain_steps)
        self.num_domains = num_domains
        self.num_classes = num_classes

        da_spec_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'da_spec_conf.json')
        assert os.path.exists(da_spec_config_file)
        # load da algorithm's standard config and update with given config
        with open(da_spec_config_file) as f:
            self.da_spec_config = json.load(f)
            # update standard config with given parameters
            self.da_spec_config.update(da_spec_config)

        # load adversarial net standard config in case of wdgrl or dann and update with given config
        if da_type in ["wdgrl", "dann"]:
            adv_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'adv_conf.json')
            assert os.path.exists(adv_config_file)
            with open(adv_config_file) as f:
                self.adv_config = json.load(f)
                # update standard config with given parameters
                self.adv_config.update(adv_config)

        self.da_net = None
        self.da_optimizer = None
        self.loss_steps = 0

        assert len(self.embeds_idx) > 0
        assert self.da_type in ["cmd", "coral", "dann", "jdot", "mmd", "swd", "wdgrl"]
        if self.lambda_auto_schedule:
            assert self.lambda_pretrain_steps < self.lambda_inc_steps
        assert self.num_domains > 1
        self._setup_da()

    def _setup_da(self):
        """Sets up an alignment component of a specific type."""
        if self.da_type == "dann":
            # Domain Adversarial Training of Neural Networks
            self.da_net = get_da_net(self.adv_config['da_net_config'], self.embeds_size, self.embeds_idx,
                                     self.num_domains)
            # if dann should be updated inside package and not via gradient reversal
            if self.da_spec_config['dann']['auto_critic_update']:
                self.da_optimizer = get_optimizer(self.da_net.get_params(), **self.adv_config['da_optimizer_config'])
        elif self.da_type == "cmd":
            # Central Moment Discrepancy
            # no da net required - works directly on the embeddings (activations)
            assert self.num_domains == 2
            assert self.da_spec_config['cmd']['n_moments'] >= 1
        elif self.da_type == "wdgrl":
            # Wasserstein Distance Guided Representation Learning
            # WDGRL uses two optimizers instead of gradient reversal (according to original paper)
            assert self.num_domains == 2
            self.da_net = get_da_net(self.adv_config['da_net_config'], self.embeds_size, self.embeds_idx,
                                     n_outs=1)
            self.da_optimizer = get_optimizer(self.da_net.get_params(), **self.adv_config['da_optimizer_config'])
        elif self.da_type == "swd":
            assert self.num_domains == 2
        elif self.da_type == "mmd":
            assert self.num_domains == 2
            assert self.da_spec_config['mmd']['kernel_num'] >= 1
        elif self.da_type == "coral":
            assert self.num_domains == 2
        elif self.da_type == "jdot":
            assert self.num_domains == 2
            # deepjdot currently only implemented for single embedding layer
            assert len(self.embeds_idx) == 1
            # deepjdot currently not implemented for regression
            assert self.num_classes is not None
            self.deepJDot = deep_j_dot.DeepJDot(self.num_classes, self.da_spec_config['jdot']['jdot_alpha'])

        if self.da_net:
            print("DA Net: ", self.da_net)

    def get_da_params(self):
        """Returns da network parameters. Use this function to get parameters and pass it to your optimizer."""
        if self.da_net:
            return list(self.da_net.get_params())
        else:
            return list()

    def get_da_nets(self):
        """Returns da network. Necessary when using e.g. Pytorch Lightening to register da networks as attribute of
        lightening module."""
        if self.da_net:
            return self.da_net.get_nets()
        else:
            return None

    def update_lambda(self, value):
        """Set new value for da_lambda. Allows for creating lambda schedule outside package."""
        self.da_lambda = value

    def get_current_lambda(self):
        """Returns current value of da_lambda."""
        return self.da_lambda

    def get_current_loss_step(self):
        """Returns total number of times get_da_loss(..) has been called."""
        return self.loss_steps

    def _auto_lambda_scheduler(self):
        """Called by get_da_loss(..) and returns updated 'da_lambda' value based on 'loss_steps'."""
        if self.loss_steps < self.lambda_pretrain_steps:
            return 0.0
        elif self.loss_steps > self.lambda_inc_steps:
            return self.final_lambda
        else:
            return self.da_lambda + self.lambda_inc_val

    def get_da_loss(self,
                    embeds: List[torch.Tensor],
                    domain_labels: torch.Tensor,
                    labels: torch.Tensor = None,
                    predictions: torch.Tensor = None):
        """Calculates weighted domain adaptation loss and gathers additional information like losses/accuracies achieved
        for each received embedding.

        Parameters
        ----------
            embeds : List[torch.Tensor]
                List of all embeddings from the base network. Those are filtered and sorted by 'embeds_idx'
            domain_labels : torch.Tensor
                List specifying whether a sample stems from source (=0) or target domain (=1)
            labels : torch.Tensor
                Labels only needed for 'jdot'; for target domain samples dummy labels can be passed
            predictions : torch.Tensor
                Predictions (output of classification or regression head) only needed for 'jdot'

        Returns
        -------
            loss : torch.Tensor
                loss produced by alignment component, including weighting by da_lambda
            da_info : dict
                additional stats that might be helpful to debug domain adaptation
                (e.g. loss produced per embedding or accuracy achieved by domain critic in case of 'dann')
        """
        assert embeds is not None and len(embeds) > 0
        assert all([type(embed) == torch.Tensor for embed in embeds])
        assert domain_labels is not None and type(domain_labels) == torch.Tensor
        assert all([embed.device == domain_labels.device for embed in embeds])
        assert all([len(embeds[i]) == len(domain_labels) for i in range(len(embeds))])
        if self.da_type == "jdot":
            assert labels is not None and type(labels) == torch.Tensor
            assert predictions is not None and type(predictions) == torch.Tensor
            assert labels.device == predictions.device == domain_labels.device
            assert len(domain_labels) == len(labels) == len(predictions)

        loss = torch.tensor(0., device=domain_labels.device)
        # statistics that are logged by algorithms
        da_info = {'embed_losses': []}

        # get correct embeds (filter them)
        try:
            embeds = [embeds[i] for i in self.embeds_idx]
        except IndexError:
            # in case given embed indices do not fit the list of embeddings given
            print("Index Error: Len of embeds: {}, embed indices: {}".format(len(embeds), self.embeds_idx))
            return loss, da_info

        self.loss_steps += 1

        # update value for lambda
        if self.lambda_auto_schedule:
            self.da_lambda = self._auto_lambda_scheduler()

        if self.da_type == "dann":
            # bring net to correct device
            self.da_net.nets.to(domain_labels.device)
            if self.da_spec_config['dann']['auto_critic_update']:
                dann.dann_update(embeds, domain_labels, self.da_optimizer, self.da_spec_config['dann']['critic_iter'],
                                 self.da_net)

            loss += dann.dann_loss(embeds, domain_labels, self.da_spec_config['dann']['grad_scale_factor'],
                                   self.da_net, da_info)
        elif self.da_type == "cmd":
            loss += cmd.cmd_loss(embeds, domain_labels, self.da_spec_config['cmd']['n_moments'], da_info)
        elif self.da_type == "wdgrl":
            # bring net to correct device
            self.da_net.nets.to(domain_labels.device)
            # first update critic
            wdgrl.wdgrl_update(embeds, domain_labels, self.da_optimizer, self.da_spec_config['wdgrl']['critic_iter'],
                               self.da_spec_config['wdgrl']['gp_da_lambda'], self.da_net,
                               da_info)
            # next calculate loss used to update feature extractor
            loss += wdgrl.wdgrl_loss(embeds, domain_labels, self.da_net, da_info)
        elif self.da_type == "mmd":
            loss += mmd.mmd_loss(embeds, domain_labels, self.da_spec_config['mmd']['kernel_mul'],
                                 self.da_spec_config['mmd']['kernel_num'],
                                 self.da_spec_config['mmd']['fix_sigma'], da_info)
        elif self.da_type == "swd":
            loss += swd.swd_loss(embeds, domain_labels, self.da_spec_config['swd']['multiplier'],
                                 self.da_spec_config['swd']['p'], da_info)
        elif self.da_type == "coral":
            loss += coral.coral_loss(embeds, domain_labels, da_info)
        elif self.da_type == "jdot":
            # jdot loss also requires labels and predictions
            loss += self.deepJDot.jdot_loss(embeds, domain_labels, labels, predictions, da_info)
        return loss * self.da_lambda, da_info


def get_da_net(da_net_config, embeds_size, embeds_idx, n_outs):
    """Create da networks given a specific network config and the input sizes (embed size)."""
    # filter embeddings on basis of embeds_idx
    embeds_size = [embeds_size[i] for i in embeds_idx]
    return DANetworks(da_net_config, embeds_size, n_outs)


def get_optimizer(params, lr=0.0001, weight_decay=0.001):
    """Returns optimizer given optimizer settings. This could be extended to allow different types
    of optimizers."""
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


class DANetworks:
    """
    This class is not a module itself, but it creates and manages all da networks (as a module list)
     that process the embeddings. It is only used by DABase and should not be called
     from outside the package.
    """

    def __init__(self, net_config, embeds_size, n_outs):
        self.nets = []
        layers_width = net_config['layers_width'] # specifies nr of layers and their width
        act_funct = get_activation(net_config['act_function']) # activation function of each layer
        dropout = net_config['dropout'] # dropout on each layers

        # loop over all embedding layers that should be processed later by a da network
        for input_size in embeds_size:
            if len(layers_width) >= 1:
                in_layer = single_unit(input_size, layers_width[0], act_funct, dropout)
                out_layer = single_unit(layers_width[-1], n_outs, None, 0.0)
                hidden = build_units(layers_width, act_funct, dropout)
            else:
                in_layer = nn.Sequential()
                out_layer = single_unit(input_size, n_outs, None, 0.0)
                hidden = nn.Sequential()
            da_net = nn.Sequential(in_layer, hidden, out_layer)
            self.nets.append(da_net)
        self.nets = nn.ModuleList(self.nets)
        print("DA networks: ", self.nets)

    def forward(self, embeds_list):
        # embeds_list - filtered list of embeddings
        # feed corresponding embeddings to respective da networks
        da_outs = []
        for i in range(len(embeds_list)):
            # pass embeddings of a certain position through corresponding da network
            da_out = self.nets[i](embeds_list[i])
            da_outs.append(da_out)
        return da_outs

    def get_params(self):
        params = list()
        for net in self.nets:
            params += net.parameters()
        return params

    def get_nets(self):
        return self.nets
