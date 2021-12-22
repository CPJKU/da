import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from da.base import DABase
import pickle
import os
import json
import logging
import time
from pathlib import Path

from data import ProcessedDataset, OversamplingDataset
from models import Net
from train import train_model, evaluate


def main(run_conf, da_conf, exp_dir):
    # create original Model from models.py
    clf_model = Net()
    print(clf_model)
    clf_model = clf_model.to(run_conf['device'])
    print(da_conf)

    # based on da_type create da_model
    if da_conf['da_type']:
        embeds_size = clf_model.get_embed_size()
        da = DABase(embeds_size, **da_conf)
    else:
        da = None

    # loading dataset
    data = pickle.load(open(run_conf['data_dir'], 'rb'))
    train_dataset = ProcessedDataset(data['train'])
    train_dataset = OversamplingDataset(train_dataset, oversampling_rate=run_conf['oversampling_rate'])
    train_loader = DataLoader(train_dataset, batch_size=run_conf['batch_size'])

    test_dataset = ProcessedDataset(data['test'])
    test_loader = DataLoader(test_dataset, batch_size=run_conf['batch_size'])

    if da_config['da_type'] == "dann" and not da_config['da_spec_config']['dann']['auto_critic_update']:
        # in case of dann da net parameters are most commonly optimized outside package (grl)
        clf_optim = torch.optim.Adam(list(clf_model.parameters()) + da.get_da_params(), lr=run_conf['lr'], weight_decay=0.001)
    else:
        clf_optim = torch.optim.Adam(clf_model.parameters(), lr=run_conf['lr'], weight_decay=0.001)
    clf_criterion = nn.MSELoss()

    model_file = os.path.join(exp_dir, 'model.pt')
    if Path(model_file).exists():
        print("Loading model and skipping training, proceed with testing: ")
        clf_model.load_state_dict(torch.load(model_file))

        test_loss, test_mae, test_rmse = evaluate(clf_model, test_loader, clf_criterion, run_conf['device'])
        logger.info(f'test_loss={test_loss:.4f}, test_mae={test_mae:.4f}, test_rmse={test_rmse:.4f}')
    else:
        train_model(clf_model, da, train_loader, test_loader,
                    clf_criterion, clf_optim, run_conf['epochs'], run_conf['device'], logger)

        # save trained model
        torch.save(clf_model.state_dict(), model_file)
        print("Saved model to: ", model_file)


if __name__ == '__main__':
    # save settings of the experiment
    arg_parser = argparse.ArgumentParser(description='Domain adaptation LCM')
    arg_parser.add_argument('--name', type=str, default=None)
    arg_parser.add_argument('--da_type', type=str, default='dann')
    arg_parser.add_argument('--da_lambda', type=int, default=2)
    arg_parser.add_argument('--batch-size', type=int, default=32)
    arg_parser.add_argument('--epochs', type=int, default=60)
    arg_parser.add_argument('--device', type=str, default='cuda:0')
    arg_parser.add_argument('--data_path', type=str, default='./data/dataset.p')
    arg_parser.add_argument('--exp_dir', type=str, default=None)
    arg_parser.add_argument('--lr', type=int, default=1e-4)

    args = arg_parser.parse_args()
    args.device = args.device if torch.cuda.is_available() else 'cpu'

    if args.exp_dir:
        exp_folder = args.exp_dir
    else:
        # create experiment folder
        timestr = str(args.da_type) + time.strftime("%Y%m%d-%H%M%S")
        exp_folder = os.path.join('experiments', timestr)
        os.makedirs(exp_folder)
        print("Created experiment folder: ", exp_folder)

    # logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('train_da')
    file_log_handler = logging.FileHandler(os.path.join(exp_folder, 'exp_out.log'))
    logger.addHandler(file_log_handler)

    run_config = {
        'data_dir': args.data_path,
        'device': args.device if torch.cuda.is_available() else 'cpu',
        # 'plot': args.plot,
        'batch_size': args.batch_size,
        'oversampling_rate': 6,
        'lr': args.lr,
        'epochs': args.epochs
    }

    da_config = {
        'embeds_idx': [-1],
        'da_type': args.da_type,
        'da_lambda': args.da_lambda,
        'lambda_auto_schedule': False,
        'lambda_pretrain_steps': 3000,
        'lambda_inc_steps': 5000,
        'lambda_final': 1,
        'num_domains': 2,
        'num_classes': 1,
        # adversarial net parameters
        'adv_config': {
            'da_net_config': {
                'layers_width': [
                    32, 16
                ],
                'act_function': "relu",
                'dropout': 0.0
            },
            'da_optimizer_config': {
                'lr': 0.0001,
                'weight_decay': 0.001
            }
        },
        # da algorithm specific parameters
        'da_spec_config':
            {
                'cmd': {
                    'n_moments': 5
                },
                'coral': {},
                'dann': {
                    'grad_scale_factor': 1.0,
                    'auto_critic_update': False,
                    'critic_iter': 5,
                    'da_optimizer_config': {
                        'lr': 0.0001,
                        'weight_decay': 0.001
                    }
                },
                'jdot': {
                    'jdot_alpha': 10.0
                },
                'mmd': {
                    'kernel_mul': 2,
                    'kernel_num': 4,
                    'fix_sigma': None
                },
                'swd': {
                    'multiplier': 8,
                    'p': 2
                },
                'wdgrl': {
                    'gp_da_lambda': 1.0,
                    'critic_iter': 5,
                },
            }
    }

    with open(os.path.join(exp_folder, 'run_config.json'), 'w+') as f:
        json.dump(run_config, f)

    with open(os.path.join(exp_folder, 'da_config.json'), 'w+') as f:
        json.dump(da_config, f)

    main(run_config, da_config, exp_folder)
