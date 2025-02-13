import argparse
import datetime
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import sys
sys.path.insert(1, './')

from debias.datasets.cifar10 import get_cifar10
from debias.networks.resnet_cifar import ResNet18
from debias.utils.logging import set_logging
from debias.utils.utils import (AverageMeter, MultiDimAverageMeter, accuracy,
                                pretty_dict, save_model, set_seed)

from debias.datasets.utils import over_sample_features, under_sample_features

from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--bs', type=int, default=128, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--corr', type=float, default=0.95)
    parser.add_argument('--ecu', type=int, default=0)
    parser.add_argument('--uw', type=int, default=1)
    parser.add_argument('--mode', type=str, default='none')

    opt = parser.parse_args()

    return opt


def set_model():
    model = ResNet18(num_classes=10).cuda()
    pred = nn.Linear(512, 10).cuda()

    models = { 
                'model': model, 
                'pred': pred 
             }

    criterion = { 
                'bin': nn.BCEWithLogitsLoss(reduction='none'),
                'multi': nn.CrossEntropyLoss(reduction='none')
                } 

    return models, criterion
def train(train_loader, model, criterion, optimizer_model, opt, optimizer_layer):
    model['model'].train()
    avg_loss = AverageMeter()

    train_iter = iter(train_loader)
    
    all_labels_nb = [] 
    all_gc = [] 
    all_feats = [] 
    all_bias = [] 

    for images, labels, labels_bin, biases, gc, _ in tqdm(train_iter, ascii=True):
         
        bsz = labels.shape[0]
        labels_bin, biases = labels_bin.cuda(), biases.cuda()
    
        images = images.cuda()
        logits, feat = model['model'](images)

        multi = torch.ones_like(labels_bin) 

        multi[labels_bin == -1] = 0 
        labels_bin[labels_bin == -1] = 0 

        loss = criterion['bin'](logits, labels_bin)
        loss = loss*multi 

        div = torch.sum(multi) 
        loss = torch.sum(loss/div)

        avg_loss.update(loss.item(), bsz)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        all_labels_nb.append(labels.cpu().detach().numpy())
        all_gc.append(gc.numpy())
        all_bias.append(biases.cpu().detach().numpy())
        all_feats.append(feat.cpu().detach().numpy())

    all_labels_nb = np.concatenate(all_labels_nb, axis=0) 
    all_gc = np.concatenate(all_gc, axis=0) 
    all_bias = np.concatenate(all_bias, axis=0) 
    all_feats = np.concatenate(all_feats, axis=0) 

    if opt.mode == 'os':
        all_feats, all_labels_nb = over_sample_features(all_bias, all_feats, all_labels_nb)

    elif opt.mode == 'us':
        all_feats, all_labels_nb = under_sample_features(all_bias, all_feats, all_labels_nb)

    batch_size = opt.bs 
    total_samples = len(all_labels_nb)
    num_batches = total_samples//batch_size 

    for _ in tqdm(range(1), ascii=True):
        all_idx = np.arange(total_samples)
        np.random.shuffle(all_idx)

        all_feats[all_idx] = all_feats 
        all_labels_nb[all_idx] = all_labels_nb
        if opt.mode == 'uw':
            all_gc[all_idx] = all_gc

        for batch_idx in range(num_batches): 
            start = batch_idx * batch_size 
            end = min(total_samples, start + batch_size)

            feats = torch.from_numpy(all_feats[start:end]).cuda()
            labels = torch.from_numpy(all_labels_nb[start:end]).cuda()
            gc = torch.from_numpy(all_gc[start:end]).cuda()

            optimizer_layer.zero_grad() 
            out_lr = model['pred'](feats)

            loss = criterion['multi'](out_lr, labels)

            if opt.mode == 'uw':
                loss *= gc
            
            loss = torch.mean(loss)    
            loss.backward() 
            optimizer_layer.step()
        
    return avg_loss.avg



def validate(val_loader, model):
    model['model'].eval()
    
    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(10, 2))
    
    with torch.no_grad():
        for images, labels, _, biases, _, _ in val_loader:
            images = images.cuda()
            bsz = labels.shape[0]
            
            output,feats = model['model'](images)
            output = model['pred'](feats).detach().cpu()
            preds = output.data.max(1, keepdim=True)[1].squeeze(1)
            
            acc1, = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            corrects = (preds == labels).long()
            attrwise_acc_meter.add(corrects.cpu(), 
                                  torch.stack([labels.cpu(), biases.cpu()], dim=1))

    bc_classes = [1]*10 
    for i in [0, 2, 4, 6, 8]: 
        bc_classes[i] = 0

    return top1.avg, attrwise_acc_meter.get_unbiased_acc(), attrwise_acc_meter.get_acc_diff(), attrwise_acc_meter.get_bias_conflict(bc_classes)


def main():
    opt = parse_option()

    exp_name = f'bm-cifar-{opt.exp_name}-lr{opt.lr}-bs{opt.bs}-seed{opt.seed}'
    opt.exp_name = exp_name
    
    output_dir = f'exp_results/{exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, 'INFO', str(save_path))
    logging.info(f'Set seed: {opt.seed}')
    set_seed(opt.seed)
    logging.info(f'save_path: {save_path}')

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    root = './data/cifar10'
    train_loader = get_cifar10(root, 
        split='train', 
        aug=False, 
        under_sample='bin', 
        corr=opt.corr) 

    val_loaders = {}
    val_loaders['valid'] = get_cifar10(
        root,
        split='valid',
        aug=False, 
        corr=opt.corr)

    val_loaders['test'] = get_cifar10(
        root,
        split='test',
        aug=False)
    
    
    model, criterion = set_model()
    
    decay_epochs = [opt.epochs//4, opt.epochs//2, opt.epochs//1.333]

    optimizer = torch.optim.SGD(model['model'].parameters(), lr=opt.lr, momentum = 0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)

    optimizer_layer = torch.optim.SGD(model['pred'].parameters(), lr=opt.lr, momentum = 0.9, weight_decay=5e-4)
    scheduler_layer = torch.optim.lr_scheduler.MultiStepLR(optimizer_layer, milestones=decay_epochs, gamma=0.1)

    logging.info(f"decay_epochs: {decay_epochs}")

    #(save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)

    best_accs = {'valid': 0, 'test': 0}
    best_diff = {'valid':0, 'test':0}
    best_epochs = {'valid': 0, 'test': 0}
    best_stats = {'valid': 0, 'test': 0}
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        logging.info(f'[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}')
        loss = train(train_loader, model, criterion, optimizer, opt, optimizer_layer)
        logging.info(f'[{epoch} / {opt.epochs}] Loss: {loss}')

        scheduler.step()
        scheduler_layer.step()

        stats = pretty_dict(epoch=epoch)
        for key, val_loader in val_loaders.items():
            _, acc_unbiased, diff, bias_conflict = validate(val_loader, model)
            stats[f'{key}/acc_unbiased'] = acc_unbiased.item() * 100
            stats[f'{key}/diff'] = diff.item() * 100
            stats[f'{key}/bias_conflict'] = bias_conflict.item() * 100
            

        logging.info(f'[{epoch} / {opt.epochs}] {stats}')
        for tag in best_accs.keys():
            if stats[f'{tag}/acc_unbiased'] > best_accs[tag]:
                best_accs[tag] = stats[f'{tag}/acc_unbiased']
                best_epochs[tag] = epoch
                best_stats[tag] = pretty_dict(**{f'best_{tag}_{k}': v for k, v in stats.items()})

            logging.info(
                f'[{epoch} / {opt.epochs}] best {tag} accuracy: {best_accs[tag]:.3f} at epoch {best_epochs[tag]} \n best_stats: {best_stats[tag]}')

        if opt.ecu: 
          train_loader.dataset.under_sample_ce(verbose=False)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Total training time: {total_time_str}')

if __name__ == '__main__':
    main()
