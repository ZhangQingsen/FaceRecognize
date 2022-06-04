import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader

import dataloader
import model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.Resize((114, 114)),
                                transforms.ToTensor(),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


def save_model(_model, dir_path, epoch, _loss, accu):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    state = {'model':_model.state_dict(), 'epoch':epoch}
    torch.save(state, os.path.join(dir_path, 'ep%03d-loss%.3f-accu%.3f.pth' % (epoch, _loss, accu)))
    # 加载数据


def loss(y_pred, Batch_size, alpha=0.2):
    if len(y_pred) < 3*Batch_size:
        Batch_size_prev = Batch_size
        Batch_size = len(y_pred) // 3
        # print(f"Not enough y_pred ({len(y_pred)}<3*{Batch_size_prev}) left for calculation, Batch_size {Batch_size_prev} ==> {Batch_size}")
    anchor, positive, negative = y_pred[:int(Batch_size)], y_pred[int(Batch_size):int(2 * Batch_size)], y_pred[int(2 * Batch_size):]

    pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), axis=-1))
    neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), axis=-1))

    keep_all = (neg_dist - pos_dist < alpha).cpu().numpy().flatten()
    hard_triplets = np.where(keep_all == 1)

    pos_dist = pos_dist[hard_triplets]
    neg_dist = neg_dist[hard_triplets]

    basic_loss = pos_dist - neg_dist + alpha
    res_loss = torch.sum(basic_loss) / torch.max(torch.tensor(1), torch.tensor(len(hard_triplets[0])))
    return res_loss

def train(_model, train_df, epoch, Batch_size, scaler, optimizer):
    _model.train()
    train_loader = DataLoader(dataloader.DataLoader(train_df, 114), Batch_size, shuffle=True, collate_fn=dataloader.dataset_collate)
    train_length = train_df.shape[0]

    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0

    for iteration, batch in enumerate(train_loader):
        images, labels = batch
        with torch.no_grad():
            images = torch.from_numpy(images).type(torch.FloatTensor).to(device)
            labels = torch.from_numpy(labels).long().to(device)
            # images = images.to(device)
            # labels = labels.long().to(device)
        optimizer.zero_grad()
        torchV_lgt_171 = True
        if not torchV_lgt_171:
            outputs1, outputs2 = _model(images, "train")
            _triplet_loss = loss(outputs1, Batch_size)
            _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
            _loss = _triplet_loss + _CE_loss

            _loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs1, outputs2 = _model(images, "train")

                _triplet_loss = loss(outputs1, Batch_size)
                _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
                _loss = _triplet_loss + _CE_loss
            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            accuracy = torch.mean(
                (torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
        total_triple_loss += _triplet_loss.item()
        total_CE_loss += _CE_loss.item()
        total_accuracy += accuracy.item()

        print(f"\tTrain:{((iteration+1)*Batch_size/train_length) if ((iteration+1)*Batch_size/train_length) < 1 else 1:.2%} \
        -----accu:{(total_accuracy / (iteration + 1)):.2%}-----loss:{(total_CE_loss / (iteration + 1)):.2f}-----------\
            {(iteration+1)*Batch_size if (iteration+1)*Batch_size < train_length else train_length}/{train_length}")
    # save_model(_model, "./model_data/", epoch, total_CE_loss, total_accuracy)
    return total_CE_loss/ (iteration + 1), total_accuracy/ (iteration + 1)

def test(_model, test_df, epoch, Batch_size):
    _model.eval()
    test_loader = DataLoader(dataloader.DataLoader(test_df, 114), Batch_size, shuffle=True, collate_fn=dataloader.dataset_collate)
    test_length = test_df.shape[0]

    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0

    for iteration, batch in enumerate(test_loader):
        images, labels = batch
        with torch.no_grad():
            images = torch.from_numpy(images).type(torch.FloatTensor).to(device)
            labels = torch.from_numpy(labels).long().to(device)
            # images = images.to(device)
            # labels = labels.long().to(device)

        outputs1, outputs2 = _model(images, "train")

        _triplet_loss = loss(outputs1, Batch_size)
        _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
        _loss = _triplet_loss + _CE_loss


        with torch.no_grad():
            accuracy = torch.mean(
                (torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
        total_triple_loss += _triplet_loss.item()
        total_CE_loss += _CE_loss.item()
        total_accuracy += accuracy.item()

        # print(f"\t{((iteration+1)*Batch_size/test_length) if ((iteration+1)*Batch_size/test_length) < 1 else 1:.2%} \
        # -----accu:{(total_accuracy / (iteration + 1)):.2%}-----loss:{(total_CE_loss / (iteration + 1)):.2f}-----------\
        #     {(iteration+1)*Batch_size if (iteration+1)*Batch_size < test_length else test_length}/{test_length}")
        print(f"\tTest:\t{((iteration+1)*Batch_size/test_length) if ((iteration+1)*Batch_size/test_length) < 1 else 1:.2%}-----------{(iteration+1)*Batch_size if (iteration+1)*Batch_size < test_length else test_length}/{test_length}")
    
    print(f'Test------test accu:{total_accuracy/ (iteration + 1):.2%}, test loss:{total_CE_loss/ (iteration + 1):.2f}')
    # save_model(_model, "./model_data/", epoch, total_CE_loss, total_accuracy)
    return total_CE_loss/ (iteration + 1), total_accuracy/ (iteration + 1)


def run(df, curr_epoch, epoch_step, Batch_size, lr=0.01, split_rate=0.7, model_load_dir=''):
    
    train_df = df.sample(frac=split_rate)
    test_df = df.sample(frac=(1-split_rate))

    
    model_train = model.ConvNet(num_classes=len(np.unique(df["Name"])))
    if len(model_load_dir):
        checkpoint = torch.load(model_load_dir)
        model_train.load_state_dict(checkpoint['model'])
        curr_epoch = checkpoint['epoch']

    model_train.to(device)
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.SGD(model_train.parameters(), lr, momentum=0.9)
    # optimizer = optim.Adam(model_train.parameters(), lr=0.01,betas=(0.9,0.999))
    for e in range(1,epoch_step+1):
        print(f"Epoch:{curr_epoch+e}")
        train_loss, train_accu = train(model_train, train_df,curr_epoch+e,Batch_size,scaler,optimizer)
        test_loss, test_accu = test(model_train, test_df,curr_epoch+e,Batch_size)
        save_model(model_train, "./model_data/", curr_epoch+e, test_loss, test_accu)
        
