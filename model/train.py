import os
import torch
import datetime
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from model import VideoDetection
from dataload import GenVideoTrain, GenVideoVal

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

def train():
    epochs = 30
    train_batch_size = 128
    val_batch_size = 64
    lr = 0.0001
    ckpt_save_dir = '/data2/ljc/code/T2VE/ckpt/'

    model = VideoDetection()
    model.encoder.load_state_dict(torch.load('vision_encoder.pth'))
    for param in model.encoder.parameters():
        param.requires_grad = False
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-8)
    loss_fn = nn.BCEWithLogitsLoss()

    train_dataset = GenVideoTrain()
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=32, drop_last=True)
    print('train_dataset_len:', len(train_dataset))
    print('train_loader_len:', len(train_loader))
    val_dataset = GenVideoVal()
    val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=32, drop_last=True)
    print('test_dataset_len:', len(val_dataset))
    print('test_loader_len:', len(val_loader))

    val_info = []
    max_acc = 0.0
    max_acc_index = 0
    print('start training ...')
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_loss1 = 0.0
        for idx, (frames, label) in enumerate(train_loader):
            frames_torch, label_torch = frames.float().cuda(), label.float().cuda()
            predict = model(frames_torch)
            optimizer.zero_grad()
            loss = loss_fn(predict.squeeze(1), label_torch)
            loss.backward()
            optimizer.step()
            iter_loss = loss.cpu().float()
            epoch_train_loss = epoch_train_loss + iter_loss
            epoch_train_loss1 = epoch_train_loss + iter_loss
            time = datetime.datetime.now().strftime('%H:%M:%S')
            if (idx + 1) % 50 == 0 and idx != 0:
                epoch_train_loss = epoch_train_loss / 50
                log_string = '[%s] epoch:[%d] iter:[%d]/[%d] loss:[%.5f]' % (time, epoch + 1, idx + 1, len(train_loader), epoch_train_loss)
                print(log_string)
                epoch_train_loss = 0.0
        epoch_train_loss1 = epoch_train_loss1 / len(train_loader)
        log_string = 'epoch:[%d] train_loss:[%.5f]' % (epoch + 1, epoch_train_loss1)
        print(log_string)
        model.eval()
        val_loss = 0.0
        predict_list = []
        label_list = []
        with torch.no_grad():
            for idx, (frames, label) in enumerate(val_loader):
                frames_torch, label_torch = frames.float().cuda(), label.float().cuda()
                predict = model(frames_torch)
                loss = loss_fn(predict.squeeze(1), label_torch)
                iter_loss = loss.cpu().float()
                val_loss = val_loss + iter_loss
                predict = predict.cpu().detach().numpy().astype('float').flatten()
                label = label.cpu().detach().numpy().astype('int').flatten()
                predict = predict >= 0.5
                predict_list.extend(predict.tolist())
                label_list.extend(label.tolist())
                time = datetime.datetime.now().strftime('%H:%M:%S')
                if (idx + 1) % 50 == 0 and idx != 0:
                    log_string = '[%s] epoch:[%d] iter:[%d]/[%d]' % (time, epoch + 1, idx + 1, len(val_loader))
                    print(log_string)
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(label_list, predict_list)
        torch.save(model.classifier.state_dict(), os.path.join(ckpt_save_dir, str(epoch + 1) + '.pth'))
        log_string = 'epoch:[%d] val_loss:[%.5f] val_acc:[%.5f]' % (epoch + 1, val_loss, val_acc)
        print(log_string)
        val_info.append(log_string)
        if val_acc > max_acc:
            max_acc = val_acc
            max_acc_index = epoch + 1
    for i in val_info:
        print(i)
    print('max_acc:[%.5f] max_acc_index:[%d]' % (max_acc, max_acc_index))

if __name__ == '__main__':
    train()
