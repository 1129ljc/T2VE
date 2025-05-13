import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from model import VideoDetection
from dataload import GenVideoTest


def test():
    device = torch.device('cuda:0')
    ckpt_path = 'T2VE-ckpt.pth'
    ckpt = torch.load(ckpt_path)
    new_checkpoint = {}
    for key, value in ckpt.items():
        if 'module.' in str(key):
            new_key = str(key).replace('module.', '')
            new_checkpoint[new_key] = value
    model = VideoDetection()
    model.load_state_dict(new_checkpoint)
    model = model.to(device)
    model.eval()

    test_dataset = GenVideoTest()
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=32)
    print('test_dataset_len:', len(test_dataset))
    print('test_loader_len:', len(test_loader))

    print('start testing ...')
    predict_list = []
    label_list = []
    with torch.no_grad():
        for idx, (frames, label) in enumerate(test_loader):
            frames_torch, label_torch = frames.float().to(device), label.float().to(device)
            predict = model(frames_torch)
            predict = predict.cpu().detach().numpy().astype('float').flatten()
            predict = predict >= 0.5
            label = label.cpu().detach().numpy().astype('float').flatten()
            predict_list.append(predict)
            label_list.append(label)
    test_acc = accuracy_score(label_list, predict_list)
    print('test_acc :', test_acc)


if __name__ == '__main__':
    test()
