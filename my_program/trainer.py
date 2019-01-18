import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from my_program.read_data import face_dataset
from my_program import nets


class Trainer():
    def __init__(self, net, path, save_path, img_size, isCuda=True):
        self.net = net
        self.path = path
        self.save_path = save_path
        self.isCuda = isCuda
        self.img_size = img_size

        if self.isCuda:
            self.net.cuda()

        self.cond_loss_function = nn.BCELoss()
        self.offset_loss_function = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters())

        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))

    def train(self):
        dataset = face_dataset(self.path, self.img_size)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
        length = len(dataloader)
        for epoch in range(10):
            for i, (img, label, offset) in enumerate(dataloader):
                if self.isCuda:
                    img = img.cuda()
                    label = label.cuda()
                    offset = offset.cuda()

                out_cond, out_offset = self.net(img)

                out_cond = out_cond.view(-1, 1)
                cond_mask = torch.lt(label, 2)
                label_ = label[cond_mask]
                out_cond_ = out_cond[cond_mask]
                cond_loss = self.cond_loss_function(out_cond_, label_)

                out_offset = out_offset.view(-1, 4)
                offset_mask = torch.gt(label, 0)
                offset_ = offset[offset_mask[:, 0]]
                out_offset_ = out_offset[offset_mask[:, 0]]
                offset_loss = self.offset_loss_function(out_offset_, offset_)

                loss = cond_loss + offset_loss * 0.5
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print("epoch{0}: {1}/{2}//// loss:{3}".format(epoch, i+1, length, loss.cpu().data.numpy()))

            torch.save(self.net.state_dict(), self.save_path)
            print("epoch{} save success".format(epoch))


if __name__ == '__main__':
    train_1 = Trainer(nets.pnet(), r"E:\celeba\train_dataset",
                      r"D:\pycharm_project\yxy_mtcnn\parameters\pnet_params.pt", 12)
    train_1.train()
    train_2 = Trainer(nets.rnet(), r"E:\celeba\train_dataset",
                      r"D:\pycharm_project\yxy_mtcnn\parameters\rnet_params.pt", 24)
    train_2.train()
    train_3 = Trainer(nets.onet(), r"E:\celeba\train_dataset",
                      r"D:\pycharm_project\yxy_mtcnn\parameters\onet_params.pt", 48)
    train_3.train()



