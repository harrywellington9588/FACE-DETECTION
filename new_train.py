import os
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from dataset import MyDataSet
import onet
import rnet


class Trainer:
    def __init__(self, net, data_path, save_path, isCuda=True):
        self.net = net
        self.data_path = data_path
        self.save_path = save_path
        self.isCuda = isCuda

        if isCuda:
            self.net.cuda()

        self.cls_loss_func = nn.BCELoss()
        self.offset_loss_func = nn.MSELoss()

        self.opt = optim.Adam(self.net.parameters())

        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))

    def train(self):
        data = MyDataSet(self.data_path)
        dataloader = DataLoader(data, batch_size=256, shuffle=True, num_workers=4, drop_last=True)

        for epoch in range(15):
            for i, (_img_data, _label, _offset) in enumerate(dataloader):
                if self.isCuda:
                    _img_data = _img_data.cuda()
                    _label = _label.cuda()
                    _offset = _offset.cuda()

                out_category, out_offset = self.net(_img_data)

                out_category = out_category.view(-1, 1)
                category_mask = torch.lt(_label, 2)
                label_ = _label[category_mask]
                out_category_ = out_category[category_mask]
                cls_loss = self.cls_loss_func(out_category_, label_)

                out_offset = out_offset.view(-1, 4)
                offset_mask = torch.gt(_label, 0)  # [None, 4, 1, 1]
                offset_ = _offset[offset_mask[:, 0]]
                out_offset_ = out_offset[offset_mask[:, 0]]
                offset_loss = self.offset_loss_func(out_offset_, offset_)

                loss = cls_loss + offset_loss * 0.5

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                # if i % 5 == 0:
                #     print(" loss:", loss.cpu().data.numpy(), " cls_loss:", cls_loss.cpu().data.numpy(),
                #           " offset_loss", offset_loss.cpu().data.numpy())

            torch.save(self.net.state_dict(), self.save_path)
            print(epoch, "save success")


if __name__ == '__main__':
    # train_1 = Trainer(net.pnet(), r"E:\plane\train\12", r"E:\plane\train/p_params.pkl")
    # train_1.train()
    #
    train_2 = Trainer(rnet.rnet(), r"E:\plane\train\24", r"E:\plane\train/r_params_new.pkl")
    train_2.train()

    # train_3 = Trainer(onet.onet(), r"E:\plane\train\48", r"E:\plane\train/o_params_new.pkl")
    # train_3.train()





