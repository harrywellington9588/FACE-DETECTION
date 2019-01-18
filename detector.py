import torch
from PIL import Image, ImageDraw
import numpy as np
from my_tool import tool
import net
import onet
import rnet
import cv2


class Detector():
    def __init__(self, p_params=r"E:\plane\train/p_params.pkl",
                 r_params=r"E:\plane\train/r_params_new.pkl",
                 o_params=r"E:\plane\train/o_params_new.pkl", isCuda=False):
        self.isCuda = isCuda
        self.pnet = net.pnet()
        self.rnet = rnet.rnet()
        self.onet = onet.onet()

        if isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.pnet.load_state_dict(torch.load(p_params))
        self.rnet.load_state_dict(torch.load(r_params))
        self.onet.load_state_dict(torch.load(o_params))

    def detect(self, image):
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([])

        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])

        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])

        return onet_boxes

    def __pnet_detect(self, image):

        width, height = image.size
        minside = np.minimum(width, height)
        img = image
        boxes = []
        scale = 1
        while minside > 12:
            im_data = np.array(img) / 255. - 0.5
            im_data = im_data.transpose([2, 0, 1])
            im_data = torch.Tensor(im_data)
            img_data = im_data.unsqueeze(0)
            if self.isCuda:
                img_data.cuda()
            _cls, _offset = self.pnet(img_data)
            cls = _cls[0][0].cpu().data
            offset = _offset[0].cpu().data

            idxs = torch.nonzero(torch.gt(cls, 0.6))
            for idx in idxs:
                boxes.append(self.__box(idx, offset, cls[idx[0], idx[1]], scale))

            scale *= 0.7
            _width = int(width * scale)
            _height = int(height * scale)
            img = img.resize((_width, _height), Image.ANTIALIAS)
            minside = np.minimum(_width, _height)

        return tool.NMS(np.array(boxes), 0.7)

    def __rnet_detect(self, image, pnet_boxes):
        _img_dataset = []
        _pnet_boxes = tool.convert_to_square(pnet_boxes)

        for _box in _pnet_boxes:
            _x1, _y1, _x2, _y2 = map(int, _box[:4])
            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24), Image.ANTIALIAS)
            im_data = np.array(img) / 255. - 0.5
            im_data = im_data.transpose([2, 0, 1])
            im_data = torch.Tensor(im_data)

            _img_dataset.append(im_data)
        img_dataset = torch.stack(_img_dataset)

        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)
        cls, offset = _cls.cpu().data.numpy(), _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.7)
        for idx in idxs:
            _box = _pnet_boxes[idx]
            _x1, _y1, _x2, _y2 = map(int, _box[:4])
            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])
        return tool.NMS(np.array(boxes), 0.5)

    def __onet_detect(self, image, rnet_boxes):
        _img_dataset = []
        _rnet_boxes = tool.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1, _y1, _x2, _y2 = map(int, _box[:4])
            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48), Image.ANTIALIAS)
            im_data = np.array(img) / 255. - 0.5
            im_data = im_data.transpose([2, 0, 1])
            im_data = torch.Tensor(im_data)
            _img_dataset.append(im_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)
        cls, offset = _cls.cpu().data.numpy(), _offset.cpu().data.numpy()

        boxes = []

        idxs, _ = np.where(cls > 0.9999999)

        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1, _y1, _x2, _y2 = map(int, _box[:4])
            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        return tool.NMS(np.array(boxes), 0.7, isMin=True)

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        start_index_ = np.array(start_index, np.float32)
        _x1 = (start_index_[1] * stride) / scale
        _y1 = (start_index_[0] * stride) / scale
        _x2 = (start_index_[1] * stride + side_len) / scale
        _y2 = (start_index_[0] * stride + side_len) / scale
        w = _x2 - _x1
        h = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]
        x1 = _x1 + w * _offset[0]
        y1 = _y1 + h * _offset[1]
        x2 = _x2 + w * _offset[2]
        y2 = _y2 + h * _offset[3]
        return [x1, y1, x2, y2, cls]


if __name__ == '__main__':
    n = 0
    detecter = Detector()
    cap = cv2.VideoCapture(r"C:\Users\win10\Downloads\Video\plane.FLV")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 24
    videoWriter = cv2.VideoWriter('4.avi', fourcc, fps, (852, 480))
    while cap.isOpened():
        global w
        ret, frame = cap.read()
        # print(frame.shape)

        if ret:
            # 转换通道, 转成Image 格式099988

            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # print(frame.shape)
            # t1 = time.time()
            timer = cv2.getTickCount()
            b, g, r = cv2.split(frame)

            img = cv2.merge([r, g, b])
            img = Image.fromarray(img, "RGB")
            onet_boxes = detecter.detect(img)

            draw = ImageDraw.Draw(img)
            for box in onet_boxes:

                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255))

            cv2.imshow('img', frame)
                # draw.rectangle((box[0], box[1], box[2], box[3]), outline="red")
                # draw.text((box[0], box[1]+50), "%.4s" % str(box[4]), "green")
            # img.save('D:\pycharm_project\yxy_mtcnn\q.JPG')
            n =+ 1
            if n < 3000:
                videoWriter.write(frame)
            j = cv2.waitKey(10)

            # q键退出
            if j & 0xff == ord('q'):
                break