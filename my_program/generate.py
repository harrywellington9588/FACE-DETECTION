import os
import traceback
from PIL import Image
import numpy as np
from my_program import tools


class gen_samle():
    def __init__(self, img_path, anno_path, save_path, img_size):
        self.img_path = img_path
        self.anno_path = anno_path
        self.save_path = save_path
        self.img_size = img_size

    def gen_data(self):
        positive_img_path = os.path.join(self.save_path, str(self.img_size), "positive")
        negative_img_path = os.path.join(self.save_path, str(self.img_size), "negative")
        part_img_path = os.path.join(self.save_path, str(self.img_size), "part")
        for path in [positive_img_path, negative_img_path, part_img_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        positive_count = 0
        negative_count = 0
        part_count = 0
        try:
            positive_label = open(os.path.join(self.save_path, str(self.img_size), "positive.txt"), "w")
            negative_label = open(os.path.join(self.save_path, str(self.img_size), "negative.txt"), "w")
            part_label = open(os.path.join(self.save_path, str(self.img_size), "part.txt"), "w")

            for i, lines in enumerate(open(self.anno_path).readlines()):
                if i < 2:
                    continue
                try:
                    line = lines.strip().split(" ")
                    line = list(filter(lambda x: bool(x), line))

                    image_name = line[0]
                    print(image_name)
                    img = Image.open(os.path.join(self.img_path, image_name))
                    width, height = img.size
                    x1 = float(line[1])
                    y1 = float(line[2])
                    w = float(line[3])
                    h = float(line[4])
                    x2 = x1 + w
                    y2 = y1 + h
                    if x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    box = np.array([int(x1), int(y1), int(x2), int(y2)])
                    side_len_1 = np.minimum(w, h)
                    side_len_2 = np.maximum(w, h)

                    for _ in range(5):
                        move_w = np.random.randint(int(-w*0.2), int(w*0.2))
                        move_h = np.random.randint(int(-h*0.2), int(h*0.2))
                        cx_ = cx + move_w
                        cy_ = cy + move_h

                        side_len_ = np.random.randint(int(side_len_1*0.85), int(side_len_2*1.15))

                        x1_ = np.maximum((cx_ - side_len_ / 2), 0)
                        y1_ = np.maximum((cy_ - side_len_ / 2), 0)
                        x2_ = cx_ + side_len_ / 2
                        y2_ = cy_ + side_len_ / 2

                        x1_offset = (x1 - x1_) / side_len_
                        y1_offset = (y1 - y1_) / side_len_
                        x2_offset = (x2 - x2_) / side_len_
                        y2_offset = (y2 - y2_) / side_len_

                        boxes = np.array([[int(x1_), int(y1_), int(x2_), int(y2_)]])
                        iou = tools.iou(box, boxes)

                        crop_img = img.crop(boxes[0])
                        crop_img = crop_img.resize((self.img_size, self.img_size), Image.ANTIALIAS)

                        if iou > 0.65:
                            positive_label.write("positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                positive_count, x1_offset, y1_offset, x2_offset, y2_offset, 1))
                            positive_label.flush()
                            crop_img.save(os.path.join(positive_img_path, "{}.jpg".format(positive_count)))
                            positive_count += 1

                        elif iou > 0.4:
                            part_label.write("part/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                positive_count, x1_offset, y1_offset, x2_offset, y2_offset, 2))
                            part_label.flush()
                            crop_img.save(os.path.join(part_img_path, "{}.jpg".format(part_count)))
                            part_count += 1

                        elif iou < 0.3:
                            negative_label.write("negative/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                negative_count, x1_offset, y1_offset, x2_offset, y2_offset, 0))
                            positive_label.flush()
                            crop_img.save(os.path.join(negative_img_path, "{}.jpg".format(negative_count)))
                            negative_count += 1

                    for _ in range(5):
                        _x1 = np.random.randint(0, np.maximum(int(width - side_len_), 1))
                        _y1 = np.random.randint(0, np.maximum(int(height - side_len_), 1))
                        _x2 = _x1 + side_len_
                        _y2 = _y1 + side_len_
                        _x1_offset = (_x1 - x1) / side_len_
                        _y1_offset = (_y1 - y1) / side_len_
                        _x2_offset = (_x2 - x2) / side_len_
                        _y2_offset = (_y2 - y2) / side_len_

                        n_boxes = np.array([[_x1, _y1, _x2, _y2]])
                        n_crop_img = img.crop(n_boxes[0])
                        n_crop_img = n_crop_img.resize((self.img_size, self.img_size), Image.ANTIALIAS)
                        iou = tools.iou(box, n_boxes)
                        if iou < 0.3:
                            negative_label.write("negative/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                negative_count, _x1_offset, _y1_offset, _x2_offset, _y2_offset, 0))
                            positive_label.flush()
                            n_crop_img.save(os.path.join(negative_img_path, "{}.jpg".format(negative_count)))
                            negative_count += 1

                except Exception as e:
                    traceback.print_exc()
        finally:
            positive_label.close()
            negative_label.close()
            part_label.close()


if __name__ == '__main__':
    img_path = r"E:\celeba\rewrite"
    anno_path = r"E:\celeba/re.txt"
    save_path = r"E:\celeba\train_dataset"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for img_size in [12, 24, 48]:
        program = gen_samle(img_path, anno_path, save_path, img_size)
        program.gen_data()








