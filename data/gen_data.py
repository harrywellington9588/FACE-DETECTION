import os
from PIL import Image
import numpy as np
import traceback
from my_tool import tool

annotation_path = 'E:\plane/label.txt'   # anno_src
image_path = 'E:\plane\positive'                       #img_dir
save_path ='E:\plane/train'

for face_size in [12, 24, 48]:
    print("generate {} sample".format(face_size))
    # 样本图片存储路径
    positive_image_direction = os.path.join(save_path, str(face_size), 'positive')
    negative_image_direction = os.path.join(save_path, str(face_size), 'negative')
    part_image_direction = os.path.join(save_path, str(face_size), 'part')

    for direction_path in [positive_image_direction, negative_image_direction, part_image_direction]:
        if not os.path.exists(direction_path):
            os.makedirs(direction_path)

    # 样本描述(label)存储路径
    positive_anno_direction = os.path.join(save_path, str(face_size), 'positive.txt')
    negative_anno_direction = os.path.join(save_path, str(face_size), 'negative.txt')
    part_anno_direction = os.path.join(save_path, str(face_size), 'part.txt')

    # 样本图片计数
    positive_image_count = 0
    negative_image_count = 0
    part_image_count = 0

    try:
        positive_anno_file = open(positive_anno_direction, 'w')
        negative_anno_file = open(negative_anno_direction, 'w')
        part_anno_file = open(part_anno_direction, 'w')

        for i, line in enumerate(open(annotation_path)):
            # if i > 2:
            #     continue
            try:
                strs = line.strip().split(",")
                strs = list(filter(lambda x: bool(x), strs))
                image_filename = strs[0]
                print(image_filename)
                image_file_path = os.path.join(image_path, image_filename)

                with Image.open(image_file_path) as img:
                    image_width, image_height = img.size
                    x1 = float(strs[1])
                    y1 = float(strs[2])
                    x2 = float(strs[3])
                    y2 = float(strs[4])
                    w = float(x2 - x1)
                    h = float(y2 - y1)

                    if x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue
                    boxes = [[x1, y1, x2, y2]]
                    # print(boxes)
                    # 计算人脸中心点坐标
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    # 使正样本和部分样本数量翻倍
                    for i in range(10):
                        # 让人脸中点的有少许偏移
                        w_ = np.random.randint(-w * 0.1, w * 0.1)
                        h_ = np.random.randint(-h * 0.1, h * 0.1)
                        cx_ = cx + w_
                        cy_ = cy + h_

                        # 让人脸形成正方形,并且让坐标也有少许的偏移
                        side_len = np.random.randint(int(min(w, h) * 0.95), np.ceil(1.01 * max(w, h)))
                        x1_ = np.max(cx_ - side_len / 2, 0)
                        y1_ = np.max(cy_ - side_len / 2, 0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        crop_box = np.array([x1_, y1_, x2_, y2_])

                        # 计算坐标的偏移值
                        offset_x1 = (x1 - x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        # 剪切下图片,并进行大小缩放
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                        iou_value = tool.iou(crop_box, np.array(boxes))
                        # print(crop_box)
                        # print(boxes)

                        # 正样本
                        if iou_value > 0.65:
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                    positive_image_count, 1, offset_x1, offset_y1, offset_x2, offset_y2
                                )
                            )
                            positive_anno_file.flush()
                            # face_resize.show()
                            # break
                            face_resize.save(os.path.join(positive_image_direction, "{}.jpg".format(
                                positive_image_count
                            )))
                            positive_image_count += 1
                        # 部分样本
                        elif iou_value > 0.4:
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                    part_image_count, 2, offset_x1, offset_y1, offset_x2, offset_y2
                                )
                            )

                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_direction, "{}.jpg".format(
                                part_image_count
                            )))
                            part_image_count += 1
                        elif iou_value < 0.3:
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0\n".format(negative_image_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_direction, "{}.jpg".format(
                                negative_image_count
                            )))
                            negative_image_count += 1

                    # 生成负样本
                    _boxes = np.array(boxes)

                    for j in range(10):
                        side_len = np.random.randint(face_size, min(image_width, image_height) / 2)
                        x_ = np.random.randint(0, image_width - side_len)
                        y_ = np.random.randint(0, image_height - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                        if np.max(tool.iou(crop_box, _boxes)) < 0.3:
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size,face_size), Image.ANTIALIAS)

                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0\n".format(negative_image_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_direction,
                                                          "{}.jpg".format(negative_image_count)))
                            negative_image_count += 1

            except Exception as e:
                traceback.print_exc()
    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()
