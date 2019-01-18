import os
from PIL import Image
import numpy as np
import traceback

annotation_path = 'E:\celeba\Anno\list_bbox_celeba.txt'
image_path = 'E:\celeba\img_celeba'
save_path = 'E:\celeba\sample'

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
            if i < 2:
                continue
            if i < 4:
                process = line.strip().split(" ")
                print(process)
    except:
        traceback.print_exc()

