from  PIL import Image,ImageDraw
import  numpy as np

# image_path = 'E:\plane\positive/apic0.jpg'



# x1 = 95
# y1 = 71
# w = 226
# h = 313
#
# cx = x1 + w / 2
# cy = y1 + h / 2
# w_ = np.random.randint(-w * 0.2, w * 0.2)
# h_ = np.random.randint(-h * 0.2, h * 0.2)
# cx_ = cx + w_
# cy_ = cy + h_
#
# # 让人脸形成正方形，并且让坐标也有少许的偏离
# side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
# x1_ = np.max(cx_ - side_len / 2, 0)
# y1_ = np.max(cy_ - side_len / 2, 0)
# x2_ = x1_ + side_len
# y2_ = y1_ + side_len
# image_path = r"E:\celeba\smallset/000002.jpg"
# img = Image.open(image_path)
#
# w,h = img.size
# print(w)
# print(h)

# draw = ImageDraw.Draw(img)
# draw.rectangle((95,71,226,313), outline='red')
# # draw.rectangle((x1_, y1_, x2_, y2_), outline='red')
# img.show()

box = np.array([1,2,3,4])



a = box[-1]
print(a)

# a = 9
#
# if a < 10 and a >6:
#     print(a)
