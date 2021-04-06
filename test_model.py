

# 取出一张图片来，看看模型的预测结果对不对
import torch
import torch.nn as nn
from models.MobileNetV1 import *
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


path = 'D:/创意组线下赛/learning_vscode/.vscode/model_paras1.pth'
model = MobileNetV1(3,2)
model.cuda()
model.load_state_dict(torch.load(path))
model.eval()

path2='D:/kaggle_CatsVsDogs/validation/dog.7501.jpg'

transform=transforms.Compose([
    # transforms.RandomHorizontalFlip(),   训练集还是要做下数据增强，验证集不用
    # transforms.RandomRotation(60),
    transforms.Resize((224,224)),  # resize是压缩，即图片只是减小了，不会丢失某部分
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.488, 0.455, 0.417), std=(0.229, 0.225, 0.225)) # 这几个值自己写函数算过，没问题
    ])

# 显示图片
sample = Image.open(path2)
plt.show()
plt.imshow(sample)


# transforms应用于Image.open类型的数据
# unsqueeze(0)是把tensor从(3,224,224)增加维度，变成(1,3,224,224)模型输入是4维的
img = transform(sample).unsqueeze(0).cuda()
out = model(img)
if out[0,0] > out[0,1]:
    print('It''s a cat!')
else:
    print('It''s a dog!')

