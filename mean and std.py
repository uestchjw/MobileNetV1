import numpy as np
import os
from skimage import io


# 计算数据集的均值和方差,用作transforms.Normalize的参数
def mean_std(path):
    files = os.listdir(path)
    img_files = []
    for i in files:
        img_files.append(path+'/'+i)
    means = [0.,0.,0.]
    stds = [0.,0.,0.]
    for img_path in img_files:
        img = io.imread(img_path).astype('float32')/255.0
        means[0] += img[:,:,0].reshape(1,-1).mean()
        means[1] += img[:,:,1].reshape(1,-1).mean()
        means[2] += img[:,:,2].reshape(1,-1).mean()
        stds[0] += img[:,:,0].reshape(1,-1).std()
        stds[1] += img[:,:,1].reshape(1,-1).std()
        stds[2] += img[:,:,2].reshape(1,-1).std()
    means[0] /= len(files)
    means[1] /= len(files)
    means[2] /= len(files)
    stds[0] /= len(files)
    stds[1] /= len(files)
    stds[2] /= len(files)
    return means,stds
