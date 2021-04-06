
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_tools import LabelSmoothingCrossEntropy


def plot_loss(train_loss, valid_loss):

    plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss)+1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1
    plt.axvline(minposs, linestyle='--', color='r',
                label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss)+1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plot_picture/loss_plot.png', bbox_inches='tight')
    plt.show()


def show_result(model,test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images = images.to(device)
    # get sample outputs
    output = model(images)
    # convert output probabilities to predicted class
    _, preds = torch.max(output, 1)
    # prep images for display
    images = images.cpu().numpy()

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(60, 100))
    for idx in np.arange(10):
        ax = fig.add_subplot(10, 6, idx+1, xticks=[], yticks=[])
        image = images[idx]
        min_pixel = np.min(image)
        max_pixel = np.max(image)
        image = (image - min_pixel)/(max_pixel- min_pixel)
        ax.imshow(np.transpose(image,(1,2,0)))
        ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                    color=("blue" if preds[idx]==labels[idx] else "red"))


def show_false_result(model,config,test_loader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    model.eval()  # prep model for evaluation

    num=0
    for data, target in test_loader:
        if len(target.data) != config.batch_size:
            break
        # forward pass: compute predicted outputs by passing inputs to the model
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        _, pred = torch.max(output, 1)
        result = pred == target
        index = np.where(result.cpu().numpy() == 0)[0]
        false_num = len(index)
        if false_num:
            row = false_num // 6 + 1
            fig = plt.figure(figsize=(30, 6*row))
            for idx in range(false_num):
                ax = fig.add_subplot(row, 6, idx+1, xticks=[], yticks=[])
                i = int(index[idx])
                image =np.transpose(data[i].cpu().numpy(),(1,2,0))
                min_pixel = np.min(image)
                max_pixel = np.max(image)
                image = (image - min_pixel)/(max_pixel- min_pixel)
                ax.set_title("{} ({})".format(str(pred[i].item()), str(target[i].item())),color="blue" )
                ax.imshow(image)
                num+=1
    print('false predicted number : %d'%num)

