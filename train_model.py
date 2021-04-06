
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from pytorch_tools import EarlyStopping,LabelSmoothingCrossEntropy


def train_accuracy(model, train_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            img, labels = data
            img = img.to(device)
            labels = labels.to(device)
            model = model.to(device)
            out = model(img)
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print('Accuracy of the network on the train image: %d %%' %
          (100 * correct / total))
    return 100.0 * correct / total


def validate_accuracy(model, validate_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():  # 强制之后的计算没有图构建，简单来说就是没有梯度，不影响结果
        for data in validate_loader:
            img, labels = data
            img = img.to(device)
            labels = labels.to(device)
            model = model.to(device)
            out = model(img)
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print('Accuracy of the network on the validate image: %d %%' %
          (100 * correct / total))
    return 100.0 * correct / total

save_path='C:/Users/haojiuwu/Desktop/model_paras.pth'
def train_model(model, config, train_loader, validate_loader,save_path):

    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    optimizer = optim.SGD(model.parameters(), lr=config['LR'],momentum=0.9, weight_decay=config['weight_decay'])
    # optimizer = optim.Adam(model.parameters(), lr=config.LR, betas=(0.9, 0.99),weight_decay=config.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=0.5)
    # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.005, step_size_up=2000,\
    # step_size_down=2000, mode='triangular', gamma=1.0, scale_fn=None,\
    # scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)


    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=config['patience'], verbose=True)
    n_epochs = config['n_epoches']
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch, (data, target) in enumerate(train_loader, 1):
            # clear the gradients of all optimized variables
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            model = model.to(device)
            output = model(data)
            loss = criterion(output, target)
            if batch % 50 == 0:
                print(batch, ':', loss.item())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            train_losses.append(loss.item())

        scheduler.step()
        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, target in validate_loader:
            data = data.to(device)
            model = model.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            target = target.to(device)
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        if epoch % 5 == 0:

            train_acc = train_accuracy(model,train_loader)
            validate_acc = validate_accuracy(model,validate_loader)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    torch.save(model.state_dict(), save_path)

    return model, avg_train_losses, avg_valid_losses
