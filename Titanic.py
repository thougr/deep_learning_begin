import itertools

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

from util import Accumulator, train_ch3, Animator
import torch.nn.functional as F

# 读取训练文件和测试文件
train_data = pd.read_csv("titanic/train.csv")
test_data = pd.read_csv("titanic/test.csv")
print(train_data.shape)
print(test_data.shape)
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
# 删除列 passengerId、 survive（y)
all_features = pd.concat((train_data.iloc[:, 2:-1], test_data.iloc[:, 1:]))
# all_features = train_data.iloc[:, 2:-1]
print(all_features)
# 获得数字型特征
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
print(numeric_features)
# 将其标准化
# all_features[numeric_features] = all_features[numeric_features].apply(
#     lambda x: (x - x.mean()) / (x.std()))
print(all_features)
# 空位填0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
print(all_features)
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features)
print(all_features.shape)
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
# print("train_features", train_features.shape[0], train_features)
# train_labels = torch.tensor(
#     train_data.Survived.values.reshape(-1, 1), dtype=torch.float32)
train_labels = torch.tensor(
    train_data.Survived.values.reshape(-1, 1), dtype=torch.long)
# train_labels = torch.tensor(train_data.Survived.values, dtype=torch.long)
# print("train_labels", train_labels)
# print("test_features", test_features.shape[0], test_features)
# 将train_labels转换为one-hot编码
# real_y = F.one_hot(train_labels, num_classes=2).float()
real_y = train_labels

# 检查结果
print(real_y.shape[0], real_y)

# 将 one-hot 编码转换回原始类别标签
# original_labels = torch.argmax(real_y, dim=1)

# 检查转换回的原始标签
# print("Original labels:")
# print(original_labels)

num_inputs = all_features.shape[1]
num_outputs = 2
print(num_inputs, num_outputs)

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train reiloss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        # test_acc = evaluate_accuracy(net, test_iter)
        # animator.add(epoch + 1, train_metrics + (test_acc,))
        animator.add(epoch + 1, train_metrics)
    train_loss, train_acc = train_metrics
    # assert train_loss < 0.5, train_loss
    # assert train_acc <= 1 and train_acc > 0.7, train_acc
    # assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
num_epochs = 1

train_ch3(net, zip(train_features, real_y), None, cross_entropy, num_epochs, updater)
res = net(train_features)
max_indices = torch.argmax(res, dim=1)
preds = max_indices.detach().numpy()
test_data["Survived"] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test_data['PassengerId'], test_data['Survived']], axis=1)
submission.to_csv('submission.csv', index=False)

# loss = nn.MSELoss()
# in_features = train_features.shape[1]
#
# def get_net():
#     net = nn.Sequential(nn.Linear(in_features,1))
#     return net
#
# def log_rmse(net, features, labels):
#     # 为了在取对数时进一步稳定该值，将小于1的值设置为1
#     clipped_preds = torch.clamp(net(features), 1, float('inf'))
#     rmse = torch.sqrt(loss(torch.log(clipped_preds),
#                            torch.log(labels)))
#     return rmse.item()
#
# def train(net, train_features, train_labels, test_features, test_labels,
#           num_epochs, learning_rate, weight_decay, batch_size):
#     train_ls, test_ls = [], []
#     train_iter = d2l.load_array((train_features, train_labels), batch_size)
#     # 这里使用的是Adam优化算法
#     optimizer = torch.optim.Adam(net.parameters(),
#                                  lr = learning_rate,
#                                  weight_decay = weight_decay)
#     for epoch in range(num_epochs):
#         for X, y in train_iter:
#             optimizer.zero_grad()
#             l = loss(net(X), y)
#             l.backward()
#             optimizer.step()
#         train_ls.append(log_rmse(net, train_features, train_labels))
#         if test_labels is not None:
#             test_ls.append(log_rmse(net, test_features, test_labels))
#     return train_ls, test_ls
#
# def get_k_fold_data(k, i, X, y):
#     assert k > 1
#     fold_size = X.shape[0] // k
#     X_train, y_train = None, None
#     for j in range(k):
#         idx = slice(j * fold_size, (j + 1) * fold_size)
#         X_part, y_part = X[idx, :], y[idx]
#         if j == i:
#             X_valid, y_valid = X_part, y_part
#         elif X_train is None:
#             X_train, y_train = X_part, y_part
#         else:
#             X_train = torch.cat([X_train, X_part], 0)
#             y_train = torch.cat([y_train, y_part], 0)
#     return X_train, y_train, X_valid, y_valid
#
# def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
#            batch_size):
#     train_l_sum, valid_l_sum = 0, 0
#     for i in range(k):
#         data = get_k_fold_data(k, i, X_train, y_train)
#         net = get_net()
#         train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
#                                    weight_decay, batch_size)
#         train_l_sum += train_ls[-1]
#         valid_l_sum += valid_ls[-1]
#         if i == 0:
#             d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
#                      xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
#                      legend=['train', 'valid'], yscale='log')
#         print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
#               f'验证log rmse{float(valid_ls[-1]):f}')
#     return train_l_sum / k, valid_l_sum / k
#
# k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
#                           weight_decay, batch_size)
# print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
#       f'平均验证log rmse: {float(valid_l):f}')