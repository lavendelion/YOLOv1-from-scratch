import torch
import torch.nn as nn
import torchvision.models as tvmodel
from prepare_data import GL_CLASSES, GL_NUMBBOX, GL_NUMGRID
from util import calculate_iou


class MyNet(nn.Module):
    """
    @ 网络实际名称
    为了和后续接口对齐，此处类名固定为MyNet，具体是什么网络可以写在注释里。
    """
    def __init__(self):
        """
        :param args: 构建网络所需要的参数

        TODO:
        在__init__()函数里，将网络框架搭好，并存在self里
        """
        super(MyNet, self).__init__()
        resnet = tvmodel.resnet34(pretrained=True)  # 调用torchvision里的resnet34预训练模型
        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的最后两层
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
        )
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            nn.Linear(GL_NUMGRID * GL_NUMGRID * 1024, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, GL_NUMGRID * GL_NUMGRID * (5*GL_NUMBBOX+len(GL_CLASSES))),
            nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )


    def forward(self, inputs):
        """
        :param inputs:  输入网络的张量
        :return:  输出网络的结果

        TODO
        根据网络的结构，完成网络的前向传播计算。
        如果网络有多条分支，可以用self储存需要在别的地方使用的中间张量。
        如果网络有多个输出，需要将多个输出按后续inference的需求打包输出
        """
        x = self.resnet(inputs)
        x = self.Conv_layers(x)
        x = x.view(x.size()[0], -1)
        x = self.Conn_layers(x)
        self.pred = x.reshape(-1, (5 * GL_NUMBBOX + len(GL_CLASSES)), GL_NUMGRID, GL_NUMGRID)  # 记住最后要reshape一下输出数据
        return self.pred


    def calculate_loss(self, labels):
        """
        TODO: 根据labels和self.outputs计算训练loss
        :param labels: (bs, n), 对应训练数据的样本标签
        :return: loss数值
        """
        self.pred = self.pred.double()
        labels = labels.double()
        num_gridx, num_gridy = GL_NUMGRID, GL_NUMGRID  # 划分网格数量
        noobj_confi_loss = 0.  # 不含目标的网格损失(只有置信度损失)
        coor_loss = 0.  # 含有目标的bbox的坐标损失
        obj_confi_loss = 0.  # 含有目标的bbox的置信度损失
        class_loss = 0.  # 含有目标的网格的类别损失
        n_batch = labels.size()[0]  # batchsize的大小

        # 可以考虑用矩阵运算进行优化，提高速度，为了准确起见，这里还是用循环
        for i in range(n_batch):  # batchsize循环
            for n in range(num_gridx):  # x方向网格循环
                for m in range(num_gridy):  # y方向网格循环
                    if labels[i, 4, m, n] == 1:  # 如果包含物体
                        # 将数据(px,py,w,h)转换为(x1,y1,x2,y2)
                        # 先将px,py转换为cx,cy，即相对网格的位置转换为标准化后实际的bbox中心位置cx,xy
                        # 然后再利用(cx-w/2,cy-h/2,cx+w/2,cy+h/2)转换为xyxy形式，用于计算iou
                        bbox1_pred_xyxy = ((self.pred[i, 0, m, n] + n) / num_gridx - self.pred[i, 2, m, n] / 2,
                                           (self.pred[i, 1, m, n] + m) / num_gridy - self.pred[i, 3, m, n] / 2,
                                           (self.pred[i, 0, m, n] + n) / num_gridx + self.pred[i, 2, m, n] / 2,
                                           (self.pred[i, 1, m, n] + m) / num_gridy + self.pred[i, 3, m, n] / 2)
                        bbox2_pred_xyxy = ((self.pred[i, 5, m, n] + n) / num_gridx - self.pred[i, 7, m, n] / 2,
                                           (self.pred[i, 6, m, n] + m) / num_gridy - self.pred[i, 8, m, n] / 2,
                                           (self.pred[i, 5, m, n] + n) / num_gridx + self.pred[i, 7, m, n] / 2,
                                           (self.pred[i, 6, m, n] + m) / num_gridy + self.pred[i, 8, m, n] / 2)
                        bbox_gt_xyxy = ((labels[i, 0, m, n] + n) / num_gridx - labels[i, 2, m, n] / 2,
                                        (labels[i, 1, m, n] + m) / num_gridy - labels[i, 3, m, n] / 2,
                                        (labels[i, 0, m, n] + n) / num_gridx + labels[i, 2, m, n] / 2,
                                        (labels[i, 1, m, n] + m) / num_gridy + labels[i, 3, m, n] / 2)
                        iou1 = calculate_iou(bbox1_pred_xyxy, bbox_gt_xyxy)
                        iou2 = calculate_iou(bbox2_pred_xyxy, bbox_gt_xyxy)
                        # 选择iou大的bbox作为负责物体
                        if iou1 >= iou2:
                            coor_loss = coor_loss + 5 * (torch.sum((self.pred[i, 0:2, m, n] - labels[i, 0:2, m, n]) ** 2) \
                                        + torch.sum((self.pred[i, 2:4, m, n].sqrt() - labels[i, 2:4, m, n].sqrt()) ** 2))
                            obj_confi_loss = obj_confi_loss + (self.pred[i, 4, m, n] - iou1) ** 2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou2
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((self.pred[i, 9, m, n] - iou2) ** 2)
                        else:
                            coor_loss = coor_loss + 5 * (torch.sum((self.pred[i, 5:7, m, n] - labels[i, 5:7, m, n]) ** 2) \
                                        + torch.sum((self.pred[i, 7:9, m, n].sqrt() - labels[i, 7:9, m, n].sqrt()) ** 2))
                            obj_confi_loss = obj_confi_loss + (self.pred[i, 9, m, n] - iou2) ** 2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中,注意，对于标签的置信度应该是iou1
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((self.pred[i, 4, m, n] - iou1) ** 2)
                        class_loss = class_loss + torch.sum((self.pred[i, 10:, m, n] - labels[i, 10:, m, n]) ** 2)
                    else:  # 如果不包含物体
                        noobj_confi_loss = noobj_confi_loss + 0.5 * torch.sum(self.pred[i, [4, 9], m, n] ** 2)

        loss = coor_loss + obj_confi_loss + noobj_confi_loss + class_loss
        # 此处可以写代码验证一下loss的大致计算是否正确，这个要验证起来比较麻烦，比较简洁的办法是，将输入的pred置为全1矩阵，再进行误差检查，会直观很多。
        return loss / n_batch

    def calculate_metric(self, preds, labels):
        """
        TODO: 根据preds和labels，以及指定的评价方法计算网络效果得分， 网络validation时使用
        :param preds: 预测数据
        :param labels: 预测数据对应的样本标签
        :return: 评估得分metric
        """
        preds = preds.double()
        labels = labels[:, :(self.n_points*2)]
        l2_distance = torch.mean(torch.sum((preds-labels)**2, dim=1))
        return l2_distance


if __name__ == '__main__':
    # 自定义输入张量，验证网络可以正常跑通，并计算loss，调试用
    x = torch.zeros(5,3,448,448)
    net = MyNet()
    a = net(x)
    labels = torch.zeros(5, 30, 7, 7)
    loss = net.calculate_loss(labels)
    print(loss)
    print(a.shape)
