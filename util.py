"""
该文件里提供一些项目需要的功能性函数/类
"""
import torch
import torch.nn as nn
import numpy as np


class DepthwiseConv(nn.Module):
    """
    深度可分离卷积层
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True):
        super(DepthwiseConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=in_channels, bias=bias),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class InvertedBottleneck(nn.Module):
    """
    MobileNet v2 的InvertedBottleneck
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, t_factor, padding=0, bias=True):
        super(InvertedBottleneck, self).__init__()
        mid_channels = t_factor*in_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=bias)
        )

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs

class Flatten(nn.Module):
    """
    将三维张量拉平的网络层
    (n,c,h,w) -> (n, c*h*w)
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        n_samples = x.shape[0]
        x = x.reshape(n_samples, -1)
        return x


def calculate_iou(bbox1, bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    if bbox1[2]<=bbox1[0] or bbox1[3]<=bbox1[1] or bbox2[2]<=bbox2[0] or bbox2[3]<=bbox2[1]:
        return 0  # 如果bbox1或bbox2没有面积，或者输入错误，直接返回0

    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的重合区域的(x1,y1,x2,y2)

    intersect_bbox[0] = max(bbox1[0],bbox2[0])
    intersect_bbox[1] = max(bbox1[1],bbox2[1])
    intersect_bbox[2] = min(bbox1[2],bbox2[2])
    intersect_bbox[3] = min(bbox1[3],bbox2[3])

    w = max(intersect_bbox[2] - intersect_bbox[0], 0)
    h = max(intersect_bbox[3] - intersect_bbox[1], 0)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = w * h  # 交集面积
    iou = area_intersect / (area1 + area2 - area_intersect + 1e-6)  # 防止除0
    # print(bbox1,bbox2)
    # print(intersect_bbox)
    # input()
    return iou


# 注意检查一下输入数据的格式，到底是xywh还是xyxy
def labels2bbox(matrix):
    """
    将网络输出的7*7*30的数据转换为bbox的(98,25)的格式，然后再将NMS处理后的结果返回
    :param matrix: 注意，输入的数据中，bbox坐标的格式是(px,py,w,h)，需要转换为(x1,y1,x2,y2)的格式再输入NMS
    :return: 返回NMS处理后的结果,bboxes.shape = (-1, 6), 0:4是(x1,y1,x2,y2), 4是conf， 5是cls
    """
    if matrix.size()[0:2]!=(7,7):
        raise ValueError("Error: Wrong labels size: ", matrix.size(), " != (7,7)")
    matrix = matrix.numpy()
    bboxes = np.zeros((98, 6))
    # 先把7*7*30的数据转变为bbox的(98,25)的格式，其中，bbox信息格式从(px,py,w,h)转换为(x1,y1,x2,y2),方便计算iou
    matrix = matrix.reshape(49,-1)
    bbox = matrix[:, :10].reshape(98, 5)
    r_grid = np.array(list(range(7)))
    r_grid = np.repeat(r_grid, repeats=14, axis=0)  # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ...]
    c_grid = np.array(list(range(7)))
    c_grid = np.repeat(c_grid, repeats=2, axis=0)[np.newaxis, :]
    c_grid = np.repeat(c_grid, repeats=7, axis=0).reshape(-1)  # [0 0 1 1 2 2 3 3 4 4 5 5 6 6 0 0 1 1 2 2 3 3 4 4 5 5 6 6...]
    bboxes[:, 0] = np.maximum((bbox[:, 0] + c_grid) / 7.0 - bbox[:, 2] / 2.0, 0)
    bboxes[:, 1] = np.maximum((bbox[:, 1] + r_grid) / 7.0 - bbox[:, 3] / 2.0, 0)
    bboxes[:, 2] = np.minimum((bbox[:, 0] + c_grid) / 7.0 + bbox[:, 2] / 2.0, 1)
    bboxes[:, 3] = np.minimum((bbox[:, 1] + r_grid) / 7.0 + bbox[:, 3] / 2.0, 1)
    bboxes[:, 4] = bbox[:, 4]
    cls = np.argmax(matrix[:, 10:], axis=1)
    cls = np.repeat(cls, repeats=2, axis=0)
    bboxes[:, 5] = cls
    # 对所有98个bbox执行NMS算法，清理cls-specific confidence score较低以及iou重合度过高的bbox
    keepid = nms_multi_cls(bboxes, thresh=0.1, n_cls=20)
    ids = []
    for x in keepid:
        ids = ids + list(x)
    ids = sorted(ids)
    return bboxes[ids, :]


def nms_1cls(dets, thresh):
    """
    单类别NMS
    :param dets: ndarray,nx5,dets[i,0:4]分别是bbox坐标；dets[i,4]是置信度score
    :param thresh: NMS算法设置的iou阈值
    """
    # 从检测结果dets中获得x1,y1,x2,y2和scores的值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 计算每个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照置信度score的值降序排序的下标序列
    order = scores.argsort()[::-1]

    # keep用来保存最后保留的检测框的下标
    keep = []
    while order.size > 0:
        # 当前置信度最高bbox的index
        i = order[0]
        # 添加当前剩余检测框中得分最高的index到keep中
        keep.append(i)
        # 得到此bbox和剩余其他bbox的相交区域，左上角和右下角
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交的面积，不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IoU：重叠面积/(面积1+面积2-重叠面积)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的bbox
        inds = np.where(iou <= thresh)[0]
        order = order[inds+1]
    return keep


def nms_multi_cls(dets, thresh, n_cls):
    """
    多类别的NMS算法
    :param dets:ndarray,nx6,dets[i,0:4]是bbox坐标；dets[i,4]是置信度score；dets[i,5]是类别序号；
    :param thresh: NMS算法的阈值；
    :param n_cls: 是类别总数
    """
    # 储存结果的列表，keeps_index[i]表示第i类保留下来的bbox下标list
    keeps_index = []
    for i in range(n_cls):
        order_i = np.where(dets[:,5]==i)[0]
        det = dets[dets[:, 5] == i, 0:5]
        if det.shape[0] == 0:
            keeps_index.append([])
            continue
        keep = nms_1cls(det, thresh)
        keeps_index.append(order_i[keep])
    return keeps_index


if __name__ == '__main__':
    a = torch.randn((7,7,30))
    print(a)
    labels2bbox(a)