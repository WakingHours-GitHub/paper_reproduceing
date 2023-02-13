from torch import nn
from utils import *
from nets import *
from yolo_dataset import  *

import torch

IOU_threshold = 0.2

def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def build_targets(pred, targets, model):
    """
    匹配正样本。
    pred: yoloLayer出来的结果. 三层.  # torch.Size([1, 3, 13, 13, 25]) torch.Size([1, 3, 26, 26, 25]) torch.Size([1, 3, 52, 52, 25])
    labels: [batch_index, class. x, y, w, h]


    """

    num_targets = targets.shape[0]
    tcls, tbox, indices, av = [], [], [], [] # 初始化. 
    gain = torch.ones(6, device=targets.device).long()  # normalized to gridspace gain, 针对每一个目标的增益.

    # multi_gpu = type(model) in (nn.parallel.DataParallel,
    #                             nn.parallel.DistributedDataParallel)
    # yolov3.cfg中有三个yolo层，这部分用于获取对应yolo层的grid尺寸和anchor大小
    # ng 代表num of grid (13,13) anchor_vec [[x,y],[x,y]]
    # 注意这里的anchor_vec: 假如现在是yolo第一个层(downsample rate=32)
    # 这一层对应anchor为：[116, 90], [156, 198], [373, 326]
    # anchor_vec实际值为以上除以32的结果：[3.6,2.8],[4.875,6.18],[11.6,10.1]
    # 原图 416x416 对应的anchor为 [116, 90]
    # 下采样32倍后 13x13 对应的anchor为 [3.6,2.8]

    for i, yololayer in enumerate(model.yololayer):
        num_grid = yololayer.ng
        anchors = yololayer.anchor_vec
        gain[2:] = torch.tensor(pred[i].shape)[[3, 2, 3, 2]]  # xyxy gain # 取出3, 2, 3, 2索引对应的参数.
        na = anchors.shape[0]  # number of anchors
        # [3] -> [3, 1] -> [3, nt]
        anthor_target = torch.arange(na).view(na, 1).repeat(1, num_targets) # 将arange, (na, 1)复制n列.
        # 就变成(na, nt) # 每行对的是哪个anchor模板, 每列对应的是哪个target. 这样方便指认.
        # 然后我们通过iou筛选正样本, 得到逻辑矩阵, 这样我们就知道哪个target对应哪个anchor了.
        
        # 准备拆分出来. 这里target要转换为相对于这个layer大小的绝对坐标. 
        print(gain)
        a, t = [], targets*gain # 
        if num_targets: # 如果存在target的话.
            j = wh_iou(anchors, t[:, 4:6]) > IOU_threshold # 返回逻辑矩阵. 
            
            # 然后通过这个逻辑矩阵, 分别找到对应的a, 以及对应的target, 同一索引(位置上)相对应, 也就是这个target对应的anchor模板.
            a,t = anthor_target[j], t.repeat(na, 1, 1)[j]

        
        # 然后还需要知道target要对应到哪一个grid cell上. 知道位置, 
        # 分别解析: 
        b, c = t[:, :2].long().T # image_idx, classes # 转置, 两行, n列.
        g_x_y = t[:, 2:4] # 获取中心坐标








        break
    
    # # iou of targets-anchors
    # # targets中保存的是ground truth
    # t, a = targets, []
    # gwh = t[:, 4:6] * num_grid[0]

    if num_targets: # 如果存在ground true. 
        # anchor_vec: shape = [3, 2] 代表3个anchor, w, h.
        # gwh: shape = [n, 2] 代表 n个ground truth
        # iou: shape = [3, n] 代表 3个anchor与对应的n个ground truth的iou
        pass


def test_function_build_targets():
    device = torch.device("cuda:0")
    # x = torch.zeros(size=(1, 3, 416, 416)).to(device)
    yolo = YoLoBody(20).to(device)

    yolo_dataset = Yolo_VOC_dataset()
    yolo_dataloader = DataLoader(
        yolo_dataset,
        4,
        collate_fn=Yolo_VOC_dataset.collate_fn
    )

    for imgs, targets, shapes, indexs in yolo_dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        output = yolo(imgs)
        build_targets(output, targets, yolo)

        break

    

    # out13, out26, out52 = yolo(x) 
    # print(out13.shape, out26.shape, out52.shape)
    print(yolo.yololayer)




def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


if __name__ == "__main__":
    test_function_build_targets()







