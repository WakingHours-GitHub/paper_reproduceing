from torch import nn
from utils import *
from nets import *
from yolo_dataset import  *
import math

import torch
from torch import optim
import time


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
    tcls, tbox, indices, anch = [], [], [], [] # 初始化. 
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
        # print(gain)
        a, t = [], targets*gain # 
        if num_targets: # 如果存在target的话.
            j = wh_iou(anchors, t[:, 4:6]) > IOU_threshold # 返回逻辑矩阵. 
            
            # 然后通过这个逻辑矩阵, 分别找到对应的a, 以及对应的target, 同一索引(位置上)相对应, 也就是这个target对应的anchor模板.
            a,t = anthor_target[j], t.repeat(na, 1, 1)[j]
            # print(t)

        
        # 然后还需要知道target要对应到哪一个grid cell上. 知道位置, 
        # 分别解析: 
        b, c = t[:, :2].long().T # image_idx, classes # 转置, 两行, n列.
        g_x_y = t[:, 2:4] # 获取中心坐标
        g_w_h = t[:, 4: 6] # 获取所有目标框的wh.
        g_i_j = g_x_y.long() # gt的中心坐标。向下取整, 就是左上角边界框的坐标. 

        gi, gj = g_i_j.T # 同样, 转置, 一行代表x坐标, 一行代表j坐标. 

        # 遍历一层, 将一层的信息添加进去. 
        # .clamp()就是限制在上下界. 
        indices.append((b, a, gj.clamp_(0, gain[3]-1), gi.clamp_(0, gain[2]-1)))
        # b: 那张图片?
        # a: 所有正样本对应的anchor模板是几号.
        # gj: 对应anchor中心点的y坐标 然后限制到[0, h]
        # gi: 对应anchor中心点的x坐标. 然后限制到0和宽. [0, w]

        tbox.append(torch.cat((g_x_y - g_i_j, g_w_h), 1)) # ???什么玩应???
        # 偏移量. gt相对于anchor的偏移量, 一个是float, 一个是int后的, 以及gt的w,h

        anch.append(anchors[a]) # 对应的anchor的w, h.


        tcls.append(c) # class
        # 每个正样本对应的类别. 

        if c.shape[0]: # 对于任何的target
            assert c.max() < model.num_classes,       'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
            'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (model.nc, model.nc - 1, c.max())

    # 返回
    return tcls, tbox, indices, anch
    
    # # # iou of targets-anchors
    # # # targets中保存的是ground truth
    # # t, a = targets, []
    # # gwh = t[:, 4:6] * num_grid[0]

    # if num_targets: # 如果存在ground true. 
    #     # anchor_vec: shape = [3, 2] 代表3个anchor, w, h.
    #     # gwh: shape = [n, 2] 代表 n个ground truth
    #     # iou: shape = [3, n] 代表 3个anchor与对应的n个ground truth的iou
    #     pass




def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou

def compute_loss(p, targets, model):
    device = p[0].device # 获取设备。
    lcls = torch.zeros(1, device=device)  # Tensor(0) # 分类损失
    lbox = torch.zeros(1, device=device)  # Tensor(0) # 定位损失
    lobj = torch.zeros(1, device=device)  # Tensor(0) # 置信度损失.

    # 通过build_target计算所有正样本, 然后将结果保存过来
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets  # 计算所有的正样本.
    # h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)

    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device), reduction=red) # 针对分类损失. 
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device), reduction=red) # obj损失. 
    
    cp, cn = smooth_BCE(eps=0.0) # 平滑,  1.0, 0.

    # per output
    for i, pi in enumerate(p):  # layer index, layer predictions # 遍历每一层的输出. 
        b, a, gj, gi = indices[i]  # image_idx, anchor_idx, grid_y, grid_x
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        nb = b.shape[0]  # number of positive samples
        if nb:
            # 对应匹配到正样本的预测信息
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # GIoU
            pxy = ps[:, :2].sigmoid()
            pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
            lbox += (1.0 - giou).mean()  # giou loss

            # Obj
            tobj[b, a, gj, gi] = (1.0 - 1.0) + 1.0 * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Class
            if model.num_classes > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(nb), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    lbox *= 3.54 # h['giou']
    lobj *= 64.3  # h['obj']
    lcls *= 37.4 #  h['cls']

    # loss = lbox + lobj + lcls
    return {"box_loss": lbox,
            "obj_loss": lobj,
            "class_loss": lcls}



def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def test_train():
    Epoch = 100
    device = torch.device("cuda:0")

    yolo_dataset = Yolo_VOC_dataset()
    yolo_dataloader = DataLoader(
        yolo_dataset,
        10,
        collate_fn=Yolo_VOC_dataset.collate_fn,
        drop_last=True
    )

    net = YoLoBody(20).to(device)

    trainer = optim.SGD(net.parameters(), 0.001, 0.937, weight_decay=0.0005)

    for epoch in range(Epoch):
        for i, (imgs, targets, shapes, indexs) in enumerate(yolo_dataloader):

            trainer.zero_grad()
            imgs, targets = imgs.to(device), targets.to(device)

            pred = net(imgs)
            loss_dict = compute_loss(pred, targets, net) # 计算损失.
            # 传入pred, labels, 以及模型.
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()

            trainer.step()
            print(i, losses)

        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), f"../runs/{epoch+1}_loss_{int(losses.item())}.pth")

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6,
                        multi_label=True, classes=None, agnostic=False, max_num=100):
    """
    Performs  Non-Maximum Suppression on inference results

    param: prediction[batch, num_anchors, (num_classes+1+4) x num_anchors]
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    merge = False  # merge for best mAP
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference 遍历每张图片
        # Apply constraints
        x = x[x[:, 4] > conf_thres]  # confidence 根据obj confidence虑除背景目标
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height 虑除小目标

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:  # 针对每个类别执行非极大值抑制
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only  直接针对每个类别中概率最大的类别进行非极大值抑制处理
            conf, j = x[:, 5:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_num]  # 最多只保留前max_num个目标信息
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                # i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output
def decode(pred, net):
    pred = non_max_suppression(pred)



def test_prediction():
    device = torch.device("cuda:0")

    yolo_dataset = Yolo_VOC_dataset()
    image, target, shape, index = yolo_dataset[-203]
    image = torch.unsqueeze(image, 0).float().to(device)
    print(image.shape)
    
    net = YoLoBody(20).to(device)
    net.load_state_dict(torch.load("../runs/30_loss_16.pth"))
    net.eval()
    with torch.no_grad():
        pred = net(image) # ???????为什么要【0】
        print("pred.shape", pred[0].shape)

        # decode(pred, net)



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
    # print(yolo.yololayer) # 就是三个yololayer层。 




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
    # test_function_build_targets()
    # test_train()
    test_prediction()







