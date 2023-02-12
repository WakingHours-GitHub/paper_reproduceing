from torch import nn
from utils


def build_targets(pred, labels, model):
    """
    匹配正样本。
    pred: yoloLayer出来的结果. 三层.  # torch.Size([1, 3, 13, 13, 25]) torch.Size([1, 3, 26, 26, 25]) torch.Size([1, 3, 52, 52, 25])
    labels: [batch_index, class. x, y, w, h]


    """

    num_labels = labels.shape[0]
    tcls, tbox, indices, av = [], [], [], [] # 初始化. 

    multi_gpu = type(model) in (nn.parallel.DataParallel,
                                nn.parallel.DistributedDataParallel)
    for i, yololayer in enumerate(model.yololayer):
        pass












