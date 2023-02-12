import json
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
from pathlib import Path
import random
from lxml import etree
import numpy as np
import cv2 as cv
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

join = lambda *args: os.path.join(*args)
runtime_path = str(Path(sys.path[0]))
print("Project run on:", runtime_path)
os.chdir(runtime_path)

# FILE = Path(__file__).resolve() # 当前文件的路径.
# ROOT = FILE.parents[0]  # YOLOv5 root directory # 根路径.
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH # 添加进来. 方便导入后面的自己制作的包.
#     # 这是一个小trick
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class Yolo_VOC_dataset(Dataset):
    def __init__(
        self,
        path="../files/data.json",
        is_train=True, 
        img_size=416, # 最大尺寸. 
        batch_size=4, 
        augment=False, # augment_hsv, hsv色彩增强.
        hyp=None,
        rect=False,  # 是否使用rectangular training # 是否使用

    ) -> None:


        try:
            path = str(Path(path)) # normalization path.
            if os.path.isfile(path):
                with open(path) as f:
                    data_dict = json.loads(f.read())
        except FileExistsError:
            raise Exception("%s does not exist" % path)
        
        data_txt_path = data_dict["train_path"] if is_train else data_dict["val_path"]
        with open(data_txt_path, "r") as f:
            file_name_list = f.readlines()
        n = len(file_name_list)
        assert n > 0, "No images found in %s." % (path)

        bi = (np.floor(np.arange(n) / batch_size)).astype(np.int) # 类似于一个蒙板数据 -> mask 
        # print(bi) # [  0   0   0  1  1  1  1...] # 根据batch_size生成模板。
        # 记录数据集划分后的总batch数
        nb = bi[-1] + 1  # number of batches # 总的batch数目.

        # 参数赋值. 
        self.n = n  # number of images 图像总数目
        self.batch = bi  # batch index of image 记录哪些图片属于哪个batch
        self.img_size = img_size  # 这里设置的是预处理后输出的图片尺寸
        self.augment = augment  # 是否启用augment_hsv
        self.hyp = hyp  # 超参数字典，其中包含图像增强会使用到的超参数
        self.rect = rect  # 是否使用rectangular training
        # 注意: 开启rect后，mosaic就默认关闭DA
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.transforme = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.img_size, self.img_size)),
            ])

        with open(data_dict["classes"]) as f:
            classes = [ele.strip() for ele in f.readlines()]
        self.classes = {ele: i for i, ele in enumerate(classes)}

        # print(self.classes)
        self.parse_xml_list = list()
        # 读取xml:
        for file_name in file_name_list:
            with open(join("../VOCdevkit/VOC2007/Annotations", file_name.strip()+".xml"), "r") as f:
                temp = parse_xml(f.read())
                temp["object"] = [self.classes.get(ele, -1) for ele in temp["object"]]
                self.parse_xml_list.append(temp)
                del temp
        # print(parse_xml_list) # [{'filename': '000424', 'object': ['plastic-bag'], 'bbox': [[14, 222, 67, 261]], 'shape': [416, 416]}, {}, ...]
        self.parse_xml_list.sort(key=lambda x:  max(x["shape"])) # 从小到大排序. 

        # train_transforme = transforms.Compose([
        #         transforms.Resize(self.img_size),
        #         transforms.ToTensor()
        # ])

    def __getitem__(self, index):
        xml_dict = self.parse_xml_list[index]
        hyp = self.hyp
        if self.mosaic:
            pass
        else:
            cv_img, (h0, w0), (h, w) = self.load_image(index) # 原来的尺寸, 以及形变之后的尺寸
            # print((h0, w0), (h, w))  # (400, 225) (416, 234) # 这就是经过load之后的图片。
            
            
            label = np.hstack([np.array([xml_dict["object"]]).T, xml_dict["bbox"]]).astype(np.float32)
            
            # 对label进行处理: 转换为相对坐标. 
            # xmin, ymin, xmax, ymax -> x, y, w, h, # 相对坐标.
            print("xyxy", label)

            if label.ndim > 0:
                label[:, 3] = label[:, 3] - label[:, 1]
                label[:, 4] = label[:, 4] - label[:, 2]
                print("xywh", label)


                label[:, 1] = (label[:, 1]/float(w0)).astype(np.float32)
                label[:, 2] = (label[:, 2]/float(h0)).astype(np.float32)
                label[:, 3] = (label[:, 3]/float(w0)).astype(np.float32)
                label[:, 4] = (label[:, 4]/float(h0)).astype(np.float32)


            print(label)



            return self.transforme(cv_img), torch.from_numpy(np.hstack([np.zeros(shape=(label.shape[0], 1)), label])), xml_dict["shape"], index
            
            

    def __len__(self):
        return self.n

    def load_image(self, index):
        
        # print(self.parse_xml_list[index])
        parse_xml_dict = self.parse_xml_list[index]
        cv_img = cv.imread(parse_xml_dict["path"]) # BGR format.
        cv.imwrite("./test.jpg", cv_img)

        h0, w0 = parse_xml_dict["shape"]
        r = self.img_size / max(h0, w0)  # resize image to img_size, 得到比例
        
        if r != 1:  # if sizes are not equal
            interp = cv.INTER_AREA if r < 1 and not self.augment else cv.INTER_LINEAR
            cv_img = cv.resize(cv_img, (int(w0 * r), int(h0 * r)), interpolation=interp) # resize. 不会改变原图像比例. 就是插值. 
        return cv_img, (h0, w0), cv_img.shape[:2]  # img, hw_original, hw_resized
        


    @staticmethod
    def collate_fn(batch):
        img, label, shapes, index = zip(*batch)  # transposed
        for i, l in enumerate(label): # 这一步实际上就是在最前面加上属于那张图片. 我在dataset中做了.
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), shapes, index
    

def list2txt(path: str, l: list) -> None:
    with open(path, "wt") as f:
        f.write("\n".join(l))

def gen_data_path(file_path="../files/data.json", data_path="../VOCdevkit/VOC2007", train_rate=0.8):
    assert os.path.isfile(file_path), f"{file_path} is not exist. "

    xml_file_path = join(data_path, 'Annotations')
    total_file_name_list = [file_name.split(".")[0] for file_name in os.listdir(xml_file_path) if file_name.split(".")[-1]=='xml']
    total_num = len(total_file_name_list)
    train_num  = int(total_num*train_rate)
    val_num = total_num - train_num

    train_file_list = random.sample(total_file_name_list, train_num)
    val_file_list = random.sample(total_file_name_list, val_num)
    data_json = {"train_path": "../train_file.txt", "val_path": "../val_file.txt", "classes": "../files/classes.txt"}
    list2txt("../train_file.txt", train_file_list)
    list2txt("../val_file.txt", val_file_list)
    with open("../files/data.json", "wt") as f:
        f.write(json.dumps(data_json))

    all_xml_list = parse_all_xml(xml_file_path)
    names = []

    for ele in all_xml_list:
        names.extend(ele["object"])
    names = tuple(set(names))
    with open("../files/classes.txt", "wt") as f:
        f.write("\n".join(names))
    
    return all_xml_list
    


def parse_all_xml(Annotations_path: str):
    xml_parse_list = []
    xml_file_list = os.listdir(Annotations_path)
    for xml_file in xml_file_list:
        with open(join(Annotations_path, xml_file), "rt")  as f:
            xml_parse_list.append(parse_xml(f.read()))

    return xml_parse_list


def parse_xml(str_xml: str) -> dict:
    result_dict = dict()
    e = etree.XML(str_xml)
    result_dict["filename"] = e.xpath("//filename/text()")[0].split(".")[0]
    result_dict["object"] = e.xpath("//object/name/text()")
    bboxs = [int(ele) for ele in e.xpath("//object/bndbox/*/text()")]
    result_dict["bbox"] = np.ascontiguousarray(np.array([bboxs[bbox: bbox+4] for bbox in range(0, len(bboxs), 4)]))
    
    result_dict["shape"] = [int(ele) for ele in e.xpath("//size/*[position()<3]/text()")]
    result_dict["path"] = join(os.getcwd(), "../VOCdevkit/VOC2007/JPEGImages", result_dict["filename"]+".jpg")

    return result_dict





def test_parse_xml():
    s= """
<annotation>
	<folder>JPEGImages</folder>
	<filename>000001.jpg</filename>
	<path>/home/zbr/VOCdevkit/VOC2007/JPEGImages/000001.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>416</width>
		<height>416</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
	<!-- 主要关注object标签下面的文件夹。 -->
		<name>bottle</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>172</xmin>
			<ymin>226</ymin>
			<xmax>228</xmax>
			<ymax>283</ymax>
		</bndbox>
	</object>
</annotation>
    
"""
    print(parse_xml(s))


def draw_relavive_img(image, xywh, shapes):
    image = image.transpose(1, 2, 0) * 255
    image = image.astype(np.uint8).copy()

    h, w = image.shape[: 2]
    print(h, w)
    xywh = xywh[:, 1:]

    # xywh[:, 1] = (xywh[:, 1] * w) # x
    # xywh[:, 2] = (xywh[:, 2] * h) # y
    # xywh[:, 3] = (xywh[:, 3] * w) # w
    # xywh[:, 4] = (xywh[:, 4] * h) # h
    xywh[:, 1] = (xywh[:, 1]*float(w))
    xywh[:, 2] = (xywh[:, 2]*float(h))
    xywh[:, 3] = (xywh[:, 3]*float(w))
    xywh[:, 4] = (xywh[:, 4]*float(h))
    # xywh = np.ceil(xywh.copy())
    xywh = xywh.astype(np.uint8).copy()

    print("xywh", xywh)
    for line in xywh:
        line = line.astype(np.uint16).copy() # 这里又可能会出现截断现象。 
        print(line[2], line[4])
        print("line[2]+line[4]", line[2]+line[4])
        print((line[1], line[2]), (line[1]+line[3], line[2]+line[4]))
        cv.rectangle(image, (line[1], line[2]), (line[1]+line[3], line[2]+line[4]), color=(255, 0, 0), thickness=1, )








    plt.imshow(image)
    plt.savefig("./test_draw.jpg")
    plt.show()
    



def test_dataset():
    yolo_dataset = Yolo_VOC_dataset()
    # print(yolo_dataset[-203])


    img, label, shape0, index = yolo_dataset[-203]
    print(img.shape)
    h0, w0 = shape0
    print(h0, w0)
    label = label[:, 1:]
    label[:, 1] = (label[:, 1]*float(w0))
    label[:, 2] = (label[:, 2]*float(h0))
    label[:, 3] = (label[:, 3]*float(w0))
    label[:, 4] = (label[:, 4]*float(h0))
    label = np.ceil(label.numpy().copy())
    print(label)    

def test_dataloader():
    yolo_dataset = Yolo_VOC_dataset()
    yolo_dataloader = DataLoader(
        yolo_dataset,
        3,
        collate_fn=Yolo_VOC_dataset.collate_fn
    )
    for imgs, labels, shapes, indexs in yolo_dataloader:
        # print(imgs.shape)
        print(labels)
        
        break

def test_draw():
    yolo_dataset = Yolo_VOC_dataset()
    print()
    image, label, shape, index = yolo_dataset[-203]
    image = image.detach().cpu().numpy()
    label = label.detach().cpu().numpy()

    draw_relavive_img(image, label, shape)



if __name__ == "__main__":
    test_dataset()
    # test_dataloader()
    
    test_draw()








