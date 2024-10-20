from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        """
        初始化数据集。

        参数:
        - images_path (list): 包含所有图片路径的列表。
        - images_class (list): 包含图片对应类别的列表。
        - transform: 图像的预处理/变换（可选）。
        """
        self.images_path = images_path  # 存储图片路径列表
        self.images_class = images_class  # 存储图片对应的类别标签列表
        self.transform = transform  # 存储图像的变换方法

    def __len__(self):
        """
        返回数据集的大小。

        返回值:
        - int: 数据集中图片的总数。
        """
        return len(self.images_path)  # 返回图片路径列表的长度

    def __getitem__(self, item):
        """
        根据索引获取数据集中的样本。

        参数:
        - item (int): 索引值。

        返回值:
        - img: 经过预处理的图片。
        - label: 图片对应的类别标签。
        """
        img = Image.open(self.images_path[item])  # 打开指定索引的图片
        # 检查图片是否为 RGB 模式
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        
        label = self.images_class[item]  # 获取图片对应的标签

        # 如果指定了图像变换，则对图像进行变换
        if self.transform is not None:
            img = self.transform(img)

        return img, label  # 返回图片和标签

    @staticmethod
    def collate_fn(batch):
        """
        自定义的批量数据处理函数。
        
        参数:
        - batch: 由多个样本组成的批次。
        返回值:
        - images: 叠加后的批量图像张量。
        - labels: 叠加后的批量标签张量。
        
        参考: 官方的 default_collate 实现可以参考
        https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        """
        # 将批量样本中的图片和标签分别提取出来
        images, labels = tuple(zip(*batch))

        # 将图片列表转换为张量，并沿第 0 维进行叠加
        images = torch.stack(images, dim=0)
        # 将标签列表转换为张量
        labels = torch.as_tensor(labels)
        return images, labels  # 返回图片和标签的批量张量
