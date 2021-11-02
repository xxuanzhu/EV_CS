#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/3/5 9:39
# @Author : wbw+xxuanzhu
# @Version：V 0.1
# @File : config.py
# @desc : configuration file

import numpy as np


class Config:
    """配置文件，包含相关环境数据等

    """

    def __init__(self):
        super().__init__()

        # 权重参数
        self.weight_price = 0.6
        self.weight_dist = 0.1
        self.weight_queue = 0.3

        self.threshold = 0.000001  # 迭代截止阈值
        self.total_cs = 11
        self.total_regions = 11

        self.region_num = 6 # 下层区域的数量
        self.cs_num = 4  # 充电站的数量
        # 初始化
        self.total_price = np.array([3, 0, 6, 8, 1, 65, 68, 74, 85, 48, 58])
        self.total_vehicle_vector = np.zeros(self.region_num, dtype=np.float64)
        self.region_id = [i for i in range(self.region_num)]
        self.total_dist_vector = np.zeros((self.cs_num, self.region_num), dtype=np.float64)
        self.cs_cost_vector = np.array([200, 120, 150, 300, 244, 156, 222, 234, 69, 102, 300])
        self.cs_service_capacity_vector = np.array([10, 34, 12, 25, 67, 35, 55, 45, 67, 10, 55])

        # 开始读取
        # data = []
        # for i in range(self.region_num):
        #     line = input()
        #     split_line = line.split("\t")
        #     self.region_id[i] = int(split_line[0])-1
        #     self.total_vehicle_vector[i] = split_line[1]
        #     for j in range(self.cs_num):
        #         self.total_dist_vector[j][i] = int(math.floor(float(split_line[j+2])))

        self.total_vehicle_vector = np.array(
            [800, 457, 380, 680, 209, 432, 123, 130, 106, 189, 102, 179, 196, 186, 182, 130, 162, 171, 153, 141])
        # 横坐标是CS维度，纵坐标是区域维度  cs->region

        self.total_dist_vector = np.array([
            [167, 200, 10, 60, 205, 400, 199, 144, 55, 192, 15, 25, 194, 41, 47, 65, 147,
             94, 82, 31],
            [402, 30, 156, 209, 309, 302, 148, 71, 36, 88, 60, 23, 57, 197, 173, 190, 172,
             154, 25, 73],
            [70, 402, 306, 378, 50, 399, 96, 105, 47, 110, 101, 154, 28, 117, 94, 155,
             183, 147, 76, 113],
            [136, 113, 166, 89, 154, 175, 172, 32, 179, 25, 152, 99, 190, 169, 170, 22,
             61, 17, 197, 129],
            [162, 159, 87, 33, 48, 198, 32, 189, 81, 103, 57, 131, 107, 56, 52, 108,
             148, 19, 172, 95],
            [185, 63, 149, 98, 139, 98, 121, 15, 159, 157, 11, 140, 86, 79, 148, 103,
             29, 68, 74, 64],
            [169, 30, 77, 33, 80, 190, 199, 144, 55, 192, 15, 25, 194, 41, 47, 65, 147,
             94, 82, 31],
            [89, 108, 198, 170, 101, 190, 76, 85, 18, 165, 163, 95, 94, 144, 194, 163,
             28, 13, 62, 129],
            [158, 73, 193, 104, 13, 94, 58, 26, 46, 177, 41, 186, 73, 74, 41, 194, 75,
             85, 171, 158],
            [91, 112, 84, 68, 62, 194, 75, 141, 51, 15, 129, 89, 37, 195, 39, 146, 33,
             194, 143, 24],
            [49, 106, 86, 95, 70, 127, 100, 179, 144, 166, 29, 126, 102, 181, 121, 44,
             194, 142, 190, 157],
            [179, 69, 186, 43, 72, 35, 139, 54, 31, 156, 165, 157, 14, 80, 189, 58, 132,
             198, 59, 14],
            [119, 57, 95, 103, 59, 117, 90, 183, 115, 196, 159, 33, 147, 99, 73, 67,
             118, 80, 173, 64],
            [19, 101, 148, 198, 104, 74, 149, 192, 44, 105, 112, 27, 75, 37, 168, 65,
             39, 149, 26, 134],
            [108, 125, 127, 198, 80, 108, 181, 111, 126, 198, 141, 150, 65, 54, 60, 200,
             86, 101, 197, 163],
            [184, 111, 139, 134, 120, 63, 123, 34, 71, 12, 27, 25, 106, 165, 31, 190,
             88, 120, 146, 76],
            [65, 75, 106, 24, 11, 199, 160, 158, 143, 26, 56, 194, 119, 142, 23, 23, 39,
             95, 104, 56],
            [17, 161, 82, 54, 35, 164, 72, 198, 128, 188, 168, 122, 98, 64, 182, 78, 80,
             21, 19, 92],
            [185, 22, 33, 19, 57, 13, 187, 64, 188, 66, 173, 65, 132, 164, 194, 52, 135,
             118, 64, 55],
            [16, 88, 121, 58, 179, 73, 37, 99, 151, 187, 34, 105, 171, 38, 192, 44, 60,
             101, 113, 176],
            [171, 110, 109, 64, 38, 187, 67, 189, 58, 165, 118, 64, 138, 54, 11, 103,
             129, 35, 183, 122]]
        )

        # 根据充电站数量和研究区域的数量截取相关环境数据
        self.price = self.total_price[:self.cs_num].tolist()
        self.vehicle_vector = self.total_vehicle_vector[:self.region_num].tolist()
        self.dist_vector = self.total_dist_vector[:self.cs_num, :self.region_num].tolist()
        # print(self.price)
        # print(self.vehicle_vector)
        # print(self.dist_vector)
        # # self.dist_vector = []  # (cs_num*region_num), 根据cs_num生成二维dist
        # self.vehicle_vector = []
        # for cs in range(self.cs_num):
        #     cs_vector = []
        # for i in range(self.region_num):
        #     cs_vector.append(random.randint(10, 200))
        # self.dist_vector.append(cs_vector)
        # print(self.total_dist_vector)
        #
        # for i in range(self.region_num):
        #     self.vehicle_vector.append(round(random.uniform(40, 200), 0))

    def change_region_num(self, change_num):
        self.region_num = change_num

    def change_cs_num(self, change_num):
        self.cs_num = change_num


config = Config()
