#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/3/5 9:39
# @Author : wbw
# @Version：V 0.1
# @File : config.py
# @desc : configuration file
import random

import numpy as np


class Config:
    """配置文件，包含相关环境数据等

    """

    def __init__(self):
        super().__init__()
        self.threshold = 0.000001  # 迭代截止阈值
        self.region_num = 5  # 下层区域的数量
        self.cs_num = 5  # 充电站的数量
        # self.vehicle_vector = np.array([50, 50, 50, 50, 50])  # 下层每个区域的充电车流量

        self.dist_vector = []  # (cs_num*region_num), 根据cs_num生成二维dist
        self.vehicle_vector = []
        for cs in range(self.cs_num):
            cs_vector = []
            for i in range(self.region_num):
                cs_vector.append(round(random.uniform(10, 100), 0))
            self.dist_vector.append(cs_vector)
        # self.dist_vector = [[15, 25, 40, 60, 55],
        #                     [10, 35, 45, 20, 70],
        #                     [23, 45, 60, 75, 90],
        #                     [35, 65, 80, 15, 55],
        #                     [20, 45, 25, 60, 60],
        #                     [45, 65, 35, 55, 75],
        #                     [20, 45, 35, 65, 90],
        #                     [20, 45, 65, 90, 10],
        #                     [20, 30, 40, 55, 65],
        #                     [10, 20, 35, 45, 55]]
        # print(self.dist_vector)

        for i in range(self.region_num):
            self.vehicle_vector.append(round(random.uniform(10, 200), 0))

    def change_region_num(self, change_num):
        self.region_num = change_num

    def change_cs_num(self, change_num):
        self.cs_num = change_num



config = Config()
