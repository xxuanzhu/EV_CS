#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/3/5 11:09
# @Author : wbw
# @Version：V 0.1
# @File : verify_equilibrium.py
# @desc : verify the equilibrium of regions
import random

from EV_Equilibrium import EvEquilibrium
import numpy as np
from matplotlib import pyplot as plt
from config import config

evEquilibrium = EvEquilibrium()

"""验证底层的均衡解是否正确，原理是对于每个区域，
其在均衡状态下发到每个充电站的流量满足：p_j+q_j+d_ij+f_ij相等。
"""
if __name__ == "__main__":
    # 初始化操作, 假设目前有1个公司控制2个充电站
    # p_1 = 30
    # p_2 = 10
    price = []
    p_min = 10
    p_max = 50
    for cs in range(config.cs_num):
        price.append(random.randint(p_min, p_max))
    region_num = config.region_num
    minimize_res_list = []
    vehicle_vector = config.vehicle_vector

    dist, vehicle_num, region, strategy_vector = evEquilibrium.initiation(region_num, config.cs_num)
    print("初始化dist集合, 车辆数， agent, 策略向量集合, 价格集合分别为：", dist, vehicle_num,
          region, strategy_vector, price)

    # 求得此定价下底层的策略
    strategy = evEquilibrium.best_response_simulation(region, dist, vehicle_num,
                                                      price, minimize_res_list, strategy_vector)
    for agent in range(config.region_num):
        print("区域", agent, "的策略：", strategy[agent])

    # f_i_1 = 0  # 到充电站1的车流量
    # f_i_2 = 0  # 到充电站2的车流量
    # f_i_3 = 0
    f_i = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for cs in range(config.cs_num):
        for item in range(region_num):
            f_i[cs] = f_i[cs] + vehicle_vector[item] * strategy[item][cs]

            # f_i_1 = f_i_1 + vehicle_vector[item] * strategy[item]  # 所有区域派到充电站1的车辆数量
            # f_i_2 = f_i_2 + vehicle_vector[item] * strategy[item]  # 所有区域派到充电站2的车辆数量
            # f_i_2 = f_i_2 + vehicle_vector[item] * strategy[item]

    # 下面分别来看均衡下，每个充电站的流量负载情况是否相等

    # 对区域4
    load = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for cs in range(config.cs_num):
        load[cs] = price[cs] + dist[cs][1] + f_i[cs] + vehicle_vector[1] + strategy_vector[1][cs]

    for cs in range(config.cs_num):
        print("负载", cs+1, ": ", load[cs])


