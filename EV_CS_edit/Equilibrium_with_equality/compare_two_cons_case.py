#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/3/6 16:28
# @Author : wbw
# @Version：V 0.1
# @File : compare_two_cons_case.py
# @desc : 比较在有等式约束和没有等式约束情况下的流量分布情况

from EV_Equilibrium import evEquilibrium
from EV_Equilibrium_only_equality import evEquilibriumCompare
import numpy as np
from matplotlib import pyplot as plt
from config import config


if __name__ == "__main__":
    # 初始化操作, 假设目前有1个公司控制2个充电站
    p_1 = 30
    p_2 = 10
    region_num = config.region_num
    minimize_res_list = []
    vehicle_vector = config.vehicle_vector

    dist1, dist2, vehicle_num, region, strategy_vector1, strategy_vector2 = evEquilibrium.initiation(region_num)
    print("初始化dist1, dist2, 车辆数， agent, 策略向量1， 策略向量2分别为：", dist1, dist2, vehicle_num,
          region, strategy_vector1, strategy_vector2)

    # 求得此定价下,存在不等式约束的底层的策略
    strategy1, strategy2 = evEquilibrium.best_response_simulation(region, dist1, dist2, vehicle_num,
                                                                  p_1, p_2, minimize_res_list, strategy_vector1,
                                                                  strategy_vector2)
    f_i_1 = 0  # 到充电站1的车流量
    f_i_2 = 0  # 到充电站2的车流量
    for item in range(region_num):
        f_i_1 = f_i_1 + vehicle_vector[item] * strategy1[item]  # 所有区域派到充电站1的车辆数量
        f_i_2 = f_i_2 + vehicle_vector[item] * strategy2[item]  # 所有区域派到充电站2的车辆数量

    # 下面分别来看均衡下，每个充电站的流量负载情况是否相等
    load1 = p_1 + dist1[1] + f_i_1 + vehicle_vector[1] * strategy1[1]
    load2 = p_2 + dist2[1] + f_i_2 + vehicle_vector[1] * strategy2[1]

    # 求得此定价下,不存在不等式约束的底层的策略
    dist1, dist2, vehicle_num, region, strategy_vector1, strategy_vector2 = evEquilibriumCompare.initiation(region_num)
    print("初始化dist1, dist2, 车辆数， agent, 策略向量1， 策略向量2分别为：", dist1, dist2, vehicle_num,
          region, strategy_vector1, strategy_vector2)
    new_strategy1, new_strategy2 = evEquilibriumCompare.best_response_simulation(region, dist1, dist2, vehicle_num,
                                                                  p_1, p_2, minimize_res_list, strategy_vector1,
                                                                  strategy_vector2)

    new_f_i_1 = 0  # 到充电站1的车流量
    new_f_i_2 = 0  # 到充电站2的车流量
    for item in range(region_num):
        new_f_i_1 = new_f_i_1 + vehicle_vector[item] * new_strategy1[item]  # 所有区域派到充电站1的车辆数量
        new_f_i_2 = new_f_i_2 + vehicle_vector[item] * new_strategy2[item]  # 所有区域派到充电站2的车辆数量

    # 下面分别来看均衡下，每个充电站的流量负载情况是否相等
    new_load1 = p_1 + dist1[1] + new_f_i_1 + vehicle_vector[1] * new_strategy1[1]
    new_load2 = p_2 + dist2[1] + new_f_i_2 + vehicle_vector[1] * new_strategy2[1]

    print("随机生成的环境：")
    print("dist1:", dist1)
    print("dist2:", dist2)
    print("车辆数：", vehicle_vector)

    print("在有不等式约束情况下不同区域的流量分布策略：")
    print("CS1的策略：", strategy1)
    print("CS2的策略：", strategy2)
    print("负载1：", load1)
    print("负载2：", load2)

    print("在无不等式约束情况下不同区域的流量分布策略：")
    print("CS1的策略：", new_strategy1)
    print("CS2的策略：", new_strategy2)
    print("负载1：", new_load1)
    print("负载2：", new_load2)
