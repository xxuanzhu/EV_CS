#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/3/12 15:43
# @Author : wbw
# @Version：V 0.1
# @File : three_CS_equilibrium.py
# @desc : 基于暴力枚举的best_response求解上层三个充电站的定价均衡

from EV_Equilibrium import EvEquilibrium
import numpy as np
from config import config
from matplotlib import pyplot as plt
import random

evEquilibrium = EvEquilibrium()


def cs_best_price_simulation(region, dist1, dist2, dist3, region_num, vehicle_vector, p_min, p_max,
                             price1, price2, price3, strategy_vector1, strategy_vector2, strategy_vector3):
    # 求得初始定价price1和price2下底层的策略
    strategy1, strategy2, strategy3 = evEquilibrium.best_response_simulation(region, dist1, dist2, dist3, vehicle_vector,
                                                                  price1, price2, price3, strategy_vector1,
                                                                             strategy_vector2, strategy_vector3)
    f_i_1 = 0
    for item in range(region_num):
        f_i_1 = f_i_1 + vehicle_vector[item] * strategy1[item]  # 所有区域派到充电站1的车辆数量
    revenue_before = price1 * f_i_1

    best_price = price1
    for p_1 in np.arange(p_min, p_max, 1):
        print("当前定价", p_1)
        # 求得此定价下底层的策略
        new_strategy1, new_strategy2, new_strategy3 = evEquilibrium.best_response_simulation(region, dist1, dist2, dist3
                                                                                             , vehicle_vector, p_1,
                                                                                             price2, price3,
                                                                                             strategy_vector1,
                                                                                             strategy_vector2,
                                                                                             strategy_vector3)
        new_f_i_1 = 0
        for item in range(region_num):
            new_f_i_1 = new_f_i_1 + vehicle_vector[item] * new_strategy1[item]  # 所有区域派到充电站1的车辆数量
        revenue_temp = p_1 * new_f_i_1

        if revenue_temp > revenue_before:
            revenue_before = revenue_temp
            best_price = p_1

    return best_price


if __name__ == "__main__":
    epision = 0.000001
    delta = 1
    p_min = 50
    p_max = 250
    # price_3 = 100
    region_num = config.region_num
    vehicle_vector = config.vehicle_vector

    dist1, dist2, dist3, vehicle_num, region, strategy_vector1, strategy_vector2, strategy_vector3 = \
        evEquilibrium.initiation(region_num)

    price_list1 = []
    price_list2 = []
    price_list3 = []

    # 计算CS1和CS2,CS3的初始收益
    # init_price1 = 150
    # init_price2 = 150
    # init_price3 = 150

    init_price1 = random.randint(p_min, p_max)
    init_price2 = random.randint(p_min, p_max)
    init_price3 = random.randint(p_min, p_max)

    flag1 = init_price1
    flag2 = init_price2
    flag3 = init_price3

    price_list1.append(init_price1)
    price_list2.append(init_price2)
    price_list3.append(init_price3)

    new_price1 = cs_best_price_simulation(region, dist1, dist2, dist3, region_num, vehicle_vector, p_min,
                                                  p_max, init_price1, init_price2, init_price3, strategy_vector1,
                                                  strategy_vector2, strategy_vector3)
    new_price2 = cs_best_price_simulation(region, dist2, dist1, dist3, region_num, vehicle_vector, p_min,
                                      p_max, init_price2, new_price1, init_price3, strategy_vector2,
                                      strategy_vector1, strategy_vector3)
    new_price3 = cs_best_price_simulation(region, dist3, dist1, dist2, region_num, vehicle_vector, p_min,
                                          p_max, init_price3, new_price1, new_price2, strategy_vector3,
                                          strategy_vector1, strategy_vector2)

    price_list1.append(new_price1)
    price_list2.append(new_price2)
    price_list3.append(new_price3)

    while new_price1 - flag1 > epision or new_price2 - flag2 > epision or new_price3 - flag3 > epision:

        flag1 = new_price1
        flag2 = new_price2
        flag3 = new_price3

        new_price1 = cs_best_price_simulation(region, dist1, dist2, dist3, region_num, vehicle_vector,
                                              p_min,
                                              p_max, new_price1, new_price2, new_price3, strategy_vector1,
                                              strategy_vector2, strategy_vector3)
        new_price2 = cs_best_price_simulation(region, dist2, dist1, dist3, region_num, vehicle_vector,
                                              p_min,
                                              p_max, new_price2, new_price1, new_price3, strategy_vector2,
                                              strategy_vector1, strategy_vector3)
        new_price3 = cs_best_price_simulation(region, dist3, dist1, dist2, region_num, vehicle_vector,
                                              p_min,
                                              p_max, new_price3, new_price1, new_price2, strategy_vector3,
                                              strategy_vector1, strategy_vector2)
        price_list1.append(new_price1)
        price_list2.append(new_price2)
        price_list3.append(new_price3)

    print(new_price1)
    print(new_price2)
    print(new_price3)

    print(price_list1)
    print(price_list2)
    print(price_list3)

    plt.title("CS1 and CS2, CS3 price curve")
    plt.xlabel("CDF")
    plt.ylabel("price")
    plt.plot(range(len(price_list1)), price_list1, color='red', label='CS1 price')
    plt.plot(range(len(price_list2)), price_list2, color='green', label='CS2 price')
    plt.plot(range(len(price_list3)), price_list3, color='blue', label='CS3 price')

    plt.legend()  # 显示图例
    plt.show()
