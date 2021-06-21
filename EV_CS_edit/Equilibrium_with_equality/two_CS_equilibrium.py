#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/3/12 9:14
# @Author : wbw
# @Version：V 0.1
# @File : two_CS_equilibrium.py
# @desc : 暴力枚举使得两个CS的定价均衡


from EV_Equilibrium import EvEquilibrium
import numpy as np
from config import config
from matplotlib import pyplot as plt
import random

evEquilibrium = EvEquilibrium()


def cs_best_price_simulation(region, dist1, dist2, vehicle_num, region_num, vehicle_vector, p_min, p_max,
                             price1, price2, minimize_res_list, strategy_vector1, strategy_vector2):
    # 求得初始定价price1和price2下底层的策略
    strategy1, strategy2 = evEquilibrium.best_response_simulation(region, dist1, dist2, vehicle_num,
                                                                  price1, price2, minimize_res_list, strategy_vector1,
                                                                  strategy_vector2)
    f_i_1 = 0
    for item in range(region_num):
        f_i_1 = f_i_1 + vehicle_vector[item] * strategy1[item]  # 所有区域派到充电站1的车辆数量
    revenue_before = price1 * f_i_1

    best_price = price1
    for p_1 in np.arange(p_min, p_max, 1):
        print("当前定价", p_1)
        # 求得此定价下底层的策略
        new_strategy1, new_strategy2 = evEquilibrium.best_response_simulation(region, dist1, dist2, vehicle_num,
                                                                      p_1, price2, minimize_res_list, strategy_vector1,
                                                                      strategy_vector2)
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
    p_min = 200
    p_max = 300
    region_num = config.region_num
    minimize_res_list = []
    vehicle_vector = config.vehicle_vector

    dist1, dist2, vehicle_num, region, strategy_vector1, strategy_vector2 = evEquilibrium.initiation(region_num)
    # print("初始化dist1, dist2, 车辆数， agent, 策略向量1， 策略向量2分别为：", dist1, dist2, vehicle_num,
    #       region, strategy_vector1, strategy_vector2)

    price_list1 = []
    price_list2 = []

    # 计算CS1和CS2的初始收益
    # init_price1 = 150
    # init_price2 = 150
    init_price1 = random.randint(p_min, p_max)
    init_price2 = random.randint(p_min, p_max)

    flag1 = init_price1
    flag2 = init_price2

    price_list1.append(init_price1)
    price_list2.append(init_price2)

    new_price1 = cs_best_price_simulation(region, dist1, dist2, vehicle_num, region_num, vehicle_vector, p_min,
                                                  p_max, init_price1, init_price2, minimize_res_list, strategy_vector1,
                                                  strategy_vector2)
    new_price2 = cs_best_price_simulation(region, dist2, dist1, vehicle_num, region_num, vehicle_vector, p_min,
                                      p_max, init_price2, new_price1, minimize_res_list, strategy_vector2,
                                      strategy_vector1)
    price_list1.append(new_price1)
    price_list2.append(new_price2)

    while new_price1 - flag1 > epision or new_price2 - flag2 > epision:

        flag1 = new_price1
        flag2 = new_price2

        new_price1 = cs_best_price_simulation(region, dist1, dist2, vehicle_num, region_num, vehicle_vector, p_min,
                                              p_max, new_price1, new_price2, minimize_res_list, strategy_vector1,
                                              strategy_vector2)
        new_price2 = cs_best_price_simulation(region, dist2, dist1, vehicle_num, region_num, vehicle_vector, p_min,
                                              p_max, new_price2, new_price1, minimize_res_list, strategy_vector2,
                                              strategy_vector1)
        price_list1.append(new_price1)
        price_list2.append(new_price2)

    print(new_price1)
    print(new_price2)

    print(price_list1)
    print(price_list2)
    plt.title("CS1 and CS2 price curve")
    plt.xlabel("CDF")
    plt.ylabel("price")
    plt.plot(range(len(price_list1)), price_list1, color='red', label='CS1 price')
    plt.plot(range(len(price_list2)), price_list2, color='green', label='CS2 price')
    plt.legend()  # 显示图例
    plt.show()
