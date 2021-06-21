# -*- coding:utf-8 -*-
# @Time: 2021/2/18 12:16
# @Author: xxuanzhu
# @Email: xxuanzhu@seu.edu.cn
# @File: CS_my.py
# 此文件用于求解两个充电站之间的定价均衡
import copy
import random

from matplotlib import pyplot as plt

from EV_Equilibrium_10cs import EvEquilibrium
from config import config

evEquilibrium = EvEquilibrium()


def cs_best_price_simulation(region, dist_vector_list, vehicle_num, region_num, changable_cs_number,
                             price, p_min, p_max, minimize_res_list, strategy_vector_list):
    strategy, signal = evEquilibrium.best_response_simulation(region, dist_vector_list, vehicle_num,
                                                      price, minimize_res_list,
                                                      strategy_vector_list)
    # revenue_temp = 0
    f_i = 0
    for item in range(region_num):
        f_i = f_i + vehicle_num[item] * strategy[item][changable_cs_number]

    revenue_before = price[changable_cs_number] * f_i

    best_price = price[changable_cs_number]

    for p in range(p_min, p_max, 1):
        print("当前变化的", changable_cs_number, "号充电桩定价：", p)
        price[changable_cs_number] = p

        new_strategy, signal = evEquilibrium.best_response_simulation(region, dist_vector_list, vehicle_num,
                                                          price, minimize_res_list,
                                                          strategy_vector_list)
        if signal == False:
            return p

        new_f_i = 0
        for item in range(region_num):
            new_f_i = new_f_i + vehicle_num[item] * new_strategy[item][
                changable_cs_number]

        revenue_temp = p * new_f_i

        if revenue_temp > revenue_before:
            revenue_before = revenue_temp
            best_price = p

    return best_price


def get_new_price(init_price, price_minus_delta, price_plus_delta, revenue_minus_delta,
                  revenue, revenue_plus_delta):
    if max(revenue_minus_delta, revenue, revenue_plus_delta) == revenue_plus_delta:
        return price_plus_delta
    elif max(revenue_minus_delta, revenue, revenue_plus_delta) == revenue_minus_delta:
        return price_minus_delta
    else:
        return init_price


def get_new_revenue(price1_new, flag1_origin, price1_minus_delta, price1_plus_delta, revenue2_after_p1_minus, revenue2,
                    revenue2_after_p1_plus):
    if price1_new == flag1_origin:
        return revenue2
    elif price1_new == price1_minus_delta:
        return revenue2_after_p1_minus
    else:
        return revenue2_after_p1_plus


if __name__ == "__main__":
    epision = 0.000001
    delta = 1
    p_min = 50
    p_max = 2000
    price = []
    # price1_list = []
    # price1_list.append(50)
    for cs in range(config.cs_num):
        price.append(random.randint(p_min, p_max))
    region_num = 5
    minimize_res_list = []
    # vehicle_vector = np.array([20, 30, 30, 50, 10])

    dist, vehicle_num, region, strategy_vector = evEquilibrium.initiation(region_num, config.cs_num)
    print("初始化dist集合, 车辆数， agent, 策略向量集合, 价格集合分别为：", dist, vehicle_num,
          region, strategy_vector, price)

    # revenue1_list = []
    # revenue2_list = []
    # 记录每轮迭代的最优revenue
    # revenue1_optimal_list = []
    # revenue2_optimal_list = []
    # 记录每次一cs固定时，另一个cs的最优价格

    # flag = [i for i in range(config.cs_num)]
    flag = copy.deepcopy(price)
    # for i in range(config.cs_num):
    #     flag[i] = price[i]
    # p1_list = []
    # p2_list = []
    price_list = [[] for i in range(config.cs_num)]
    for i in range(config.cs_num):
        price_list[i].append(price[i])
    # p1_list.append(price[0])
    # p2_list.append(price[1])
    # revenue1_max = 0
    # revenue2_max = 0

    for i in range(config.cs_num):
        new_price = cs_best_price_simulation(region, dist, vehicle_num, region_num, i, price, p_min, p_max, minimize_res_list, strategy_vector)
        price_list[i].append(new_price)
        price[i] = new_price

    while abs(price[0] - flag[0]) > epision or abs(price[1] - flag[1]) > epision\
            or abs(price[2] - flag[2]) > epision or abs(price[3] - flag[3]) > epision\
            or abs(price[4] - flag[4]) > epision or abs(price[5] - flag[5]) > epision\
            or abs(price[6] - flag[6]) > epision or abs(price[7] - flag[7]) > epision\
            or abs(price[8] - flag[8]) > epision or abs(price[9] - flag[9]) > epision:

        # for i in range(config.cs_num):
        #     flag[i] = price[i]
        flag = copy.deepcopy(price)

        for i in range(config.cs_num):
            new_price = cs_best_price_simulation(region, dist, vehicle_num, region_num, i, price, p_min, p_max,
                                                 minimize_res_list, strategy_vector)
            price_list[i].append(new_price)
            price[i] = new_price



    for i in range(config.cs_num):
        print("当前充电站", i, "的最优定价为：",price[i], "定价集合为：", price_list[i])

    plt.title("CS1 and CS2 and CS3 and CS4 and CS5 and CS6 price curve")
    plt.xlabel("CDF")
    plt.ylabel("price")
    plt.plot(range(len(price_list[0])), price_list[0], color='red', label='CS1 revenue')
    plt.plot(range(len(price_list[1])), price_list[1], color='green', label='CS2 revenue')
    plt.plot(range(len(price_list[2])), price_list[2], color='blue', label='CS3 revenue')
    plt.plot(range(len(price_list[3])), price_list[3], color='black', label='CS4 revenue')
    plt.plot(range(len(price_list[4])), price_list[4], color='yellow', label='CS5 revenue')
    plt.plot(range(len(price_list[5])), price_list[5], color='gray', label='CS6 revenue')
    plt.plot(range(len(price_list[6])), price_list[6], color='chocolate', label='CS6 revenue')
    plt.plot(range(len(price_list[7])), price_list[7], color='orange', label='CS6 revenue')
    plt.plot(range(len(price_list[8])), price_list[8], color='purple', label='CS6 revenue')
    plt.plot(range(len(price_list[9])), price_list[9], color='pink', label='CS6 revenue')
    plt.legend()  # 显示图例
    plt.show()

