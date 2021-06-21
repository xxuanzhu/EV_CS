# -*- coding:utf-8 -*-
# @Time: 2021/2/18 12:16
# @Author: xxuanzhu
# @Email: xxuanzhu@seu.edu.cn
# @File: CS_vertify_jump.py
# 此文件用于验证跳跃点的正确性
import copy
import random

from matplotlib import pyplot as plt

from jump_experiment.EV_Equilibrium_only_equal import EvEquilibrium
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
        print("revenue is ", revenue_temp)

        if revenue_temp > revenue_before:
            revenue_before = revenue_temp
            best_price = p

    return best_price, revenue_before


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
    p_list = [10, 48]
    p_left = 0
    p_right = 1
    price = [10, 10, 10, 10, 10, 10, 10, 10]
    # price1_list = []
    # price1_list.append(50)
    # for cs in range(config.cs_num):
    #     price.append(random.randint(p_min, p_max))
    minimize_res_list = []
    revenue_list = []

    dist, vehicle_num, region, strategy_vector = evEquilibrium.initiation(config.region_num, config.cs_num)
    print("初始化dist集合, 车辆数， agent, 策略向量集合, 价格集合分别为：", dist, vehicle_num,
          region, strategy_vector, price)

    revenue_max = 0
    best_price = 0

    while p_right < len(p_list):
        for p in range(p_list[p_left], p_list[p_right], 1):
            # print("当前变化的", 0, "号充电桩定价：", p)
            price[0] = p

            new_strategy, signal = evEquilibrium.best_response_simulation(region, dist, vehicle_num,
                                                                          price, minimize_res_list,
                                                                          strategy_vector)

            # new_strategy = list(new_strategy)

            new_f_i = 0
            for item in range(config.region_num):
                new_f_i = new_f_i + vehicle_num[item] * new_strategy[item][
                    0]

            revenue_temp = p * new_f_i
            revenue_list.append(revenue_temp)
            # print("revenue is ", revenue_temp)

            if revenue_temp > revenue_max:
                revenue_max = revenue_temp
                best_price = p

        print("在价格区间为[", p_list[p_left], ",", p_list[p_right],")，时：")
        print("0号充电站的最优定价为：", best_price, "最佳收益为 ", revenue_max)
        # print(revenue_list)
        p_left += 1
        p_right += 1

    plt.title("CS1 price and revneue")
    plt.xlabel("CDF")
    plt.ylabel("price")
    plt.plot(range(p_list[0], p_list[len(p_list)-1]), revenue_list, color='red', label='CS1 revenue')
    # plt.plot(range(len(price_list[1])), price_list[1], color='green', label='CS2 revenue')
    plt.legend()  # 显示图例
    plt.show()

