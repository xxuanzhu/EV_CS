# -*- coding:utf-8 -*-
# @Time: 2021/2/18 12:16
# @Author: xxuanzhu
# @Email: xxuanzhu@seu.edu.cn
# @File: CS_my_10cs.py
# 此文件用于求解10个充电站之间的定价均衡
import random

from matplotlib import pyplot as plt

from EV_Equilibrium_10cs import EvEquilibrium
from config import config

evEquilibrium = EvEquilibrium()


def cs_best_price_simulation(region, dist_vector_list, vehicle_num, region_num, changable_cs_number,
                             price, p_min, p_max, minimize_res_list, strategy_vector_list):
    strategy = evEquilibrium.best_response_simulation(region, dist_vector_list, vehicle_num,
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

        new_strategy = evEquilibrium.best_response_simulation(region, dist_vector_list, vehicle_num,
                                                              price, minimize_res_list,
                                                              strategy_vector_list)

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
    p_max = 200
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
    flag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for cs in range(config.cs_num):
        flag[cs] = price[cs]
    # flag1 = price[0]
    # flag2 = price[1]
    all_cs_price_list = []

    p1_list = []
    p2_list = []
    p1_list.append(price[0])
    p2_list.append(price[1])
    # revenue1_max = 0
    # revenue2_max = 0

    new_price1 = cs_best_price_simulation(region, dist, vehicle_num, region_num, 0,
                                          price, p_min, p_max, minimize_res_list, strategy_vector)
    price[0] = new_price1

    new_price2 = cs_best_price_simulation(region, dist, vehicle_num, region_num, 1, price, p_min, p_max,
                                          minimize_res_list, strategy_vector)
    price[1] = new_price2

    p1_list.append(new_price1)
    p2_list.append(new_price2)

    while new_price1 - flag1 > epision or new_price2 - flag2 > epision:
        flag1 = new_price1
        flag2 = new_price2

        new_price1 = cs_best_price_simulation(region, dist, vehicle_num, region_num, 0,
                                              price, p_min, p_max, minimize_res_list, strategy_vector)
        price[0] = new_price1

        new_price2 = cs_best_price_simulation(region, dist, vehicle_num, region_num, 1, price, p_min, p_max,
                                              minimize_res_list, strategy_vector)
        price[1] = new_price2

        p1_list.append(new_price1)
        p2_list.append(new_price2)

    print(price[0])
    print(price[1])
    print(p1_list)
    print(p2_list)

    plt.title("CS1 and CS2 price curve")
    plt.xlabel("CDF")
    plt.ylabel("price")
    plt.plot(range(len(p1_list)), p1_list, color='red', label='CS1 revenue')
    # plt.plot(range(len(revenue_list1)), revenue_list1, color='red', label='CS1 revenue')
    # plt.plot(range(len(revenue_list1)), revenue_list2, color='green', label='CS2 revenue')
    plt.plot(range(len(p2_list)), p2_list, color='green', label='CS2 revenue')
    plt.legend()  # 显示图例
    plt.show()

    # price1_minus_delta = price[0] - delta if price[0] - delta >= p_min else p_min
    # price1_plus_delta = price[0] + delta if price[0] + delta <= p_max else p_max

    # price1变小后，cs1和cs2的新收入
    # price[0] = price1_minus_delta

    # price1按照delta变大后，cs1和cs2的新收入
    # price[0] = price1_plus_delta
    # revenue1_plus_delta, revenue2_after_p1_plus = cs_best_price_simulation(region, dist, vehicle_num, region_num, price,
    #                                                                        minimize_res_list,
    #                                                                        strategy_vector)

    # revenue1_new = max(revenue1_after_change, revenue1, revenue1_plus_delta)
    # price1_new = get_new_price(flag1_origin, price1_minus_delta, price1_plus_delta,
    #                            revenue1_after_change, revenue1, revenue1_plus_delta)
    # revenue2_new = get_new_revenue(price1_new, flag1_origin, price1_minus_delta, price1_plus_delta,
    #                                revenue2_after_p1_change, revenue2, revenue2_after_p1_plus)

    # price[0] = price1_new
    # price1_list.append(price[0])

    # while revenue1_new - flag1 > epision or revenue2_new - flag2 > epision:
    #     flag1 = revenue1_new
    #     flag2 = revenue2_new
    #     flag1_origin = price[0]
    #
    #     price1_minus_delta = price[0] - delta if price[0] - delta >= p_min else p_min
    #     price1_plus_delta = price[0] + delta if price[0] + delta <= p_max else p_max
    #
    #     # price1变小后，cs1和cs2的新收入
    #     price[0] = price1_minus_delta
    #     revenue1_after_change, revenue2_after_p1_change = cs_best_price_simulation(region, dist, vehicle_num, region_num,
    #                                                                                price, minimize_res_list,
    #                                                                                strategy_vector)
    #
    #     # price1按照delta变大后，cs1和cs2的新收入
    #     price[0] = price1_plus_delta
    #     # price1_list.append(price[0])
    #     revenue1_plus_delta, revenue2_after_p1_plus = cs_best_price_simulation(region, dist, vehicle_num, region_num,
    #                                                                            price,
    #                                                                            minimize_res_list,
    #                                                                            strategy_vector)
    #
    #     revenue1_new = max(revenue1_after_change, revenue1, revenue1_plus_delta)
    #     price1_new = get_new_price(flag1_origin, price1_minus_delta, price1_plus_delta,
    #                                revenue1_after_change, revenue1, revenue1_plus_delta)
    #     revenue2_new = get_new_revenue(price1_new, flag1_origin, price1_minus_delta, price1_plus_delta,
    #                                    revenue2_after_p1_change, revenue2, revenue2_after_p1_plus)
    #
    #     price[0] = price1_new
    #     price1_list.append(price[0])
    #     revenue1_list.append(revenue1_new)
    #     revenue2_list.append(revenue2_new)

    # print(revenue1_list)
    # print(revenue2_list)
    # print(price[0])
    # print(price[1])

    # plt.title("CS1 and CS2 revenue curve")
    # plt.xlabel("price")
    # plt.ylabel("revenue")
    # plt.plot(price1_list, revenue1_list, color='red', label='CS1 revenue')
    # # plt.plot(range(len(revenue_list1)), revenue_list1, color='red', label='CS1 revenue')
    # # plt.plot(range(len(revenue_list1)), revenue_list2, color='green', label='CS2 revenue')
    # plt.plot(price1_list, revenue2_list, color='green', label='CS2 revenue')
    # plt.legend()  # 显示图例
    # plt.show()
