# -*- coding:utf-8 -*-
# @Time: 2021/2/18 12:16
# @Author: beilwu
# @Email: beilwu@seu.edu.cn
# @File: CS_Equilibrium_10cs_change_regions.py
# 此文件用于变化区域数量，求解公司的收益曲线
import copy

from matplotlib import pyplot as plt

from config import config
from new_experiment_company_max.EV_Equilibrium_10cs import EvEquilibrium

evEquilibrium = EvEquilibrium()


def cs_best_price_simulation(region, dist_vector_list, vehicle_num, region_num, changable_cs_number,
                             price, minimize_res_list, strategy_vector_list):
    strategy, signal = evEquilibrium.best_response_simulation(region, dist_vector_list, vehicle_num,
                                                              price, minimize_res_list,
                                                              strategy_vector_list)

    f_i = 0
    for item in range(region_num):
        f_i = f_i + vehicle_num[item] * strategy[item][changable_cs_number]

    revenue_new = f_i * price[changable_cs_number]

    return revenue_new


def get_new_price(init_price, price_minus_delta, price_plus_delta, revenue_minus_delta,
                  revenue_origin, revenue_plus_delta):
    if max(revenue_minus_delta, revenue_origin, revenue_plus_delta) == revenue_plus_delta:
        return price_plus_delta
    elif max(revenue_minus_delta, revenue_origin, revenue_plus_delta) == revenue_minus_delta:
        return price_minus_delta
    else:
        return init_price


def change_price(region, dist, vehicle_num, region_num, change_price_cs, price, p_min, p_max, minimize_res_list,
                 strategy_vector, all_revenue_list):
    flag_price = price[change_price_cs]

    price_minus_delta = price[change_price_cs] - delta if price[change_price_cs] - delta >= p_min else p_min
    price_plus_delta = price[change_price_cs] + delta if price[change_price_cs] + delta <= p_max else p_max

    price[change_price_cs] = price_minus_delta
    revenue_minus_delta = cs_best_price_simulation(region, dist, vehicle_num, region_num, change_price_cs,
                                                   price, minimize_res_list, strategy_vector)
    price[change_price_cs] = price_plus_delta
    revenue_plus_delta = cs_best_price_simulation(region, dist, vehicle_num, region_num, change_price_cs,
                                                  price, minimize_res_list,
                                                  strategy_vector)
    # 计算更新后的CS1最大收益值
    revenue_new = max(revenue_minus_delta, all_revenue_list[change_price_cs][-1], revenue_plus_delta)
    price_new = get_new_price(flag_price, price_minus_delta, price_plus_delta,
                              revenue_minus_delta, all_revenue_list[change_price_cs][-1], revenue_plus_delta)

    price[change_price_cs] = price_new
    all_revenue_list[change_price_cs].append(revenue_new)


if __name__ == "__main__":
    # 初始化操作, 假设目前有1个公司控制2个充电站
    epision = 0.000001
    delta = 1
    region_num = 5
    minimize_res_list = []
    p_min = 10
    p_max = 200
    price = [20, 20, 20, 20, 20]  # 前四个为一个公司控制的充电桩，后6个价格固定

    all_revenue_list = [[] for i in range(config.cs_num)]  # 所有充电站的revenue列表

    own_cs_revenue_list = [[] for i in range(11)]  # 不同region数量下，公司控制的充电站收益之和列表

    # 变化区域
    for region_symbol in range(11):
        config.change_region_num(region_symbol+5)  # 改变region数目

        dist, vehicle_num, region, strategy_vector = evEquilibrium.initiation(region_num, config.cs_num)
        print("初始化dist集合, 车辆数， agent, 策略向量集合, 价格集合分别为：", dist, vehicle_num,
              region, strategy_vector, price)

        own_cs_revenue = 0  # 本公司在当前region_num下的收益和

        # 求当前cs的收益
        for cs in range(config.cs_num):
            revenue_cs = cs_best_price_simulation(region, dist, vehicle_num, region_num, cs,
                                                  price, minimize_res_list, strategy_vector)
            # 如果是本公司控制的4个充电站
            if cs < 4:
                own_cs_revenue += revenue_cs
            own_cs_revenue_list[region_symbol].append(own_cs_revenue)
            all_revenue_list[cs].append(revenue_cs)

        # 变化price
        flag_revenue = copy.deepcopy(all_revenue_list)  # 保存所有cs的收益列表
        flag_own = own_cs_revenue  # 保存当前公司控制的充电站的收益之和列表
        own_cs_revenue = 0

        for cs in range(4):
            change_price(region, dist, vehicle_num, region_num, cs, price, p_min, p_max, minimize_res_list,
                         strategy_vector,
                         all_revenue_list)
        for cs in range(4):
            own_cs_revenue += all_revenue_list[cs][-1]
        own_cs_revenue_list[region_symbol].append(own_cs_revenue)

        for cs in range(4, config.cs_num):
            revenue_cs = cs_best_price_simulation(region, dist, vehicle_num, region_num, cs,
                                                  price, minimize_res_list, strategy_vector)
            all_revenue_list[cs].append(revenue_cs)

        # 到上面这一步为止，算出来了初始的price下不同CS的收益，以及经过一次迭代后的不同CS的收益
        # 下面用while循环继续进行迭代，直到两个CS的收益均收敛
        while abs(flag_own - own_cs_revenue) > epision:

            flag_revenue = copy.deepcopy(all_revenue_list)
            flag_own = own_cs_revenue
            own_cs_revenue = 0

            for cs in range(4):
                change_price(region, dist, vehicle_num, region_num, cs, price, p_min, p_max, minimize_res_list,
                             strategy_vector,
                             all_revenue_list)

            for cs in range(4):
                own_cs_revenue += all_revenue_list[cs][-1]
            own_cs_revenue_list[region_symbol].append(own_cs_revenue)

            for cs in range(4, config.cs_num):
                revenue = cs_best_price_simulation(region, dist, vehicle_num, region_num, cs,
                                                   price, minimize_res_list, strategy_vector)
                all_revenue_list[cs].append(revenue)

        print("当前区域数量为：", region_symbol+5, "的情况下,", "其收益之和list为：", own_cs_revenue_list[region_symbol])
        for i in range(4):
            print("我公司控制的cs ", i, " 的定价为：", price[i])

    color = ['red', 'green', 'blue', 'black', 'yellow', 'gray', 'chocolate', 'orange', 'purple', 'pink', 'lightgreen']
    plt.title("company revenue curve under different region nums")
    plt.xlabel("num")
    plt.ylabel("revenue")
    for symbol in range(11):
        plt.plot(range(len(own_cs_revenue_list[symbol])), own_cs_revenue_list[symbol], color=color[symbol], label='region is '+str(symbol+5)+" revenue curve")
    # plt.plot(range(len(all_revenue_list[1])), all_revenue_list[1], color='green', label='CS2 revenue')
    # plt.plot(range(len(all_revenue_list[2])), all_revenue_list[2], color='blue', label='CS3 revenue')
    # plt.plot(range(len(all_revenue_list[3])), all_revenue_list[3], color='black', label='CS4 revenue')
    # plt.plot(range(len(all_revenue_list[4])), all_revenue_list[4], color='yellow', label='CS5 revenue')
    # plt.plot(range(len(all_revenue_list[5])), all_revenue_list[5], color='gray', label='CS6 revenue')
    # plt.plot(range(len(all_revenue_list[6])), all_revenue_list[6], color='chocolate', label='CS6 revenue')
    # plt.plot(range(len(all_revenue_list[7])), all_revenue_list[7], color='orange', label='CS6 revenue')
    # plt.plot(range(len(all_revenue_list[8])), all_revenue_list[8], color='purple', label='CS6 revenue')
    # plt.plot(range(len(all_revenue_list[9])), all_revenue_list[9], color='pink', label='CS6 revenue')
    plt.legend()  # 显示图例
    plt.show()
