# -*- coding:utf-8 -*-
# @Time: 2021/2/18 12:10
# @Author: beilwu
# @Email: beilwu@seu.edu.cn
# @File: optimal_pricing.py

import numpy as np
from matplotlib import pyplot as plt

from EV_Equilibrium import EvEquilibrium
from config import config

evEquilibrium = EvEquilibrium()

if __name__ == "__main__":
    # 初始化操作, 假设目前有1个公司控制2个充电站
    # p_1 = 178

    epision = 0.000001
    # p_2 = 154
    region_num = 5
    minimize_res_list = []
    # vehicle_vector = np.array([20, 30, 30, 50, 10])
    price = [30, 168]

    dist, vehicle_num, region, strategy_vector = evEquilibrium.initiation(region_num, config.cs_num)
    print("初始化dist集合, 车辆数， agent, 策略向量集合, 价格集合分别为：", dist, vehicle_num,
          region, strategy_vector, price)

    revenue_list = []
    revenue_list_second = []
    price1_list = np.arange(1, 200)
    best_price = price[0]
    price1_final_list = []
    last_revenue = 0
    # 计算CS1的收益 V_1 = p_1 * f_1
    for p_1 in range(1, 200):
        print("当前定价", p_1)
        # 求得此定价下底层的策略
        price[0] = p_1
        price1_final_list.append(p_1)
        strategy = evEquilibrium.best_response_simulation(region, dist, vehicle_num,
                                                          price, minimize_res_list, strategy_vector)
        print("CS1的策略：", strategy_vector[0])
        # revenue_temp = 0
        f_i = [0, 0]
        for cs in range(config.cs_num):
            for item in range(region_num):
                f_i[cs] = f_i[cs] + vehicle_num[item] * strategy[item][cs]

        revenue_temp = price[0] * f_i[0]
        revenue_temp_second = price[1] * f_i[1]

        if revenue_temp > last_revenue:
            last_revenue = revenue_temp
            best_price = p_1

        revenue_list.append(revenue_temp)
        revenue_list_second.append(revenue_temp_second)

    print(best_price)
    print(price[1])

    plt.title("CS1 revenue curve")
    plt.xlabel("price")
    plt.ylabel("revenue")
    plt.plot(price1_final_list, revenue_list, color='red', label='CS1 revenue')
    plt.plot(price1_final_list, revenue_list_second, color='green', label='CS2 revenue')
    plt.legend()  # 显示图例
    plt.show()
