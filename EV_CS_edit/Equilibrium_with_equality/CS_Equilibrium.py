# -*- coding:utf-8 -*-
# @Time: 2021/2/18 12:16
# @Author: xxuanzhu
# @Email: xxuanzhu@seu.edu.cn
# @File: CS_Equilibrium.py
# 此文件用于求解两个充电站之间的定价均衡

from matplotlib import pyplot as plt

from EV_Equilibrium import EvEquilibrium
from config import config

evEquilibrium = EvEquilibrium()


def cs_best_price_simulation(region, dist_vector_list, vehicle_num, region_num,
                             price, minimize_res_list, strategy_vector_list):
    strategy = evEquilibrium.best_response_simulation(region, dist_vector_list, vehicle_num,
                                                      price, minimize_res_list,
                                                      strategy_vector_list)
    # revenue_temp = 0
    f_i = [0, 0]
    for cs in range(config.cs_num):
        for item in range(region_num):
            f_i[cs] = f_i[cs] + vehicle_num[item] * strategy[item][cs]

    revenue1 = f_i[0] * price[0]
    revenue2 = f_i[1] * price[1]

    return revenue1, revenue2


def get_new_price(init_price, price_minus_delta, price_plus_delta, revenue_minus_delta,
                  revenue, revenue_plus_delta):
    if max(revenue_minus_delta, revenue, revenue_plus_delta) == revenue_plus_delta:
        return price_plus_delta
    elif max(revenue_minus_delta, revenue, revenue_plus_delta) == revenue_minus_delta:
        return price_minus_delta
    else:
        return init_price


if __name__ == "__main__":
    # 初始化操作, 假设目前有1个公司控制2个充电站
    epision = 0.000001
    delta = 1
    region_num = 5
    minimize_res_list = []
    p_min = 10
    p_max = 200
    price = [30, 30]
    # for cs in range(config.cs_num):
    #     price.append(random.randint(p_min, p_max))
    # vehicle_vector = np.array([20, 30, 30, 50, 10])

    dist, vehicle_num, region, strategy_vector = evEquilibrium.initiation(region_num, config.cs_num)
    print("初始化dist集合, 车辆数， agent, 策略向量集合, 价格集合分别为：", dist, vehicle_num,
          region, strategy_vector, price)

    revenue_list1 = []
    revenue_list2 = []

    # 计算CS1和CS2的初始收益
    # init_price1 = random.randint(p_min, p_max)
    # init_price2 = random.randint(p_min, p_max)
    revenue1, revenue2 = cs_best_price_simulation(region, dist, vehicle_num, region_num,
                                                  price, minimize_res_list, strategy_vector)
    revenue_list1.append(revenue1)
    revenue_list2.append(revenue2)

    # 变化price1，增加delta或者减少delta
    flag1 = revenue1
    flag1_origin = price[0]

    price1_minus_delta = price[0] - delta if price[0] - delta >= p_min else p_min
    price1_plus_delta = price[0] + delta if price[0] + delta <= p_max else p_max
    price[0] = price1_minus_delta
    revenue1_minus_delta, revenue2 = cs_best_price_simulation(region, dist, vehicle_num, region_num,
                                                              price, minimize_res_list, strategy_vector)
    price[0] = price1_plus_delta
    revenue1_plus_delta, revenue2 = cs_best_price_simulation(region, dist, vehicle_num, region_num, price,
                                                             minimize_res_list,
                                                             strategy_vector)
    # 计算更新后的CS1最大收益值
    revenue1_new = max(revenue1_minus_delta, revenue1, revenue1_plus_delta)
    price1_new = get_new_price(flag1_origin, price1_minus_delta, price1_plus_delta,
                               revenue1_minus_delta, revenue1, revenue1_plus_delta)
    price[0] = price1_new

    # 变化price2，增加delta或者减少delta
    flag2 = revenue2
    flag2_origin = price[1]
    price2_minus_delta = price[1] - delta if price[1] - delta >= p_min else p_min
    price2_plus_delta = price[1] + delta if price[1] + delta <= p_max else p_max
    price[1] = price2_minus_delta
    revenue1, revenue2_minus_delta = cs_best_price_simulation(region, dist, vehicle_num, region_num, price,
                                                              minimize_res_list,
                                                              strategy_vector)
    price[1] = price2_plus_delta
    revenue1, revenue2_plus_delta = cs_best_price_simulation(region, dist, vehicle_num, region_num, price,
                                                             minimize_res_list, strategy_vector)
    # 计算更新后的CS2最大值收益值
    revenue2_new = max(revenue2_minus_delta, revenue2, revenue2_plus_delta)
    price2_new = get_new_price(flag2_origin, price2_minus_delta, price2_plus_delta,
                               revenue2_minus_delta, revenue2, revenue2_plus_delta)
    price[1] = price2_new

    revenue_list1.append(revenue1_new)
    revenue_list2.append(revenue2_new)
    # 到上面这一步为止，算出来了初始的price下不同CS的收益，以及经过一次迭代后的不同CS的收益
    # 下面用while循环继续进行迭代，直到两个CS的收益均收敛
    while revenue1_new - flag1 > epision or revenue2_new - flag2 > epision:
        flag1 = revenue1_new
        flag2 = revenue2_new
        # 变化price1，增加delta或者减少delta
        flag1_origin = price[0]

        price1_minus_delta = price[0] - delta if price[0] - delta >= p_min else p_min
        price1_plus_delta = price[0] + delta if price[0] + delta <= p_max else p_max
        price[0] = price1_minus_delta
        revenue1_minus_delta, revenue2 = cs_best_price_simulation(region, dist, vehicle_num, region_num,
                                                                  price, minimize_res_list, strategy_vector)
        price[0] = price1_plus_delta
        revenue1_plus_delta, revenue2 = cs_best_price_simulation(region, dist, vehicle_num, region_num,
                                                                 price, minimize_res_list,
                                                                 strategy_vector)
        # 计算更新后的CS1最大收益值
        revenue1_new = max(revenue1_minus_delta, revenue1, revenue1_plus_delta)
        price1_new = get_new_price(flag1_origin, price1_minus_delta, price1_plus_delta,
                                   revenue1_minus_delta, revenue1, revenue1_plus_delta)

        price[0] = price1_new

        # 变化price2，增加delta或者减少delta
        flag2_origin = price[1]
        price2_minus_delta = price[1] - delta if price[1] - delta >= p_min else p_min
        price2_plus_delta = price[1] + delta if price[1] + delta <= p_max else p_max
        price[1] = price2_minus_delta
        revenue1, revenue2_minus_delta = cs_best_price_simulation(region, dist, vehicle_num, region_num,
                                                                  price, minimize_res_list,
                                                                  strategy_vector)
        price[1] = price2_plus_delta
        revenue1, revenue2_plus_delta = cs_best_price_simulation(region, dist, vehicle_num, region_num,
                                                                 price,
                                                                 minimize_res_list, strategy_vector)
        # 计算更新后的CS2最大值收益值
        revenue2_new = max(revenue2_minus_delta, revenue2, revenue2_plus_delta)
        price2_new = get_new_price(flag2_origin, price2_minus_delta, price2_plus_delta,
                                   revenue2_minus_delta, revenue2, revenue2_plus_delta)

        price[1] = price2_new
        revenue_list1.append(revenue1_new)
        revenue_list2.append(revenue2_new)

    print(revenue_list1)
    print(revenue_list2)
    print(price[0])
    print(price[1])

    plt.title("CS1 and CS2 revenue curve")
    plt.xlabel("price")
    plt.ylabel("revenue")
    plt.plot(range(len(revenue_list1)), revenue_list1, color='red', label='CS1 revenue')
    plt.plot(range(len(revenue_list1)), revenue_list2, color='green', label='CS2 revenue')
    plt.legend()  # 显示图例
    plt.show()
