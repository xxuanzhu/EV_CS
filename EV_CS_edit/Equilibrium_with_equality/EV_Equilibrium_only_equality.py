#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/3/6 16:10
# @Author : wbw
# @Version：V 0.1
# @File : EV_Equilibrium_only_equality.py
# @desc : 车流量在没有不等式约束的最优情况

import copy
import random
from random import shuffle

import numpy as np
from scipy.optimize import minimize
from config import config


class EvEquilibriumCompare(object):

    def fun_vehicles(self, args):
        v_i, dist_1, dist_2, f_minus_i_1, f_minus_i_2, p_1, p_2 = args
        v = lambda x: (p_1 + dist_1 + v_i * x[0] + f_minus_i_1) * v_i * x[0] + (p_2 + dist_2 + v_i * x[1] + f_minus_i_2) \
                      * v_i * x[1]
        return v

    def con_strategy(self, x_max):
        # 约束条件 分为eq 和ineq
        # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0
        cons = ({'type': 'eq', 'fun': lambda x: x_max - x[0] - x[1]})
        return cons

    def initiation(self, region_num):
        region_list = []

        dist_vector1 = config.dist_vector1
        dist_vector2 = config.dist_vector2

        vehicle_vector = config.vehicle_vector

        strategy_vector1 = np.random.rand(region_num)
        strategy_vector2 = np.random.rand(region_num)

        for i in range(region_num):
            region_list.append(int(i))
        return dist_vector1, dist_vector2, vehicle_vector, region_list, strategy_vector1, strategy_vector2

    def agent_best_response(self, agent, region_list, dist_1, dist_2, vehicle_vector, strategy_vector1, strategy_vector2,
                          p_1, p_2, minimize_res_list):
        v_i = vehicle_vector[agent]  # i区域内的汽车数量
        f_minus_i_1, f_minus_i_2 = 0, 0
        for item in region_list:
            if item != agent:  # 不是当前的区域
                f_minus_i_1 = f_minus_i_1 + vehicle_vector[item] * strategy_vector1[item]  # 除了i区域外其他区域派到充电站1的车辆数量
                f_minus_i_2 = f_minus_i_2 + vehicle_vector[item] * strategy_vector2[item]  # 除了i区域外其他区域派到充电站2的车辆数量
        f_minus_i_1 = round(f_minus_i_1, 0)
        f_minus_i_2 = round(f_minus_i_2, 0)
        # print("当前区域i的车辆数量 v_i为： ", v_i, "除i区域外到其他区域派到CS1的车辆数量 f-i及 到CS2的数量：", f_minus_i_1, f_minus_i_2)
        args = (v_i, dist_1, dist_2, f_minus_i_1, f_minus_i_2, p_1, p_2)
        cons = self.con_strategy(1)
        # 设置初始猜测值
        fun = self.fun_vehicles(args)
        x0 = np.asarray((0.5, 0.5))
        res = minimize(fun, x0, method='SLSQP', constraints=cons)
        minimize_res_list.append(res.fun)
        print("车辆最小化结果： ", res.fun, res.success, res.x)

        return round(res.x[0], 2), round(res.x[1], 2)

    def best_response_simulation(self, region_list, dist_vector1, dist_vector2, vehicle_vector, p_1, p_2,
                                minimize_res_list, strategy_vector1, strategy_vector2):
        epision = 0.000001
        num = 1
        round_num = num

        print("开始第", num, "轮更新：")
        flag1 = copy.deepcopy(strategy_vector1)
        flag2 = copy.deepcopy(strategy_vector2)
        new_region_list = copy.deepcopy(region_list)
        shuffle(new_region_list)  # 将序列的所有元素随机排序

        for agent in new_region_list:
            dist_1 = dist_vector1[agent]
            dist_2 = dist_vector2[agent]

            strategy_vector1[agent], strategy_vector2[agent] = self.agent_best_response(agent, region_list, dist_1, dist_2,
                                                                                 vehicle_vector, strategy_vector1,
                                                                                 strategy_vector2, p_1, p_2,
                                                                                 minimize_res_list)

        while (np.linalg.norm(flag1 - strategy_vector1) > epision or np.linalg.norm(
                flag2 - strategy_vector2) > epision):  # 求二范数
            num = num + 1

            print("开始第", num, "轮更新：")
            flag1 = copy.deepcopy(strategy_vector1)
            flag2 = copy.deepcopy(strategy_vector2)
            new_region_list = copy.deepcopy(region_list)
            shuffle(new_region_list)
            for agent in new_region_list:
                dist_1 = dist_vector1[agent]
                dist_2 = dist_vector2[agent]

                strategy_vector1[agent], strategy_vector2[agent] = self.agent_best_response(agent, region_list,
                                                                                     dist_1, dist_2,
                                                                                     vehicle_vector,
                                                                                     strategy_vector1,
                                                                                     strategy_vector2,
                                                                                     p_1, p_2, minimize_res_list)

        print("均衡下的策略向量为：", strategy_vector1, strategy_vector2, "\n")
        return strategy_vector1, strategy_vector2


evEquilibriumCompare = EvEquilibriumCompare()


if __name__ == "__main__":
    # 初始化操作, 假设目前有1个公司控制2个充电站
    threshold = 0.000001  # 迭代截止阈值
    p_1 = 30
    p_2 = 10
    region_num = config.region_num
    minimize_res_list = []
    dist1, dist2, vehicle_num, region, strategy_vector1, strategy_vector2 = evEquilibriumCompare.initiation(region_num)
    print("初始化dist1, dist2, 车辆数， agent, 策略向量1， 策略向量2分别为：", dist1, dist2, vehicle_num,
          region, strategy_vector1, strategy_vector2)

    # 求得此定价下底层的策略
    strategy1, strategy2 = evEquilibriumCompare.best_response_simulation(region, dist1, dist2, vehicle_num,
                                                   p_1, p_2, minimize_res_list, strategy_vector1, strategy_vector2)

    print("CS1的策略：", strategy1)
    print("CS2的策略：", strategy2)
