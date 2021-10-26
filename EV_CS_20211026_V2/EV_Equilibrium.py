# -*- coding:utf-8 -*-
'''
@Project ：EV_CS
@File    ：EV_Equilibrium.py
@Author  ：xxuanZhu
@Date    ：2021/6/4 9:58
@Purpose : 底层（车辆）均衡约束，已经过重构
'''

import copy
import random
from random import shuffle

import numpy as np
from scipy.optimize import minimize

from config import config


def trans(m):
    return list(zip(*m))


class EvEquilibrium(object):

    def fun_vehicles(self, args):
        v_i, dist_agent_to_cs_list, f_minus_i, price_list = args
        # v_i, dist_1, dist_2, f_minus_i_1, f_minus_i_2, p_1, p_2 = args
        # for cs in range(config.cs_num):
        #     v = lambda x: (price_list[cs] + dist_agent_to_cs_list[cs] + v_i * x[cs] + f_minus_i[cs]) * v_i * x[cs]

        # zhx重构
        v = lambda x: sum([((price_list[k] + dist_agent_to_cs_list[k] + v_i * x[k] + f_minus_i[k]) * v_i * x[k])
                           for k in range(0, config.cs_num)])

        # v = lambda x: (price_list[0] + dist_agent_to_cs_list[0] + v_i * x[0] + f_minus_i[0]) * v_i * x[0] + \
        #               (price_list[1] + dist_agent_to_cs_list[1] + v_i * x[1] + f_minus_i[1]) * v_i * x[1] + \
        #               (price_list[2] + dist_agent_to_cs_list[2] + v_i * x[2] + f_minus_i[2]) * v_i * x[2] + \
        #               (price_list[3] + dist_agent_to_cs_list[3] + v_i * x[3] + f_minus_i[3]) * v_i * x[3]
                      # (price_list[4] + dist_agent_to_cs_list[4] + v_i * x[4] + f_minus_i[4]) * v_i * x[4] + \
                      # (price_list[5] + dist_agent_to_cs_list[5] + v_i * x[5] + f_minus_i[5]) * v_i * x[5] +\
                      # (price_list[6] + dist_agent_to_cs_list[6] + v_i * x[6] + f_minus_i[6]) * v_i * x[6] + \
                      # (price_list[7] + dist_agent_to_cs_list[7] + v_i * x[7] + f_minus_i[7]) * v_i * x[7] +\
                      # (price_list[8] + dist_agent_to_cs_list[8] + v_i * x[8] + f_minus_i[8]) * v_i * x[8] +\
                      # (price_list[9] + dist_agent_to_cs_list[9] + v_i * x[9] + f_minus_i[9]) * v_i * x[9]
                      # (price_list[7] + dist_agent_to_cs_list[10] + v_i * x[10] + f_minus_i[10]) * v_i * x[10] +\
                      # (price_list[7] + dist_agent_to_cs_list[11] + v_i * x[11] + f_minus_i[11]) * v_i * x[11] +\
                      # (price_list[7] + dist_agent_to_cs_list[12] + v_i * x[12] + f_minus_i[12]) * v_i * x[12] +\
                      # (price_list[7] + dist_agent_to_cs_list[13] + v_i * x[13] + f_minus_i[13]) * v_i * x[13] +\
                      # (price_list[7] + dist_agent_to_cs_list[14] + v_i * x[14] + f_minus_i[14]) * v_i * x[14]

        return v

    def con_strategy(self, x_min, x_max):
        # 约束条件 分为eq 和ineq
        # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0

        cs_num = config.cs_num
        # zhx重构，先创建列表动态添加元素，最后转换为元组返回即可，不需要一开始就元组
        cons = [{'type': 'eq',
                 # 'fun': lambda x: x_max - x[0] - x[1] - x[2] - x[3]  # - x[4] - x[5] - x[6] - x[7] - x[8] - x[9]
                 'fun': lambda x: x_max - sum(x[i] for i in range(cs_num))
                   }]
        for i in range(0, cs_num):
            cons.append({'type': 'ineq', 'fun': lambda x,i=i: x[i] - x_min})
            cons.append({'type': 'ineq', 'fun': lambda x,i=i: x_max - x[i]})
        # cons.append({'type': 'ineq', 'fun': lambda x: x[0] - x_min})
        # cons.append({'type': 'ineq', 'fun': lambda x: x_max - x[0]})
        # cons.append({'type': 'ineq', 'fun': lambda x: x[1] - x_min})
        # cons.append({'type': 'ineq', 'fun': lambda x: x_max - x[2]})
        # cons.append({'type': 'ineq', 'fun': lambda x: x[2] - x_min})
        # cons.append({'type': 'ineq', 'fun': lambda x: x_max - x[0]})
        # cons.append({'type': 'ineq', 'fun': lambda x: x[3] - x_min})
        # cons.append({'type': 'ineq', 'fun': lambda x: x_max - x[3]})
        return tuple(cons)
        # cons = (
        #     {'type': 'eq',
        #      'fun': lambda x: x_max - x[0] - x[1] - x[2] - x[3]
        #                       # - x[10] - x[11] - x[12] - x[13] - x[14],
        #      },
        #     {'type': 'ineq', 'fun': lambda x: x[0] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[0]},
        #     {'type': 'ineq', 'fun': lambda x: x[1] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[1]},
        #     {'type': 'ineq', 'fun': lambda x: x[2] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[2]},
        #     {'type': 'ineq', 'fun': lambda x: x[3] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[3]}
            # {'type': 'ineq', 'fun': lambda x: x[4] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[4]},
            # {'type': 'ineq', 'fun': lambda x: x[5] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[5]},
            # {'type': 'ineq', 'fun': lambda x: x[6] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[6]},
            # {'type': 'ineq', 'fun': lambda x: x[7] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[7]},
            # {'type': 'ineq', 'fun': lambda x: x[8] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[8]},
            # {'type': 'ineq', 'fun': lambda x: x[9] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[9]},
        #     {'type': 'ineq', 'fun': lambda x: x[10] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[10]},
        #     {'type': 'ineq', 'fun': lambda x: x[11] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[11]},
        #     {'type': 'ineq', 'fun': lambda x: x[12] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[12]},
        #     {'type': 'ineq', 'fun': lambda x: x[13] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[13]},
        #     {'type': 'ineq', 'fun': lambda x: x[14] - x_min}, {'type': 'ineq', 'fun': lambda x: x_max - x[14]},
        # )
        # return cons

    def initiation(self, region_num, cs_num):
        region_list = []
        # dist_vector1 = np.arange(region_num, dtype="float64")
        # dist_vector2 = np.arange(region_num, dtype="float64")
        # dist_vector1 = np.array([10, 15, 20, 40, 24])
        # dist_vector2 = np.array([20, 30, 60, 10, 14])
        dist_vector = config.dist_vector
        # dist_vector2 = config.dist_vector2
        # dist_vector2 = np.array([10, 10, 10, 10, 10])
        # vehicle_vector = np.arange(region_num, dtype="float64")
        # vehicle_vector = np.array([20, 30, 40, 30, 50])
        # vehicle_vector = np.array([20, 20, 20, 20, 20])
        vehicle_vector = config.vehicle_vector

        # 生成策略向量, (regin_num * cs_num)
        strategy_vector = []
        for i in range(config.region_num):
            s = np.random.rand(config.cs_num)
            strategy_vector.append(s)
        # for cs in range(cs_num):

        # print(strategy_vector)

        # strategy_vector1 = np.random.rand(region_num)
        # strategy_vector2 = np.random.rand(region_num)

        for i in range(region_num):
            # dist_vector1[i] = round(random.uniform(1, dist_range), 0)
            # dist_vector2[i] = round(random.uniform(5, dist_range), 0)
            # vehicle_vector[i] = round(random.uniform(10, vehicle_range), 0)
            region_list.append(int(i))
        return dist_vector, vehicle_vector, region_list, strategy_vector

    # 每个区域agent的best_response
    def agent_best_response(self, agent, region_list, dist_agent_to_cs_list, vehicle_vector, strategy_vector_list,
                            price_list, minimize_res_list):
        v_i = vehicle_vector[agent]  # i区域内的汽车数量
        # f_minus_i_1, f_minus_i_2 = 0, 0
        f_minus_i = [0 for i in range(config.cs_num)]  # 除去i区域其他区域到充电桩们的车辆数量集合 size: cs_num
        for item in region_list:  # 区域i
            if item != agent:  # 不是当前的区域
                for cs in range(config.cs_num):
                    f_minus_i[cs] = f_minus_i[cs] + vehicle_vector[item] * strategy_vector_list[item][cs]  # 除了i区域外其他区域派到充电站cs的车辆数量
                    # f_minus_i_1 = f_minus_i_1 + vehicle_vector[item] * strategy_vector1[item]  # 除了i区域外其他区域派到充电站1的车辆数量
                    # f_minus_i_2 = f_minus_i_2 + vehicle_vector[item] * strategy_vector2[item]  # 除了i区域外其他区域派到充电站2的车辆数量
        for cs in range(config.cs_num):
            f_minus_i[cs] = round(f_minus_i[cs], 0)
            # f_minus_i_1 = round(f_minus_i_1, 0)
            # f_minus_i_2 = round(f_minus_i_2, 0)
        # print("当前区域i的车辆数量 v_i为： ", v_i, "除i区域外到其他区域派到CS1的车辆数量 f-i及 到CS2的数量：", f_minus_i_1, f_minus_i_2)
        args = (v_i, dist_agent_to_cs_list, f_minus_i, price_list)
        cons = self.con_strategy(0, 1)
        # 设置初始猜测值
        fun = self.fun_vehicles(args)

        # zhx重构
        list_x0 = tuple([0.1 for i in range(0, config.cs_num)])
        x0 = np.asarray(list_x0)  # , 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
        res = minimize(fun, x0, method='SLSQP', constraints=cons)
        minimize_res_list.append(res.fun)
        # print("车辆最小化结果： ", res.fun, res.success, res.x)

        # zhx重构
        result_list = []
        for i in range(config.cs_num):
            result_list.append(round(res.x[i], 3))
        return tuple(result_list)
        #
        # x0 = np.asarray((0.1, 0.1, 0.1, 0.1))
        # res = minimize(fun, x0, method='SLSQP', constraints=cons)
        # minimize_res_list.append(res.fun)
        # #print("车辆最小化结果： ", res.fun, res.success, res.x)
        #
        # return round(res.x[0], 3), round(res.x[1], 3), round(res.x[2], 3), round(res.x[3], 3)
        # round(res.x[10], 3), round(res.x[11], 3), round(res.x[12], 3), round(res.x[13], 3), round(res.x[14], 3)

        # return np.ndarray(res.x[0], res.x[1], res.x[2])

    def best_response_simulation(self, region_list, dist_vector_list, vehicle_vector, price_list,
                                 minimize_res_list, strategy_vector_list):
        epision = 0.000001
        num = 1
        round_num = num

        # print("开始第", num, "轮更新：")
        flag_new = copy.deepcopy(np.array(list(trans(strategy_vector_list))))
        new_region_list = copy.deepcopy(region_list)
        shuffle(new_region_list)  # 将序列的所有元素随机排序

        for agent in new_region_list:  # 区域i对桩
            dist_agent_to_cs_list = []  # 某个区域i到不同桩的距离 -> [2,4,5], 2是到桩1,4是到桩2,5是到桩3
            for cs in range(config.cs_num):
                dist = dist_vector_list[cs]  # [1,2,3,4,5]
                dist_agent_to_cs_list.append(dist[agent])  # [2]

            # 求当前区域i的最优策略
            strategy_vector_list[agent] = np.array(list(self.agent_best_response(agent, region_list,
                                                                                 dist_agent_to_cs_list,
                                                                                 vehicle_vector,
                                                                                 strategy_vector_list, price_list,
                                                                                 minimize_res_list)))

            # print("计算agent ", agent, "派到桩的策略：", strategy_vector_list[agent])
            # print("此时策略向量为：", strategy_vector1, strategy_vector2, "\n")

        # 把region*cs->cs*region
        strategy_vector_list = np.array(list(trans(strategy_vector_list)))
        # new_strategy_vector = np.array(list(trans(strategy_vector_list)))
        #
        # while (np.linalg.norm(flag_new[0] - strategy_vector_list[0]) > epision or \
        #        np.linalg.norm(flag_new[1] - strategy_vector_list[1]) > epision or \
        #        np.linalg.norm(flag_new[2] - strategy_vector_list[2]) > epision or \
        #        np.linalg.norm(flag_new[3] - strategy_vector_list[3]) > epision
        #        # np.linalg.norm(flag_new[4] - strategy_vector_list[4]) > epision or \
        #        # np.linalg.norm(flag_new[5] - strategy_vector_list[5]) > epision or \
        #        # np.linalg.norm(flag_new[6] - strategy_vector_list[6]) > epision or \
        #        # np.linalg.norm(flag_new[7] - strategy_vector_list[7]) > epision or \
        #        # np.linalg.norm(flag_new[8] - strategy_vector_list[8]) > epision or \
        #        # np.linalg.norm(flag_new[9] - strategy_vector_list[9]) > epision
        #        # np.linalg.norm(flag_new[10] - strategy_vector_list[10]) > epision or \
        #        # np.linalg.norm(flag_new[11] - strategy_vector_list[11]) > epision or \
        #        # np.linalg.norm(flag_new[12] - strategy_vector_list[12]) > epision or \
        #        # np.linalg.norm(flag_new[13] - strategy_vector_list[13]) > epision or \
        #        # np.linalg.norm(flag_new[14] - strategy_vector_list[14]) > epision
        # ):  # 求二范数
        #     num = num + 1
        #     # cs * region->region * cs
        #     strategy_vector_list = np.array(list(trans(strategy_vector_list)))
        #
        #     if num > 100:
        #         print("没找到均衡！\n")
        #         return strategy_vector_list, False
        #
        #     # print("开始第", num, "轮更新：")
        #     flag_new = copy.deepcopy(np.array(list(trans(strategy_vector_list))))
        #     new_region_list = copy.deepcopy(region_list)
        #     shuffle(new_region_list)
        #     for agent in new_region_list:  # 区域i对桩
        #         dist_agent_to_cs_list = []  # 某个区域i到不同桩的距离 -> [2,4,5], 2是到桩1,4是到桩2,5是到桩3
        #         for cs in range(config.cs_num):
        #             dist = dist_vector_list[cs]  # [1,2,3,4,5]
        #             dist_agent_to_cs_list.append(dist[agent])  # [2]
        #
        #         # 求当前区域i的最优策略
        #         strategy_vector_list[agent] = self.agent_best_response(agent, region_list,
        #                                                                dist_agent_to_cs_list,
        #                                                                vehicle_vector,
        #                                                                strategy_vector_list, price_list,
        #                                                                minimize_res_list)
        #         # print("计算agent ", agent, "派到CS的策略：", strategy_vector_list[agent])
        #         # print("此时策略向量为：", strategy_vector1, strategy_vector2, "\n")
        #     # print("更新策略为：", strategy_vector1)
        #     # new_strategy_vector = np.array(list(trans(strategy_vector_list)))
        #     # region*cs->cs*region
        #     strategy_vector_list = np.array(list(trans(strategy_vector_list)))

        # zhx重构：考虑在外面计算用循环计算较复杂的表达式
        while True:
            calculate_flag = False
            for i in range(config.cs_num):
                if np.linalg.norm(flag_new[i] - strategy_vector_list[i]) > epision:
                    calculate_flag = True
                    break
            if (calculate_flag is True
                   #  np.linalg.norm(flag_new[0] - strategy_vector_list[0]) > epision or \
                   # np.linalg.norm(flag_new[1] - strategy_vector_list[1]) > epision
                    # np.linalg.norm(flag_new[2] - strategy_vector_list[2]) > epision or \
                    # np.linalg.norm(flag_new[3] - strategy_vector_list[3]) > epision or \
                    # np.linalg.norm(flag_new[4] - strategy_vector_list[4]) > epision or \
                    # np.linalg.norm(flag_new[5] - strategy_vector_list[5]) > epision or \
                    # np.linalg.norm(flag_new[6] - strategy_vector_list[6]) > epision or \
                    # np.linalg.norm(flag_new[7] - strategy_vector_list[7]) > epision or \
                    # np.linalg.norm(flag_new[8] - strategy_vector_list[8]) > epision or \
                    # np.linalg.norm(flag_new[9] - strategy_vector_list[9]) > epision
                    # np.linalg.norm(flag_new[10] - strategy_vector_list[10]) > epision or \
                    # np.linalg.norm(flag_new[11] - strategy_vector_list[11]) > epision or \
                    # np.linalg.norm(flag_new[12] - strategy_vector_list[12]) > epision or \
                    # np.linalg.norm(flag_new[13] - strategy_vector_list[13]) > epision or \
                    # np.linalg.norm(flag_new[14] - strategy_vector_list[14]) > epision
            ):  # 求二范数
                num = num + 1
                # cs * region->region * cs
                strategy_vector_list = np.array(list(trans(strategy_vector_list)))

                if num > 100:
                    print("没找到均衡！\n")
                    return strategy_vector_list, False

                # print("开始第", num, "轮更新：")
                flag_new = copy.deepcopy(np.array(list(trans(strategy_vector_list))))
                new_region_list = copy.deepcopy(region_list)
                shuffle(new_region_list)
                for agent in new_region_list:  # 区域i对桩
                    dist_agent_to_cs_list = []  # 某个区域i到不同桩的距离 -> [2,4,5], 2是到桩1,4是到桩2,5是到桩3
                    for cs in range(config.cs_num):
                        dist = dist_vector_list[cs]  # [1,2,3,4,5]
                        dist_agent_to_cs_list.append(dist[agent])  # [2]

                    # 求当前区域i的最优策略
                    # zhx重构，原来函数返回的是tuple，现在变成了list，所以要变回去
                    strategy_vector_list[agent] = tuple(self.agent_best_response(agent, region_list,
                                                                           dist_agent_to_cs_list,
                                                                           vehicle_vector,
                                                                           strategy_vector_list, price_list,
                                                                           minimize_res_list))
                    # print("计算agent ", agent, "派到CS的策略：", strategy_vector_list[agent])
                    # print("此时策略向量为：", strategy_vector1, strategy_vector2, "\n")
                # print("更新策略为：", strategy_vector1)
                # new_strategy_vector = np.array(list(trans(strategy_vector_list)))
                # region*cs->cs*region
                strategy_vector_list = np.array(list(trans(strategy_vector_list)))
            else:
                break


        # print("均衡下的策略向量集合为：", trans(strategy_vector_list), "\n")
        return trans(strategy_vector_list), True


evEquilibrium = EvEquilibrium()

if __name__ == "__main__":
    # 初始化操作, 假设目前有公司控制1个充电站, 其他充电站的定价不变
    threshold = 0.000001  # 迭代截止阈值
    # p_1 = 30
    # p_2 = 10
    price = [0, 10, 20]
    p_min = 10
    p_max = 50
    # for cs in range(config.cs_num):
    #     price.append(random.randint(p_min, p_max))
    # price = [30, 10, 5]
    region_num = config.region_num
    minimize_res_list = []
    # dist1, dist2, vehicle_num, region, strategy_vector1, strategy_vector2 = evEquilibrium.initiation(region_num)
    dist, vehicle_num, region, strategy_vector = evEquilibrium.initiation(region_num, config.cs_num)
    print("初始化dist集合, 车辆数， agent, 策略向量集合, 价格集合分别为：", dist, vehicle_num,
          region, strategy_vector, price)

    # 求得此定价下底层的策略
    strategy = evEquilibrium.best_response_simulation(region, dist, vehicle_num,
                                                      price, minimize_res_list, strategy_vector)

    for agent in range(config.region_num):
        print("区域", agent, "的策略：", strategy[agent])

    # # 计算CS1的收益 V_1 = p_1 * f_1
    # for p_1 in range(60):
