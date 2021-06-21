# -*- coding:utf-8 -*-
# @Time: 2021/2/18 12:16
# @Author: xxuanzhu
# @Email: xxuanzhu@seu.edu.cn
# @File: CS_jump.py
# 此文件用于求解价格的跳跃点
import copy
import re as regular
from itertools import chain

import sympy
from pyomo.environ import *
from sympy import *

from config import config
from jump_experiment.EV_Equilibrium_only_equal import EvEquilibrium

evEquilibrium = EvEquilibrium()


def get_revenue(vehicle_num, region_num, changable_cs_number,
                price, strategy):
    # revenue_temp = 0
    f_i = 0
    for item in range(region_num):
        if strategy[item][changable_cs_number] > 0:
            f_i = f_i + vehicle_num[item] * strategy[item][changable_cs_number]

    revenue = price[changable_cs_number] * f_i

    return revenue


def get_max_revenue(dist, region_num, cs_num, priceIndex, price, vehicle_num, flag, results, index):

    N = region_num
    M = cs_num
    dist_vector = dist
    vehicle_vector = vehicle_num
    price_list = price

    model = ConcreteModel()
    model.priceIndex = range(priceIndex)  # 表示有四个需要研究的充电站
    model.region = range(N)
    model.CS = range(M)

    if index == 0:
        print("---------------下面开始求价格区间为：[", 0, ",", results[index], "]的最大收益---------------")
        model.p = Var(model.priceIndex, bounds=(0, results[index]))
    else:
        print("---------------下面开始求价格区间为：[", results[index-1]+1, ",", results[index], "]的最大收益---------------")
        model.p = Var(model.priceIndex, bounds=(results[index - 1]+1, results[index]))
    model.f = Var(model.region, model.CS, bounds=(0.0, 1.0))  # 车流量
    model.lamuda = Var(model.region)  # 等式约束的乘子

    model.obj = Objective(expr=sum(model.p[k] * sum(vehicle_vector[i] * model.f[i, k] for i in model.region)
                                   for k in model.priceIndex), sense=maximize)

    model.single_f = ConstraintList()
    for i in model.region:
        model.single_f.add(sum(model.f[i, j] for j in model.CS) == 1.0)

    # 拉格朗日乘子约束，一共有m * n个
    model.lagrange = ConstraintList()
    for i in model.region:
        for j in model.CS:
            if flag[i][j] == 0:
                model.lagrange.add(model.f[i, j] == 0)
                continue
            if j < 1:
                model.lagrange.add(
                    model.p[j] + sum(model.f[k, j] * vehicle_vector[k] for k in model.region) + dist_vector[j][
                        i] +
                    model.f[i, j] * vehicle_vector[i] - model.lamuda[i] == 0.0)
            else:
                model.lagrange.add(
                    price_list[j] + sum(model.f[k, j] * vehicle_vector[k] for k in model.region) + dist_vector[j][i] +
                    model.f[i, j] * vehicle_vector[i] - model.lamuda[i] == 0.0)

    opt = SolverFactory('ipopt')
    opt.solve(model)
    flow_list = []
    p_list = []
    lamuda_list = []
    revenue = round(model.obj(), 3)
    print('\nDecision Variables')
    for i in model.region:
        temp_flow_list = []
        for j in model.CS:
            temp_flow_list.append(round(model.f[i, j](), 3))
        flow_list.append(temp_flow_list)

    for j in model.priceIndex:
        p_list.append(round(model.p[j](), 3))

    for i in model.region:
        lamuda_list.append(round(model.lamuda[i](), 3))

    print("profits:")
    print(model.obj())

    print("f_ij:")
    for i in model.region:
        print(flow_list[i])
    print("p_j:")
    print(p_list)
    print("lamuda:")
    print(lamuda_list)
    return revenue, flow_list, p_list, lamuda_list


if __name__ == "__main__":
    epision = 0.000001
    p_min = 0
    p_max = 1000
    price = [0, 280]  # 公司只控制1个充电站（0号），其余充电站定价不变
    # region_num = 5
    minimize_res_list = []
    region_num = config.region_num
    cs_num = config.cs_num
    priceIndex = 1
    revenue_list = []
    p_list = []
    index = 0

    dist, vehicle_num, region, strategy_vector = evEquilibrium.initiation(region_num, cs_num)
    print("初始化dist集合, 车辆数， agent, 策略向量集合, 价格集合分别为：", dist, vehicle_num,
          region, strategy_vector, price)


    # 得到了p0=0时，各个区域的策略
    strategy, signal = evEquilibrium.best_response_simulation(region, dist, vehicle_num,
                                                              price, minimize_res_list,
                                                              strategy_vector)

    # 打印策略
    print("当p_0=0时，各个区域的策略为：", strategy, "是否找到均衡：", signal)  # strategy[agent] 区域到充电站的策略 即 f_i_j
    # 记录Qj
    Q_j = [0, 0]
    for cs in range(config.cs_num):
        for item in range(config.region_num):
            Q_j[cs] = round(Q_j[cs] + vehicle_num[item] * strategy[item][cs])
    print("区域到充电站的车辆数：", Q_j)

    print("revenue is ", get_revenue(vehicle_num, config.region_num, 0, price, strategy))

    # 下面开始求跳跃点
    p0 = sympy.symbols("pp")
    f_0_0, f_0_1, f_1_0, f_1_1, lamuda0, lamuda1, Q0, Q1 = sympy.symbols(
        'f_0_0 f_0_1 f_1_0 f_1_1 lamuda0 lamuda1 Q0 Q1')
    Q0 = f_0_0 + f_1_0
    Q1 = f_0_1 + f_1_1

    expr1 = f_0_0 + Q0 + dist[0][0] - lamuda0 + p0
    expr2 = f_0_1 + Q1 + dist[1][0] - lamuda0 + price[1]
    expr3 = f_1_0 + Q0 + dist[0][1] - lamuda1 + p0
    expr4 = f_1_1 + Q1 + dist[1][1] - lamuda1 + price[1]
    expr5 = f_0_0 + f_0_1 - vehicle_num[0]
    expr6 = f_1_0 + f_1_1 - vehicle_num[1]
    if strategy[0][0] <= 0:
        expr1 = f_0_0
    if strategy[0][1] <= 0:
        expr2 = f_0_1
    if strategy[1][0] <= 0:
        expr3 = f_1_0
    if strategy[1][1] <= 0:
        expr4 = f_1_1


    ans = solve([expr1, expr2, expr3, expr4, expr5, expr6], [f_0_0, f_0_1, f_1_0, f_1_1, lamuda0, lamuda1, Q0, Q1])

    f00 = ans[f_0_0]
    f01 = ans[f_0_1]
    f10= ans[f_1_0]
    f11 = ans[f_1_1]
    l0 = ans[lamuda0]
    l1 = ans[lamuda1]
    # 求解何时fij从>0变成==0
    ans_1 = str(solve(ans[f_0_0] > 0))
    ans_2 = str(solve(ans[f_0_1] > 0))
    ans_3 = str(solve(ans[f_1_0] > 0))
    ans_4 = str(solve(ans[f_1_1] > 0))
    # fij从==0变成>0
    expr7 = f01 + f11 + dist[1][0] - l0 + price[1]
    ans_5 = solve(expr7)
    expr8 = f01 + f11 + dist[1][0] - l1 + price[1]
    ans_6 = solve(expr8)
    print("ans is ", ans)
    # anss = [ans_1, ans_2, ans_3, ans_4]
    # #print(zip(fij, anss))
    print(ans_1)
    print(ans_2)
    print(ans_3)
    print(ans_4)
    print(ans_5)
    print(ans_6)
    w1 = regular.findall('-?\d+', ans_1)
    w2 = regular.findall('-?\d+', ans_2)
    w3 = regular.findall('-?\d+', ans_3)
    w4 = regular.findall('-?\d+', ans_4)
    w5 = ans_5
    w6 = ans_6
    r1 = list(map(int, w1))
    r2 = list(map(int, w2))
    r3 = list(map(int, w3))
    r4 = list(map(int, w4))
    r5 = list(map(int, w5))
    r6 = list(map(int, w6))
    p_set = []
    if r1 != [] and r1[0] > 0:
        p_set.append(r1[0])
    if r2 != [] and r2[0] > 0:
        p_set.append(r2[0])
    if r3 != [] and r3[0] > 0:
        p_set.append(r3[0])
    if r4 != [] and r4[0] > 0:
        p_set.append(r4[0])
    if r5 != [] and r5[0] > 0:
        p_set.append(r5[0])
    if r6 != [] and r6[0] > 0:
        p_set.append(r6[0])
    p_set.sort()
    print(p_set)
    result_p = []
    result_p.append(p_set[0])
    flag = [[1, 0],
            [1, 1]]

    revenue, flow_list, ppp, lamuda_list = get_max_revenue(dist, region_num, cs_num, priceIndex, price, vehicle_num,
                                                               flag, result_p, index)
    revenue_list.append(revenue)
    p_list.append(ppp)

    # p0=90，求下一个跳跃点
    expr2 = f_0_1 + Q1 + dist[1][0] - lamuda0 + price[1]

    ans = solve([expr1, expr2, expr3, expr4, expr5, expr6], [f_0_0, f_0_1, f_1_0, f_1_1, lamuda0, lamuda1, Q0, Q1])

    print("ans is ", ans)

    f00 = ans[f_0_0]
    f01 = ans[f_0_1]
    f10 = ans[f_1_0]
    f11 = ans[f_1_1]
    l0 = ans[lamuda0]
    l1 = ans[lamuda1]

    ans_1 = str(solve(ans[f_0_0] > 0))
    ans_2 = str(solve(ans[f_0_1] > 0))
    ans_3 = str(solve(ans[f_1_0] > 0))
    ans_4 = str(solve(ans[f_1_1] > 0))


    print(ans_1)
    print(ans_2)
    print(ans_3)
    print(ans_4)
    # print(ans_5)
    # print(ans_6)

    # expr7 = f01 + f11 + dist[1][0] - l0 + price[1]
    # ans_5 = solve(expr7)
    # expr8 = f01 + f11 + dist[1][0] - l1 + price[1]
    # ans_6 = solve(expr8)
    # print("ans is ", ans)
    # anss = [ans_1, ans_2, ans_3, ans_4]
    # #print(zip(fij, anss))
    # print(ans_1)
    # print(ans_2)
    # print(ans_3)
    # print(ans_4)
    w1 = regular.findall('-?\d+', ans_1)
    w2 = regular.findall('-?\d+', ans_2)
    w3 = regular.findall('-?\d+', ans_3)
    w4 = regular.findall('-?\d+', ans_4)
    # w5 = ans_5
    # w6 = ans_6
    r1 = list(map(int, w1))
    r2 = list(map(int, w2))
    r3 = list(map(int, w3))
    r4 = list(map(int, w4))
    # r5 = list(map(int, w5))
    # r6 = list(map(int, w6))
    p_set = []
    if r1 != [] and r1[0] > 0:
        p_set.append(r1[0])
    if r2 != [] and r2[0] > 0:
        p_set.append(r2[0])
    if r3 != [] and r3[0] > 0:
        p_set.append(r3[0])
    if r4 != [] and r4[0] > 0:
        p_set.append(r4[0])
    # if r5 != [] and r5[0] > 0:
    #     p_set.append(r5[0])
    # if r6 != [] and r6[0] > 0:
    #     p_set.append(r6[0])
    p_set.sort()
    print(p_set)
    result_p.append(p_set[1])


    flag = [[1, 1],
            [1, 1]]

    revenue, flow_list, ppp, lamuda_list = get_max_revenue(dist, region_num, cs_num, priceIndex, price, vehicle_num,
                                                           flag, result_p, index+1)
    revenue_list.append(revenue)
    p_list.append(ppp)


    # 570-710

    expr1 = f_0_0
    ans = solve([expr1, expr2, expr3, expr4, expr5, expr6], [f_0_0, f_0_1, f_1_0, f_1_1, lamuda0, lamuda1, Q0, Q1])

    f00 = ans[f_0_0]
    f01 = ans[f_0_1]
    f10 = ans[f_1_0]
    f11 = ans[f_1_1]
    l0 = ans[lamuda0]
    l1 = ans[lamuda1]

    ans_1 = str(solve(ans[f_0_0] > 0))
    ans_2 = str(solve(ans[f_0_1] > 0))
    ans_3 = str(solve(ans[f_1_0] > 0))
    ans_4 = str(solve(ans[f_1_1] > 0))

    expr7 = f00 + f10 + dist[0][0] - l0 + p0
    ans_5 = solve(expr7)
    expr8 = f00 + f10 + dist[0][0] - l1 + p0
    ans_6 = solve(expr8)
    print("ans is ", ans)
    # anss = [ans_1, ans_2, ans_3, ans_4]
    # #print(zip(fij, anss))
    print(ans_1)
    print(ans_2)
    print(ans_3)
    print(ans_4)
    print(ans_5)
    print(ans_6)
    w1 = regular.findall('-?\d+', ans_1)
    w2 = regular.findall('-?\d+', ans_2)
    w3 = regular.findall('-?\d+', ans_3)
    w4 = regular.findall('-?\d+', ans_4)
    w5 = ans_5
    w6 = ans_6
    r1 = list(map(int, w1))
    r2 = list(map(int, w2))
    r3 = list(map(int, w3))
    r4 = list(map(int, w4))
    r5 = list(map(int, w5))
    r6 = list(map(int, w6))
    p_set = []

    if r1 != [] and r1[0] > 0:
        p_set.append(r1[0])
    if r2 != [] and r2[0] > 0:
        p_set.append(r2[0])
    if r3 != [] and r3[0] > 0:
        p_set.append(r3[0])
    if r4 != [] and r4[0] > 0:
        p_set.append(r4[0])
    if r5 != [] and r5[0] > 0:
        p_set.append(r5[0])
    if r6 != [] and r6[0] > 0:
        p_set.append(r6[0])
    p_set.sort()
    print(p_set)
    result_p.append(p_set[2])
    flag = [[0, 1],
            [1, 1]]

    revenue, flow_list, ppp, lamuda_list = get_max_revenue(dist, region_num, cs_num, priceIndex, price, vehicle_num,
                                                           flag, result_p, index+1+1)
    revenue_list.append(revenue)
    p_list.append(ppp)

    # 670



    print("跳跃点为： ", result_p)
    print("跳跃区间内的最优价格为： ", p_list)
    print("区间内最优价格下的收益为： ", revenue_list)

