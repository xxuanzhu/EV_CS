#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：EV_CS_edit 
@File    ：get_optimal_p.py
@Author  ：xxuanZhu
@Date    ：2021/5/27 16:05
@Purpose : get all cs optimal p with logging.
'''

import collections
import copy
import logging
import re as regular
import time

import numpy as np
import sympy
from pyomo.environ import *
from sympy import *

from config import config
from jump_experiment.EV_Equilibrium_only_equal import EvEquilibrium

logger = logging.getLogger()
logger.setLevel(logging.NOTSET)
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))

logfile = "get_optimal_p.log"
fh = logging.FileHandler(logfile, mode="a", encoding='utf-8')
fh.setLevel(logging.NOTSET)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] -%(funcName)s - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

evEquilibrium = EvEquilibrium()


# 求编号为changable_cs_number的桩的收益
def get_revenue(vehicle_num, region_num, changable_cs_number,
                price, strategy):
    # revenue_temp = 0
    f_i = 0
    for item in range(region_num):
        if strategy[item][changable_cs_number] > 0:
            f_i = f_i + vehicle_num[item] * strategy[item][changable_cs_number]

    revenue = price[changable_cs_number] * f_i

    return revenue


def get_max_revenue(dist, region_num, cs_num, priceIndex, price, vehicle_num, flag, results, price_section_index,
                    change_cs):
    N = region_num
    M = cs_num
    dist_vector = dist
    vehicle_vector = vehicle_num
    price_list = price

    model = ConcreteModel()
    model.priceIndex = range(priceIndex)  # 表示有priceIndex个需要研究的充电站,即一个公司控制priceIndex个桩
    model.region = range(N)
    model.CS = range(M)
    logging.info("---------------下面开始求价格区间为：[%s, %s]的最大收益---------------", results[price_section_index - 1],
                 results[price_section_index], )
    # print("---------------下面开始求价格区间为：[", results[price_section_index - 1], ",", results[price_section_index],
    #           "]的最大收益---------------")
    model.p = Var(model.priceIndex, bounds=(results[price_section_index - 1], results[price_section_index]))
    # if price_section_index == 0:
    #     # print("---------------下面开始求价格区间为：[", 0, ",", results[price_section_index], "]的最优价格和最大收益---------------")
    #     model.p = Var(model.priceIndex, bounds=(0, results[price_section_index]))
    # else:
    #     # print("---------------下面开始求价格区间为：[", results[price_section_index - 1], ",", results[price_section_index],
    #     #       "]的最大收益---------------")
    #     model.p = Var(model.priceIndex, bounds=(results[price_section_index - 1], results[price_section_index]))
    model.f = Var(model.region, model.CS, bounds=(0.0, 1.0))  # 车流量
    model.lamuda = Var(model.region)  # 等式约束的乘子

    model.obj = Objective(expr=sum(
        model.p[k] * sum(vehicle_vector[i] * model.f[i, change_cs] for i in model.region) for k in model.priceIndex),
        sense=maximize)

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
            # if j < 1:
            if j == change_cs:
                model.lagrange.add(
                    (model.p + sum(model.f[k, j] * vehicle_vector[k] for k in model.region) + dist_vector[j][
                        i] + model.f[i, j] * vehicle_vector[i]) * vehicle_vector[i] - model.lamuda[i] == 0.0)
            else:
                model.lagrange.add(
                    (price_list[j] + sum(model.f[k, j] * vehicle_vector[k] for k in model.region) + dist_vector[j][i] +
                     model.f[i, j] * vehicle_vector[i]) * vehicle_vector[i] - model.lamuda[i] == 0.0)

    opt = SolverFactory('ipopt')
    opt.solve(model)
    flow_list = []
    p_list = []
    lamuda_list = []
    revenue = round(model.obj(), 3)
    # print('Decision Variables')
    for i in model.region:
        temp_flow_list = []
        for j in model.CS:
            temp_flow_list.append(round(model.f[i, j](), 3))
        flow_list.append(temp_flow_list)

    for j in model.priceIndex:
        p_list.append(round(model.p[j](), 3))

    for i in model.region:
        lamuda_list.append(round(model.lamuda[i](), 3))

    logging.info("profits: %s", model.obj())
    # print("profits:")
    # print(model.obj())

    logging.info("f_ij: %s", flow_list)
    # for i in model.region:
    #     logging.info(flow_list[i])
    # print("f_ij:")
    # for i in model.region:
    #     print(flow_list[i])
    logging.info("p_j: %s", p_list)
    # print("p_j:")
    # print(p_list)
    logging.info("lamuda: %s", lamuda_list)
    # print("lamuda:")
    # print(lamuda_list)
    # print("\n")
    return revenue, flow_list, p_list, lamuda_list


def check_if_conflict(expression_dict, fij_symbol_list, q_symbol_list, lamuda_symbol_list, state_dict, price, p,
                      change_cs, head_fij):
    logging.info("开始检查是否存在矛盾！")
    ans = solve(expression_dict.values())
    logging.debug("将其他变量表示为关于p的线性函数：%s", ans)
    # print("\n将其他变量表示为关于p的线性函数：\n", ans)

    linear_expression_dict = {}  # 其他变量关于p的线性函数表示 fij:f(p)
    points_dict = {}  # 跳跃点表示字典 fij:points

    # 提取跳跃点
    for item in ans.items():
        if item[0] not in lamuda_symbol_list:
            linear_expression_dict[item[0]] = str(solve([item[1] < 0, p - price[change_cs] > 0]))
    logging.debug("检查矛盾时，fij小于0的区间为：%s", linear_expression_dict)
    for item in linear_expression_dict.items():
        if item[0] not in lamuda_symbol_list and item[1] != "False":  # 只提取不为0的fij
            if '/' in str(item[1]):
                points_dict[item[0]] = list(regular.findall(r'-?\d+/-?\d+', str(linear_expression_dict[item[0]])))
            else:
                # points_dict[item[0]] = list(map(int, regular.findall(r'(-?\d+/-?\d+| -?\d+)', str(linear_expression_dict[item[0]]))))
                points_dict[item[0]] = list(regular.findall(r'-?\d+', str(linear_expression_dict[item[0]])))
    for item in points_dict.items():
        if str(item[0])[1] == str(head_fij)[0] and str(item[0])[2] == str(head_fij)[1]:  # 如果第一种情况从大于0变成小于0，包含此跳跃点
            if state_dict[item[0]] == 1:
                if len(item[1]) == 1:
                    if '/' in str(item[1][0]):  # 如果是分数
                        list_item = list(item)  # 将待选值转为list，好进行除法操作
                        nums_arr = regular.findall(r'-?\d+', list_item[1][0])  # 提取分数中的数字
                        item = tuple(list_item)
                        if round(int(nums_arr[0]) / int(nums_arr[1])) == price[change_cs]:
                            logging.info("出现了矛盾，上一轮是小于0，这轮在该点还是小于0")
                            return False
                    else:  # 如果不是分数
                        if price[change_cs] == int(item[1][0]):
                            logging.info("出现了矛盾，上一轮是小于0，这轮在该点还是小于0")
                            return False
                if len(item[1]) == 2:
                    if '/' in str(item[1][0]) and '/' in str(item[1][1]):
                        nums_left = regular.findall(r'-?\d+', str(item[1][0]))
                        res_left = round(int(nums_left[0]) / int(nums_left[1]))
                        nums_right = regular.findall(r'-?\d+', str(item[1][1]))
                        res_right = round(int(nums_right[0]) / int(nums_right[1]))
                        # 如果集合相交
                        if res_left < res_right:
                            logging.info("出现了矛盾，上一轮是小于0，在此区间内还是小于0，并非从大于0变成了小于0")
                            return False
                    elif '/' not in str(item[1][0]) and '/' not in str(item[1][1]):
                        int_now_p = int(item[1][0])
                        int_last_p = int(item[1][1])
                        if int_now_p < int_last_p:
                            logging.info("出现了矛盾，上一轮是小于0，在此区间内还是小于0，并非从大于0变成了小于0")
                            return False
                    elif '/' in str(item[1][0]) and '/' not in str(item[1][1]):
                        nums_left = regular.findall(r'-?\d+', str(item[1][0]))
                        res_left = round(int(nums_left[0]) / int(nums_left[1]))
                        if res_left < int(item[1][1]):
                            logging.info("出现了矛盾，上一轮是小于0，在此区间内还是小于0，并非从大于0变成了小于0")
                            return False
                    elif '/' not in str(item[1][0]) and '/' in str(item[1][1]):
                        nums_right = regular.findall(r'-?\d+', str(item[1][1]))
                        res_right = round(int(nums_right[0]) / int(nums_right[1]))
                        if int(item[1][0]) < res_right:
                            logging.info("出现了矛盾，上一轮是小于0，在此区间内还是小于0，并非从大于0变成了小于0")
                            return False

    return True


def get_next_jump_p(dist, vehicle_num, price, expression_dict, fij_symbol_list, q_symbol_list, lamuda_symbol_list,
                    revenue_list,
                    result_jump_p, price_section_index, lower_0_flag, last_lower_0_flag, great0_to_lower_0_flag,
                    last_great0_to_lower_0_flag, revenue_max_p_list, lower_0_fij_list, last_min_p_queue,
                    last_min_p_fij_queue, state_dict, flag, p_equal_flag, p, strategy, priceIndex, change_cs,
                    start_flag, flow_list_list, p_max, Q_list):
    # print("是否出现相等的跳跃点：\n", p_equal_flag)
    logging.debug("是否出现相等的跳跃点：%s", p_equal_flag)
    # 如果上轮有算过小于0的fij什么时候大于0，则要恢复Q的表示，然后计算方程组
    if last_lower_0_flag:
        logging.info("上一轮出现了fij小于0的情况，计算了fij何时==0，需要恢复方程组")
        for j in range(config.cs_num):
            expr = 0
            for i in range(config.region_num):
                expr += fij_symbol_list[i][j]
            q_symbol_list[j] = expr

    # p0=0时，依靠迭代产生的strategy进行方程组的确定
    # start_flag, 即是不是算法刚开始运行，如果是，就按照strategy其他时候的fij不是这样确定的！
    if price_section_index == 1:
        logging.info("index为1，刚开始迭代，根据上一轮的strategy确定方程组")
        for i in range(config.region_num):
            for j in range(config.cs_num):
                if strategy[i][j] <= 0:
                    expr_f = str(i) + str(j)
                    lower_0_flag = True
                    expression_dict[expr_f] = fij_symbol_list[i][j]
                    # fij<0，将其表示为2
                    state_dict[fij_symbol_list[i][j]] = 2
        logging.debug("当前的方程组表示为：%s", expression_dict)
    else:
        logging.info("根据上一轮解出的跳跃点情况确定方程组")
        copy_last_min_p_fij_queue = last_min_p_fij_queue.copy()
        if len(copy_last_min_p_fij_queue) == 1:
            head_fij = copy_last_min_p_fij_queue.popleft()
            logging.debug("上一轮解出的方程组只有1个, 是 %s", head_fij)
            if p_equal_flag:
                if str(head_fij)[0] == 'f':
                    logging.info("出现了fij波动的情况！！调整不参与此轮寻找跳跃点")
                    expr_f = str(head_fij)[1] + str(head_fij)[2]
                    int_i = int(str(head_fij)[1])
                    int_j = int(str(head_fij)[2])
                    expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                    state_dict[fij_symbol_list[int_i][int_j]] = 2
                    logging.debug("当前的方程组表示为：%s", expression_dict)
                else:
                    logging.info("出现了fij波动的情况！！调整不参与此轮寻找跳跃点")
                    int_i = int(head_fij[0])
                    int_j = int(head_fij[1])
                    expr_f = str(int_i) + str(int_j)
                    expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                    state_dict[fij_symbol_list[int_i][int_j]] = 2
                    logging.debug("当前的方程组表示为：%s", expression_dict)
            else:
                if str(head_fij)[0] == 'f':  # 如果上一轮确定的p是因为fxx变化，说明上一轮求p时fij会从大于0变成小于0，当求下一个跳跃点时需要去掉
                    logging.debug("没有出现fij波动的情况，是 %s 在变化，是其从大于0变成小于0所导致，从方程组中去除", head_fij)
                    expr_f = str(head_fij)[1] + str(head_fij)[2]
                    int_i = int(str(head_fij)[1])
                    int_j = int(str(head_fij)[2])
                    expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                    state_dict[fij_symbol_list[int_i][int_j]] = 2
                    logging.debug("去除后的方程组表示为：%s", expression_dict)
                else:  # 如果上一轮确定的p是因为xxx变化，考虑是否有多个变量变为从小于0变为大于0 说明上一轮求p时fij会从小于0变成大于0，当求下一个跳跃点时要加上
                    logging.debug("没有出现fij波动的情况，是 %s 在变化，是其从小于0变成大于0所导致，添加到方程组", head_fij)
                    int_i = int(head_fij[0])
                    int_j = int(head_fij[1])
                    expr_f = str(int_i) + str(int_j)
                    if int(int_j) == change_cs:
                        expression_dict[expr_f] = fij_symbol_list[int_i][int_j] + q_symbol_list[int_j] + dist[int_j][
                            int_i] + p - lamuda_symbol_list[int_i]
                    else:
                        expression_dict[expr_f] = fij_symbol_list[int_i][int_j] + q_symbol_list[int_j] + dist[int_j][
                            int_i] + \
                                                  price[int_j] - lamuda_symbol_list[int_i]
                    state_dict[fij_symbol_list[int_i][int_j]] = 1
                    logging.debug("添加后的方程组表示为：%s", expression_dict)
        else:
            # 此处应该加一个逻辑判断，有一个会有矛盾
            logging.info("出现了多个fij变化的情况！")
            while copy_last_min_p_fij_queue:
                copy_expression_dict = copy.deepcopy(expression_dict)
                head_fij = copy_last_min_p_fij_queue.popleft()  # 先拿到可能的fij
                if p_equal_flag:
                    logging.debug("fij出现波动，是 %s 在变化，调整不参与此次寻找跳跃点", head_fij)
                    if str(head_fij)[0] == 'f':  # 如果上一轮确定的p是因为fxx变化，说明上一轮求p时fij会从大于0变成小于0，当求下一个跳跃点时需要去掉
                        expr_f = str(head_fij)[1] + str(head_fij)[2]
                        int_i = int(str(head_fij)[1])
                        int_j = int(str(head_fij)[2])
                        expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                        state_dict[fij_symbol_list[int_i][int_j]] = 2
                        logging.debug("当前的方程组表示为：%s", expression_dict)
                    else:  # 如果上一轮确定的p是因为xxx变化，考虑是否有多个变量变为从小于0变为大于0 说明上一轮求p时fij会从小于0变成大于0，当求下一个跳跃点时要加上
                        int_i = int(head_fij[0])
                        int_j = int(head_fij[1])
                        expr_f = str(int_i) + str(int_j)
                        expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                        state_dict[fij_symbol_list[int_i][int_j]] = 2
                        logging.debug("当前的方程组表示为：%s", expression_dict)
                else:
                    if str(head_fij)[0] == 'f':  # 如果上一轮确定的p是因为fxx变化，说明上一轮求p时fij会从大于0变成小于0，当求下一个跳跃点时需要去掉
                        logging.debug("没有出现fij波动的情况，是 %s 在变化，是其从大于0变成小于0所导致，从方程组中去除", head_fij)
                        expr_f = str(head_fij)[1] + str(head_fij)[2]
                        int_i = int(str(head_fij)[1])
                        int_j = int(str(head_fij)[2])
                        expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                        state_dict[fij_symbol_list[int_i][int_j]] = 2
                        logging.debug("去除后的方程组表示为：%s", expression_dict)
                    else:  # 如果上一轮确定的p是因为xxx变化，考虑是否有多个变量变为从小于0变为大于0 说明上一轮求p时fij会从小于0变成大于0，当求下一个跳跃点时要加上
                        logging.debug("没有出现fij波动的情况，是 %s 在变化，是其从小于0变成大于0所导致，添加到方程组", head_fij)
                        int_i = int(head_fij[0])
                        int_j = int(head_fij[1])
                        expr_f = str(int_i) + str(int_j)
                        if int(int_j) == change_cs:
                            expression_dict[expr_f] = fij_symbol_list[int_i][int_j] + q_symbol_list[int_j] + \
                                                      dist[int_j][
                                                          int_i] + p - lamuda_symbol_list[int_i]
                        else:
                            expression_dict[expr_f] = fij_symbol_list[int_i][int_j] + q_symbol_list[int_j] + \
                                                      dist[int_j][
                                                          int_i] + \
                                                      price[int_j] - lamuda_symbol_list[int_i]
                        state_dict[fij_symbol_list[int_i][int_j]] = 1
                        # 加入后检查是否存在矛盾
                        logging.debug("添加后的方程组表示为：%s", expression_dict)
                        if not check_if_conflict(expression_dict, fij_symbol_list, q_symbol_list, lamuda_symbol_list,
                                                 state_dict, price, p, change_cs, head_fij):  # 如果有矛盾
                            logging.debug("f %s %s存在矛盾！复原之前的方程组", int_i, int_j)
                            # print("f", int_i, int_j, "存在矛盾！")
                            # 复原之前的方程组
                            expression_dict = copy.deepcopy(copy_expression_dict)
                            logging.debug("复原后方程组表示为：%s", expression_dict)
                            state_dict[fij_symbol_list[int_i][int_j]] = 2
                        else:
                            logging.debug("f %s %s没有矛盾", int_i, int_j)
                            logging.debug("最终方程组为：%s", expression_dict)

    # 求解方程组，将其他变量表示为关于p的线性函数
    ans = solve(expression_dict.values())
    logging.debug("将其他变量表示为关于p的线性函数：%s", ans)
    # print("\n将其他变量表示为关于p的线性函数：\n", ans)

    linear_expression_dict = {}  # fij在此区间内小于0
    points_dict = {}  # 跳跃点表示字典 fij:points

    # 提取跳跃点
    for item in ans.items():
        linear_expression_dict[item[0]] = str(solve([item[1] < 0]))
    for item in ans.items():
        if item[0] not in lamuda_symbol_list and item[1] != 0:  # 只提取不为0的fij
            if 'p' in str(item[1]):
                ans_item0 = str(solve([item[1] < 0]))
                if '/' in ans_item0:
                    points_dict[item[0]] = list(regular.findall(r'(-?\d+/-?\d+)', ans_item0))
                else:
                    points_dict[item[0]] = list(regular.findall(r'(-?\d+)', ans_item0))

    logging.debug("fij在此区间内小于0：%s", linear_expression_dict)
    logging.debug("对应的跳跃点：%s", points_dict)

    # print("\nfij在此区间内小于0：\n", linear_expression_dict)
    # print("\n对应的跳跃点：\n", points_dict)

    # 第二种情况：看state_dict有没有==2的，说明有小于0的，需要计算什么时候大于0
    for item in state_dict.items():
        if item[1] == 2:  # 说明存在当前为负的值
            lower_0_flag = True
            break
    ineq_expression_dict = {}
    if lower_0_flag == True:
        logging.info("存在当前为0的fij")
        for j in range(config.cs_num):
            expr = 0
            for i in range(config.region_num):
                expr += ans[fij_symbol_list[i][j]]
            q_symbol_list[j] = expr
        fluctuate_i_j = []
        if p_equal_flag:
            copy_last_min_p_fij_queue = last_min_p_fij_queue.copy()
            fluctuate_fij = copy_last_min_p_fij_queue.popleft()
            if str(fluctuate_fij)[0] == 'f':  # 如果上一轮确定的p是因为fxx变化，说明上一轮求p时fij会从大于0变成小于0，当求下一个跳跃点时需要去掉
                expr_f = str(fluctuate_fij)[1] + str(fluctuate_fij)[2]
                int_i = int(str(fluctuate_fij)[1])
                int_j = int(str(fluctuate_fij)[2])
                fluctuate_i_j.append(int_i)
                fluctuate_i_j.append(int_j)
            else:  # 如果上一轮确定的p是因为xxx变化，考虑是否有多个变量变为从小于0变为大于0 说明上一轮求p时fij会从小于0变成大于0，当求下一个跳跃点时要加上
                int_i = int(fluctuate_fij[0])
                int_j = int(fluctuate_fij[1])
                fluctuate_i_j.append(int_i)
                fluctuate_i_j.append(int_j)
            logging.debug("存在波动的fij，是f %s", fluctuate_i_j)

        lower_0_fij_list = []
        for i in range(config.region_num):
            for j in range(config.cs_num):
                if (ans[fij_symbol_list[i][j]] == 0):
                    if fluctuate_i_j != []:
                        if i == fluctuate_i_j[0] and j == fluctuate_i_j[1]:
                            logging.info("波动的fij: %s 不参与从小于0变大于0的计算", fluctuate_i_j)
                            continue
                    else:
                        logging.info("其他小于0的fij参与从小于0变大于0的计算")
                        for k in range(config.region_num):
                            fij_name = str(i) + str(j) + str(k)  # ijx(x表示lamuba）
                            low_0_fij_name = str(i) + "," + str(j)
                            lower_0_fij_list.append(low_0_fij_name)
                            if j == change_cs:
                                expr = dist[j][i] + q_symbol_list[j] + p - ans[lamuda_symbol_list[k]]
                            else:
                                expr = dist[j][i] + q_symbol_list[j] + price[j] - ans[lamuda_symbol_list[k]]
                            ineq_expression_dict[fij_name] = expr

        logging.debug("从小于0变成大于0时的式子：%s", ineq_expression_dict)

        for item in ineq_expression_dict.items():
            if 'p' in str(item[1]):
                logging.debug("该式子中存在变量p，提取该点")
                points_dict[item[0]] = list([str(solve(item[1])[0])])

    logging.info("找出所有可能的跳跃点，开始比较寻找最小值")
    # 开始比较大小，找出最小的p和对应的fij
    min_P_fij = collections.deque()
    min_P_fij.append(0)
    min_p = collections.deque()
    min_p.append(0)

    for item in points_dict.items():
        # 有两种情况，分别是['-445']+['6789/4']
        if item[1][0] != []:  # 首先不能为空
            if '/' in item[1][0]:  # 如果待选值是分数
                list_item = list(item)  # 将待选值转为list，好进行除法操作
                nums_arr = regular.findall(r'-?\d+', list_item[1][0])  # 提取分数中的数字
                item = tuple(list_item)
                if round(int(nums_arr[0]) / int(nums_arr[1])) > price[change_cs]:  # 首先待选值（分数）要大于当前的price
                    logging.info("待选值 %s 不为空，待选值为分数， 大于当前的price %s", round(int(nums_arr[0]) / int(nums_arr[1])),
                                 price[change_cs])
                    # 拿到当前的最小值
                    head = min_p.popleft()
                    head_fij = min_P_fij.popleft()
                    if '/' in str(head):  # 如果最小的值也是分数形式
                        logging.info("最小值是分数")
                        nums_arr_head = regular.findall(r'-?\d+', str(head))
                        res_head = int(nums_arr_head[0]) / int(nums_arr_head[1])
                        if res_head == 0:  # head==0,将该值放进去
                            logging.info("head为0，将待选值放入")
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                        elif res_head == (int(nums_arr[0]) / int(nums_arr[1])):  # 值相同的话，都放进去
                            logging.info("待选值与最小值相同，都放入队列")
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                        elif res_head > (int(nums_arr[0]) / int(nums_arr[1])):  # 小的话，则加入新的fij
                            logging.info("待选值小于最小值，待选值放入队列")
                            if min_p:  # 说明不止一个head相同
                                min_p.popleft()
                                min_P_fij.popleft()
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                        elif res_head < (int(nums_arr[0]) / int(nums_arr[1])):  # 大的话，继续放进去head
                            logging.info("待选值大于最小值，将head放入队列")
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                    else:  # 如果当前最小的值不是分数
                        logging.info("最小值为整数")
                        int_head = int(head)
                        if int_head == 0:
                            logging.info("最小值head为0")
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                        elif int_head == (int(nums_arr[0]) / int(nums_arr[1])):
                            logging.info("待选值与最小值相同，都放入队列")
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                        elif int_head > (int(nums_arr[0]) / int(nums_arr[1])):
                            logging.info("待选值小于最小值，待选值放入队列")
                            while min_p:  # 说明不止一个head相同
                                min_p.popleft()
                                min_P_fij.popleft()
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                        elif int_head < (int(nums_arr[0]) / int(nums_arr[1])):
                            logging.info("待选值大于最小值，将head放入队列")
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
            else:  # 如果是整数
                int_item = int(item[1][0])
                if int_item > price[change_cs]:
                    logging.info("待选值 %s 不为空，为整数，大于当前的price：%s", int_item, price[change_cs])
                    # 先拿到最小的值
                    head = min_p.popleft()
                    head_fij = min_P_fij.popleft()
                    if '/' in str(head):  # 如果最小的head也是分数形式
                        logging.info("最小值是分数")
                        nums_arr_head = regular.findall(r'-?\d+', str(head))
                        res = round(int(nums_arr_head[0]) / int(nums_arr_head[1]))
                        if res == 0:
                            logging.info("head为0，将待选值放入")
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                        elif res == int_item:
                            logging.info("待选值与最小值相同，都放入队列")
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                        elif res > int_item:
                            logging.info("待选值小于最小值，待选值放入队列")
                            while min_p:  # 说明不止一个head相同,要全部pop出来
                                min_p.popleft()
                                min_P_fij.popleft()
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                        elif res < int_item:
                            logging.info("待选值大于最小值，将head放入队列")
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                    else:  # 如果最小值不是分数形式
                        int_head = int(head)
                        logging.info("最小值是整数")
                        if int_head == 0:
                            logging.info("head为0，将待选值放入")
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                        elif int_head == int_item:
                            logging.info("待选值与最小值相同，都放入队列")
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                        elif int_head > int_item:
                            logging.info("待选值小于最小值，待选值放入队列")
                            while min_p:  # 说明不止一个head相同
                                min_p.popleft()
                                min_P_fij.popleft()
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)
                        elif int_head < int_item:
                            logging.info("待选值大于最小值，将head放入队列")
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            logging.info("当前价格队列为：%s", min_p)
                            logging.info("当前价格对应的fij队列为：%s", min_P_fij)

    logging.info("跳跃点为：%s", min_p)
    logging.info("跳跃点发生变化的fij是：%s", min_P_fij)
    # print("跳跃点为：", min_p)
    # print("跳跃点发生变化的fij是：",min_P_fij)

    # 先检查是否和上个p一样,直接转到最新的情况
    if price_section_index != 1:
        logging.info("检查是否和上一个确定的跳跃点相同")
        now_p = min_p.popleft()
        now_p_fij = min_P_fij.popleft()
        last_p = last_min_p_queue.popleft()
        if '/' in str(now_p) and '/' in str(last_p):
            nums_now_head = regular.findall(r'-?\d+', str(now_p))
            res_now_head = round(int(nums_now_head[0]) / int(nums_now_head[1]))
            nums_last_head = regular.findall(r'-?\d+', str(last_p))
            res_last_head = round(int(nums_last_head[0]) / int(nums_last_head[1]))
            logging.info("都为分数，上个跳跃点为：%s, 本次求得的跳跃点为：%s", res_last_head, res_now_head)
            # 如果相等的话
            if abs(res_now_head - res_last_head) < 1e-6:
                p_equal_flag = True
            else:
                p_equal_flag = False
        elif '/' not in str(now_p) and '/' not in str(last_p):
            int_now_p = int(now_p)
            int_last_p = int(last_p)
            logging.info("都为整数，上个跳跃点为：%s, 本次求得的跳跃点为: %s", int_last_p, int_now_p)
            if int_now_p == int_last_p:
                p_equal_flag = True
            else:
                p_equal_flag = False
        elif '/' in str(now_p) and '/' not in str(last_p):
            nums_now_head = regular.findall(r'-?\d+', str(now_p))
            res_now_head = round(int(nums_now_head[0]) / int(nums_now_head[1]))
            int_last_p = int(last_p)
            logging.info("一个分数一个整数，上个跳跃点为：%s, 本次求得的跳跃点为：%s", int_last_p, res_now_head)
            if res_now_head == int_last_p:
                p_equal_flag = True
            else:
                p_equal_flag = False
        elif '/' not in str(now_p) and '/' in str(last_p):
            nums_last_head = regular.findall(r'-?\d+', str(last_p))
            res_last_head = round(int(nums_last_head[0]) / int(nums_last_head[1]))
            int_now_p = int(now_p)
            logging.info("一个分数一个整数，上个跳跃点为：%s, 本次求得的跳跃点为：%s", res_last_head, int_now_p)
            if res_last_head == int_now_p:
                p_equal_flag = True
            else:
                p_equal_flag = False

        min_p.append(now_p)
        min_P_fij.append(now_p_fij)
        last_min_p_queue.append(last_p)

    # 拷贝这轮的跳跃点价格及对应的fij变化
    logging.info("当前的跳跃点为：%s", min_p)
    logging.info("当前的跳跃点所对应的fij为：%s", min_P_fij)
    last_min_p_queue = min_p.copy()
    last_min_p_fij_queue = min_P_fij.copy()

    # 求区间的最大revenue
    # 先指定求解包的方程规则
    logging.info("当前参与博弈的fij为：%s", state_dict)
    for item in state_dict.items():
        if item[1] == 1:
            int_i = int(str(item[0])[1])
            int_j = int(str(item[0])[2])
            flag[int_i][int_j] = 1
        elif item[1] == 2:
            int_i = int(str(item[0])[1])
            int_j = int(str(item[0])[2])
            flag[int_i][int_j] = 0
    logging.info("pyomo库所利用的拉格朗日算子状态：%s", flag)

    # 结束条件判断
    head_p = min_p.popleft()

    # if head_p == 0:
    #     return 0, last_lower_0_flag, last_great0_to_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, flag, expression_dict, p_equal_flag, False
    if '/' in str(head_p):
        nums_arr_head = regular.findall(r'-?\d+', str(head_p))
        res = round(int(nums_arr_head[0]) / int(nums_arr_head[1]))
        if res > p_max:
            logging.debug("当前所求跳跃点为分数且值大于p_max，返回0，以及结束标志")
            return 0, last_lower_0_flag, last_great0_to_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, flag, expression_dict, p_equal_flag, False, start_flag
        logging.debug("当前所求跳跃点为分数且值小于p_max")
        result_jump_p.append(res)
    else:
        int_head = int(head_p)
        if int_head > p_max or int_head == 0:
            logging.debug("当前所求跳跃点为整数且值大于p_max或者等于0，返回0，以及结束标志")
            return 0, last_lower_0_flag, last_great0_to_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, flag, expression_dict, p_equal_flag, False, start_flag
        logging.debug("当前所求跳跃点为分数且值小于p_max")
        result_jump_p.append(int_head)

    logging.debug("开始计算最优的revenue以及对应的p")
    revenue, flow_list, optimal_p, lamuda_list = get_max_revenue(dist, config.region_num, config.cs_num, priceIndex,
                                                                 price,
                                                                 vehicle_num,
                                                                 flag, result_jump_p, price_section_index,
                                                                 change_cs)

    # 把最优p对应的revenue加入revenue_list
    revenue_list.append(revenue)
    # 把最优p加入revenue_max_p_list
    revenue_max_p_list.append(optimal_p[0])

    flow_list_list.append(flow_list)

    qj = 0
    # 计算该最优价格下的Qj
    for i in range(config.region_num):
        qj += round(vehicle_num[i] * flow_list[i][change_cs])
    Q_list.append(qj)

    logging.info("最大收益为：%s, 最优价格为：%s, 最优价格下的fij配置：%s", revenue, optimal_p[0], flow_list)

    price[change_cs] = result_jump_p[price_section_index]
    # 记录上一轮是何种情况导致产生的跳跃点
    last_lower_0_flag = lower_0_flag
    last_great0_to_lower_0_flag = great0_to_lower_0_flag

    return result_jump_p[
               price_section_index], last_lower_0_flag, last_great0_to_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, flag, expression_dict, p_equal_flag, True, start_flag


def get_optima_p(price, change_cs, region, dist, vehicle_num, minimize_res_list, strategy_vector, lun, start_flag,
                 p_max):
    region_num = config.region_num
    cs_num = config.cs_num
    priceIndex = 1  # 公司控制priceIndex个充电站
    revenue_list = []  # 最优价格list
    price_section_index = 1  # 价格区间坐标
    result_jump_p = []  # 求得的跳跃点集合
    lower_0_flag = False  # 第二种情况，从小于0变成大于0
    last_lower_0_flag = False  # 上一轮是小于0变成大于0的情况
    great0_to_lower_0_flag = False  # 第一种情况，从大于0变成小于0
    last_great0_to_lower_0_flag = False  # 上一轮是大于0变小于0的情况
    p_equal_flag = False
    revenue_max_p_list = []  # revenue最大所对应的p
    lower_0_fij_list = []  # 从小于0变成大于0的fij的集合
    last_min_p_queue = collections.deque()  # 记录上一轮的跳跃点
    last_min_p_fij_queue = collections.deque()  # 记录上一轮跳跃点 变化的fij
    state_dict = {}  # 记录fij的状态, 1:fij>0, 2:fij=0
    flag = [[1 for j in range(config.cs_num)] for i in range(config.region_num)]  # 求解最优解时对应的方程
    flow_list_list = []  # 用来接收每个区间内最优p对应的fij情况
    Q_list = []  # 计算每个最优pj对应的Qj

    result_jump_p.append(price[change_cs])

    # 得到了p0=0时，各个区域的策略
    if start_flag:
        logging.info("初始轮，strategy由迭代得到，用以确定方程组")
        strategy, signal = evEquilibrium.best_response_simulation(region, dist, vehicle_num,
                                                                  price, minimize_res_list,
                                                                  strategy_vector)
        start_flag = False
    else:
        logging.info("非初始轮，strategy由上一个cs确定最优价后的fij配置确定方程组")
        strategy = strategy_vector

    qj = 0
    for i in range(config.region_num):
        qj += round(vehicle_num[i] * strategy[i][change_cs])

    Q_list.append(qj)



    logging.info("当前各个充电站的定价为：%s", price)

    # 打印策略
    # print("当p_index为", change_cs, "时，各个区域的策略为：", strategy, "是否找到均衡：", signal)  # strategy[agent] 区域到充电站的策略 即 f_i_j
    # # 记录Qj
    # Q_j = [0 for j in range(config.cs_num)]
    # for cs in range(config.cs_num):
    #     for item in range(config.region_num):
    #         Q_j[cs] = round(Q_j[cs] + vehicle_num[item] * strategy[item][cs])
    # print("区域到充电站的车辆数：", Q_j)

    # print("开始求跳跃点：\n")
    p = sympy.symbols("p")  # price[cs]
    # 方程组表示
    expression_dict = {}

    # fij的symbol表示
    fij_symbol_list = [[] for i in range(config.region_num)]
    for i in range(config.region_num):
        for j in range(config.cs_num):
            expr_f = "f"
            expr_f += str(i) + str(j)
            fij = sympy.symbols(expr_f)
            fij_symbol_list[i].append(fij)
    # print("f_i_j的symbol表示：\n", fij_symbol_list)

    # q的symbol表示
    q_symbol_list = []
    for j in range(config.cs_num):
        expr_q = "Q"
        expr_q += str(j)
        qj = sympy.symbols(expr_q)
        q_symbol_list.append(qj)
    # print("\nQ_j的symbol表示：\n", q_symbol_list)

    # lamuda的symbol表示：
    lamuda_symbol_list = []
    for i in range(config.region_num):
        expr_lamuda = "l"
        expr_lamuda += str(i)
        lamuda = sympy.symbols(expr_lamuda)
        lamuda_symbol_list.append(lamuda)
    # print("\nlamuda_i的symbol表示：\n", lamuda_symbol_list)

    # ni等式约束
    for i in range(config.region_num):
        expr_n = "n"
        expr_n += str(i)
        expression_dict[expr_n] = -vehicle_num[i]
        for j in range(config.cs_num):
            expression_dict[expr_n] += fij_symbol_list[i][j]

    # Q用fij表示：
    for j in range(config.cs_num):
        expr = 0
        for i in range(config.region_num):
            expr += fij_symbol_list[i][j]
        q_symbol_list[j] = expr

    # 方程表示：
    for i in range(config.region_num):
        for j in range(config.cs_num):
            expr_f = str(i) + str(j)
            if j == change_cs:
                expression_dict[expr_f] = fij_symbol_list[i][j] + q_symbol_list[j] + dist[j][i] + p - \
                                          lamuda_symbol_list[i]
            else:
                expression_dict[expr_f] = fij_symbol_list[i][j] + q_symbol_list[j] + dist[j][i] + price[j] - \
                                          lamuda_symbol_list[i]
            # 将状态全部初始化为1，表示fij>0
            state_dict[fij_symbol_list[i][j]] = 1

    # 循环求跳跃点
    result_p = 0
    end_flag = True
    while end_flag:
        result_p, last_lower_0_flag, last_great0_to_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, flag, expression_dict, p_equal_flag, end_flag, start_flag = get_next_jump_p(
            dist, vehicle_num, price,
            expression_dict,
            fij_symbol_list,
            q_symbol_list,
            lamuda_symbol_list,
            revenue_list,
            result_jump_p, price_section_index,
            lower_0_flag,
            last_lower_0_flag,
            great0_to_lower_0_flag,
            last_great0_to_lower_0_flag,
            revenue_max_p_list,
            lower_0_fij_list,
            last_min_p_queue,
            last_min_p_fij_queue, state_dict, flag, p_equal_flag, p, strategy, priceIndex, change_cs, start_flag,
            flow_list_list, p_max, Q_list)
        price_section_index += 1

    filename = 'pj_and_Qj.txt'
    with open(filename, 'a', encoding="utf-8") as file_object:
        file_object.write("当前第"+ str(lun)+ "轮，求充电站"+ str(change_cs) + "的最优价格下，跳跃点与optimal_p与Qj分别是：\n")
        file_object.write(str(result_jump_p) + "\n")
        file_object.write(str(revenue_max_p_list) + "\n")
        file_object.write(str(Q_list)+"\n")

    if revenue_max_p_list != []:
        logging.debug("当前充电站为 %s时，每个区间最优定价为：%s ", change_cs, revenue_max_p_list)
        logging.debug("当前充电站为 %s时，每个区间最大收益为：%s ", change_cs, revenue_list)
        # print("当前充电站为", change_cs, "时，每个区间最优定价为：", revenue_max_p_list)
        # print("当前充电站为", change_cs, "时，每个区间最大收益为：", revenue_list)
        max_index = revenue_list.index(max(revenue_list))
        logging.debug("当前充电站为 %s时，最优定价为：%s ", change_cs, revenue_max_p_list[max_index])
        logging.debug("当前充电站为 %s时，每个区间最优收益为：%s ", change_cs, max(revenue_list))
        logging.debug("当前充电站为 %s时，最优p下的fij配置为：%s ", change_cs, flow_list_list[max_index])
        # print("当前充电站为", change_cs, "时，最优定价为：", revenue_max_p_list[max_index])
        # print("当前充电站为", change_cs, "时，最优收益为：", max(revenue_list))
        return int(revenue_max_p_list[max_index]), int(max(revenue_list)), flow_list_list[max_index], True, start_flag
    else:
        logging.info("当前充电站为 %s时，最优定价没找到", change_cs)
        return 0, 0, 0, False, start_flag


if __name__ == "__main__":

    minimize_res_list = []
    last_cs_optimal_p_list = [-1 for i in range(config.cs_num)]
    last_cs_optimal_p_list = np.array(last_cs_optimal_p_list)
    last_cs_optimal_revenue_list = []
    revenue_list = [0 for i in range(config.cs_num)]
    optimal_flow_list = []
    price = [0 for i in range(config.cs_num)]
    price = np.array(price)
    start_flag = True
    p_max = 1000

    dist, vehicle_num, region, strategy_vector = evEquilibrium.initiation(config.region_num, config.cs_num)
    logging.info("初始轮，初始化dist集合: %s, 车辆数: %s, agent: %s , 策略向量集合: %s , 价格集合: %s：", dist, vehicle_num,
                 region, strategy_vector, price)
    # print("初始化dist集合, 车辆数， agent, 策略向量集合, 价格集合分别为：", dist, vehicle_num,
    #       region, strategy_vector, price)

    lun = 1
    while np.linalg.norm(last_cs_optimal_p_list - price) > 1e-8:
        last_cs_optimal_p_list = copy.deepcopy(price)
        logging.info("第 %s 轮更新开始：", lun)
        print("第", lun, "轮更新开始：")
        for cs in range(config.cs_num):
            p, r, optimal_flow_list, flag, start_flag = get_optima_p(price, cs, region, dist, vehicle_num,
                                                                     minimize_res_list, strategy_vector, lun,
                                                                     start_flag, p_max)
            if flag:
                price[cs] = p
                revenue_list[cs] = r
                strategy_vector = optimal_flow_list
        price = np.array(price)
        logging.info("第%s轮结束后，各充电站价格为：%s", lun, price)
        lun += 1
        print("更新后各充电站的价格为: ", price)
    logging.info("找到均衡啦！")
    logging.info("价格列表：%s", price)
    logging.info("收益列表：%s", revenue_list)
