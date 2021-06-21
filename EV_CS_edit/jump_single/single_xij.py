#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：EV_CS_edit 
@File    ：single_xij.py
@Author  ：xxuanZhu
@Date    ：2021/6/8 11:02 
@Purpose : 统一求跳跃点时利用xij
'''



import collections
import copy
import re as regular

import sympy
from pyomo.environ import *
from sympy import *

from config import config
from jump_single.EV_Equilibrium_only_equal import EvEquilibrium

evEquilibrium = EvEquilibrium()


def get_max_revenue(dist, region_num, cs_num, priceIndex, price, vehicle_num, flag, jump_point_list,
                    price_section_index,
                    change_cs):
    """
    :param dist: 区域到充电站的距离
    :param region_num: 区域数目
    :param cs_num: 充电站数目
    :param priceIndex: 当前公司控制priceIndex个充电站
    :param price: 充电站当前的价格列表
    :param vehicle_num: 各个区域的车辆总数
    :param flag: 当前fij的配置，即有无流量，有流量为1，无流量为0
    :param jump_point_list: 求得的跳跃点list
    :param price_section_index: 跳跃点区间索引
    :param change_cs: 当前需要求解最优p的充电站索引
    :return:
        revenue: 当前区间最大收益
        flow_list：当前区间最大收益下fij的配置，fij=[0,1]
        p_list: 当前区间最大收益对应的最优价格
        lamuda_list： 当前区间最大收益对应的lamuda乘子
    """
    N = region_num
    M = cs_num
    dist_vector = dist
    vehicle_vector = vehicle_num
    price_list = price

    model = ConcreteModel()
    model.priceIndex = range(priceIndex)  # 表示有priceIndex个需要研究的充电站,即一个公司控制priceIndex个桩
    model.region = range(N)
    model.CS = range(M)
    # logging.info("---------------下面开始求价格区间为：[%s, %s]的最大收益---------------", results[price_section_index - 1],
    #              results[price_section_index], )
    print("---------------下面开始求价格区间为：[", jump_point_list[price_section_index - 1], ",",
          jump_point_list[price_section_index],
          "]的最大收益---------------")
    model.p = Var(bounds=(jump_point_list[price_section_index - 1], jump_point_list[price_section_index]))
    # if price_section_index == 0:
    #     # print("---------------下面开始求价格区间为：[", 0, ",", results[price_section_index], "]的最优价格和最大收益---------------")
    #     model.p = Var(model.priceIndex, bounds=(0, results[price_section_index]))
    # else:
    #     # print("---------------下面开始求价格区间为：[", results[price_section_index - 1], ",", results[price_section_index],
    #     #       "]的最大收益---------------")
    #     model.p = Var(model.priceIndex, bounds=(results[price_section_index - 1], results[price_section_index]))
    model.f = Var(model.region, model.CS, bounds=(0.0, 1.0))  # fij = [0,1]
    model.lamuda = Var(model.region, bounds=(0, None))  # 等式约束的乘子
    # 目标
    model.obj = Objective(expr=model.p * sum(vehicle_vector[i] * model.f[i, change_cs] for i in model.region),
                          sense=maximize)
    # 一些约束
    model.single_f = ConstraintList()
    for i in model.region:
        model.single_f.add(sum(model.f[i, j] for j in model.CS) == 1.0)  # 每个区域派到不同充电站的fij加和为1

    # 拉格朗日乘子约束，一共有m * n个
    model.lagrange = ConstraintList()
    for i in model.region:
        for j in model.CS:
            if flag[i][j] == 0:  # 当前fij流量为0，约束为0
                model.lagrange.add(model.f[i, j] == 0.0)
                continue
            # 如果j是要求最优定价的充电站，p要加入模型变量
            if j == change_cs:
                model.lagrange.add(
                    (model.p + sum(model.f[k, j] * vehicle_vector[k] for k in model.region) + dist_vector[j][
                        i] + model.f[i, j] * vehicle_vector[i]) * vehicle_vector[i] - model.lamuda[i] == 0.0)
            else:  # 如果不是，就加入充电站当前的价格
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

    # for j in model.priceIndex:
    p_list.append(round(model.p(), 3))

    for i in model.region:
        lamuda_list.append(round(model.lamuda[i](), 3))

    # logging.info("profits: %s", model.obj())
    print("profits:")
    print(round(model.obj(), 3))

    # logging.info("f_ij: %s", flow_list)
    # for i in model.region:
    #     logging.info(flow_list[i])
    print("f_ij:")
    for i in model.region:
        print(flow_list[i])
        print("当前区域", i, "的流量之和为：", sum(flow_list[i]))
    # logging.info("p_j: %s", p_list)
    print("p_j:")
    print(p_list)
    # logging.info("lamuda: %s", lamuda_list)
    # print("lamuda:")
    # print(lamuda_list)
    print("\n")
    return revenue, p_list, flow_list, lamuda_list


def get_optimal_p(dist, region_num, cs_num, priceIndex, coppy_price, vehicle_num, flag_each_jump_step, jump_point_list,
                  change_cs):
    """
    :param dist: 距离
    :param region_num: 区域数
    :param cs_num: 充电站数量
    :param priceIndex: 控制cs个数
    :param coppy_price: price的副本
    :param vehicle_num: 车辆数量
    :param flag_each_jump_step: 每个区间flag的记录
    :param jump_point_list: 跳跃点记录
    :param change_cs: 变化的cs编号
    :return:
        optimal_revenue：整体最大收益
        optimal_p：整体最优价格
        optimal_fij：整体最大收益时的fij
        optimal_lamuda：整体最大收益时的lamuda
        all_revenue_list：求解过程中所有revenue
        all_p_list：求解过程中所有p
        all_fij_list：求解过程中所有fij
        all_lamuda_list：求解过程中所有lamuda
    """
    optimal_revenue = 0
    optimal_p = 0
    optimal_fij = []
    optimal_lamuda = []

    all_revenue_list = []
    all_p_list = []
    all_fij_list = []
    all_lamuda_list = []

    price_section_index = 1

    while price_section_index < len(jump_point_list):
        revenue_, p_, fij_, lamuda_= get_max_revenue(dist, region_num, cs_num, priceIndex, coppy_price, vehicle_num, flag_each_jump_step[price_section_index-1], jump_point_list,
                    price_section_index,
                    change_cs)
        # 记录求解过程中的值
        all_revenue_list.append(revenue_)
        all_p_list.append(p_)
        all_fij_list.append(fij_)
        all_lamuda_list.append(lamuda_)
        price_section_index += 1
        # 取最优的值
        if optimal_revenue < revenue_:
            optimal_revenue = revenue_
            optimal_p = p_
            optimal_fij = fij_
            optimal_lamuda = lamuda_

    return optimal_revenue, optimal_p, optimal_fij, optimal_lamuda, all_revenue_list, all_p_list, all_fij_list, all_lamuda_list



def check_if_conflict(expression_dict, lamuda_symbol_list, state_dict, copy_price, p, change_cs, head_fij):
    """
    :param expression_dict: 方程组表示
    :param lamuda_symbol_list: lamuda的符号表示
    :param state_dict: 当前fij的配置，1：大于0， 2: 等于0
    :param price: 充电站的价格列表
    :param p: 最优p
    :param change_cs: 当前求解的充电站索引
    :param head_fij: 上一轮求的跳跃点变化的fij
    :return: True: 存在矛盾， False：不存在矛盾
    """

    ans = solve(expression_dict.values())  # 解出关于p的线性表示
    linear_expression_dict = {}  # fij小于0的表示，且大于当前的price
    points_dict = {}  # 跳跃点表示字典 fij:points

    # 提取跳跃点
    for item in ans.items():
        if item[0] not in lamuda_symbol_list:  # 不提取lamuda的表示
            linear_expression_dict[item[0]] = str(solve([item[1] < 0, p - copy_price[change_cs] > 0]))  # 提取时该区间要先大于当前充电站的价格

    for item in linear_expression_dict.items():
        if item[0] not in lamuda_symbol_list and item[1] != "False":  # 只提取不为空集的fij
            if '/' in str(item[1]):  # 如果是分数
                points_dict[item[0]] = list(regular.findall(r'-?\d+/-?\d+', str(linear_expression_dict[item[0]])))
            else:
                points_dict[item[0]] = list(regular.findall(r'-?\d+', str(linear_expression_dict[item[0]])))

    for item in points_dict.items():
        if str(item[0])[1] == str(head_fij)[0] and str(item[0])[2] == str(head_fij)[1]:  # 看与上个fij相等的情况，才可能会有矛盾出现
            if state_dict[item[0]] == 1:  # fij本已大于0
                if len(item[1]) == 1:  # 非区间
                    if '/' in str(item[1][0]):  # 如果是分数
                        list_item = list(item)  # 将待选值转为list，好进行除法操作
                        nums_arr = regular.findall(r'-?\d+', list_item[1][0])  # 提取分数中的数字
                        item = tuple(list_item)
                        if round(int(nums_arr[0]) / int(nums_arr[1])) == copy_price[change_cs]: # 相等，只可能是小于0的区间在copy_price[]的左边，因此矛盾
                            return True
                    else:  # 如果不是分数
                        if copy_price[change_cs] == int(item[1][0]): # 相等，只可能是小于0的区间在copy_price[]的左边，因此矛盾
                            return True
                if len(item[1]) == 2: # 如果是一个区间
                    if '/' in str(item[1][0]) and '/' in str(item[1][1]):
                        nums_left = regular.findall(r'-?\d+', str(item[1][0]))
                        res_left = round(int(nums_left[0]) / int(nums_left[1]))
                        nums_right = regular.findall(r'-?\d+', str(item[1][1]))
                        res_right = round(int(nums_right[0]) / int(nums_right[1]))
                        # 如果集合相交
                        if res_left < res_right:  # 是p < xx && p > copy_price[]的情况，但p>copy_price[]之后本来就是大于0的，因此矛盾
                            return True
                    elif '/' not in str(item[1][0]) and '/' not in str(item[1][1]):  # 全整数
                        int_now_p = int(item[1][0])
                        int_last_p = int(item[1][1])
                        if int_now_p < int_last_p:  # 是p < xx && p > copy_price[]的情况，但p>copy_price[]之后本来就是大于0的，因此矛盾
                            return True
                    elif '/' in str(item[1][0]) and '/' not in str(item[1][1]):  # 一分数一整数
                        nums_left = regular.findall(r'-?\d+', str(item[1][0]))
                        res_left = round(int(nums_left[0]) / int(nums_left[1]))
                        if res_left < int(item[1][1]):  # 是p < xx && p > copy_price[]的情况，但p>copy_price[]之后本来就是大于0的，因此矛盾
                            return True
                    elif '/' not in str(item[1][0]) and '/' in str(item[1][1]):  # 一分数一整数
                        nums_right = regular.findall(r'-?\d+', str(item[1][1]))
                        res_right = round(int(nums_right[0]) / int(nums_right[1]))
                        if int(item[1][0]) < res_right:  # 是p < xx && p > copy_price[]的情况，但p>copy_price[]之后本来就是大于0的，因此矛盾
                            return True

    return False


def get_next_jump_p( dist, vehicle_num, copy_price, expression_dict, xij_symbol_list, q_symbol_list, lamuda_symbol_list, jump_point_list,
                     price_section_index, lower_0_flag, last_lower_0_flag,  last_min_p_queue,
                     last_min_p_fij_queue, state_dict, flag, p_equal_flag, jp, strategy,  change_cs, p_max, flag_each_jump_step):
    """
    :param dist: 区域到充电站的距离
    :param vehicle_num: 每个区域的车辆数
    :param copy_price: 充电站当前的价格
    :param expression_dict: 方程组表示
    :param xij_symbol_list: xij的符号表示,xij=[0,1]
    :param q_symbol_list: q的符号表示
    :param lamuda_symbol_list: lamuda的符号表示
    :param revenue_list: 每个区间得到的最大收益列表
    :param jump_point_list: 计算出的跳跃点列表
    :param price_section_index: 跳跃点的索引
    :param lower_0_flag: True：fij==0， False：fij>0
    :param last_lower_0_flag: True: 上一轮计算过fij何时从等于0变成大于0，要恢复Q的表示； False：没算过
    :param revenue_max_p_list: 每个区间得到的最大收益所对应的最优价格的列表
    :param lower_0_fij_list: fij等于0的fij列表
    :param last_min_p_queue: 上一轮求得的跳跃点价格
    :param last_min_p_fij_queue: 上一轮求得的跳跃点所对应的fij
    :param state_dict: fij的配置，1：fij>0, 2:fij=0
    :param flag: fij的配置，1：fij>0, 2:fij=0
    :param p_equal_flag: True:跳跃点出现振荡，False：没有出现
    :param p: 价格
    :param strategy: 第一轮所需要确定的fij配置
    :param priceIndex: 公司控制priceIndex个cs
    :param change_cs: 要求最优p的充电站编号
    :param flow_list_list: 所有区间内最大收益对应的的fij配置
    :param p_max: 价格最大值
    :param flag_each_jump_step: 记录每一个flag区间的list
    :return:
        result_p：求解出的下一个跳跃点
        last_lower_0_flag：是否进行了fij等于0变成大于0的求解
        last_min_p_queue：上一轮求出的价格
        last_min_p_fij_queue：上一轮变化的fij
        flag：底层fij约束
        expression_dict：方程组表示
        p_equal_flag：跳跃点振荡
        end_flag：求解是否结束
        start_flag：是否是第一轮
    """


    if last_lower_0_flag:  # 如果上轮有算过小于0的fij什么时候大于0，则要恢复Q的表示，然后计算方程组
        for j in range(config.cs_num):
            expr = 0
            for i in range(config.region_num):
                expr += (xij_symbol_list[i][j] * vehicle_num[i])
            q_symbol_list[j] = expr

    if price_section_index == 1: # 如果是刚开始迭代，通过scipy确定的底层fij确定方程组
        for i in range(config.region_num):
            for j in range(config.cs_num):
                if strategy[i][j] <= 0:
                    expr_f = str(i) + str(j)
                    lower_0_flag = True  # 如果有小于0，置True，后面进行第二种情况的计算
                    expression_dict[expr_f] = xij_symbol_list[i][j]
                    state_dict[xij_symbol_list[i][j]] = 2  # fij=0，将其表示为2
    else:  # 如果不是，则通过上轮解出来的跳跃点情况进行方程组的操作
        copy_last_min_p_fij_queue = last_min_p_fij_queue.copy()
        if len(copy_last_min_p_fij_queue) == 1:  # 发生变化的fij只有1个
            head_fij = copy_last_min_p_fij_queue.popleft() # 拿到上一轮变化的fij
            if p_equal_flag:  # 如果出现了跳跃点波动的情况
                if str(head_fij)[0] == 'x':  # 如果是因为fij从大于0变成等于0引起的波动
                    expr_f = str(head_fij)[1] + str(head_fij)[2]
                    int_i = int(str(head_fij)[1])
                    int_j = int(str(head_fij)[2])
                    expression_dict[expr_f] = xij_symbol_list[int_i][int_j]  # 直接置为0，且不参与从0变成大于0的计算
                    state_dict[xij_symbol_list[int_i][int_j]] = 2
                else:  # 如果是因为fij从等于0变成大于0引起的波动，同样的处理方式
                    int_i = int(head_fij[0])
                    int_j = int(head_fij[1])
                    expr_f = str(int_i) + str(int_j)
                    expression_dict[expr_f] = xij_symbol_list[int_i][int_j]
                    state_dict[xij_symbol_list[int_i][int_j]] = 2
            else:  # 没有出现波动，正常处理方程组
                if str(head_fij)[0] == 'x':  # 如果因为fij变化，说明此轮fij会从大于0变成等于0，需要去掉该方程
                    expr_f = str(head_fij)[1] + str(head_fij)[2]
                    int_i = int(str(head_fij)[1])
                    int_j = int(str(head_fij)[2])
                    expression_dict[expr_f] = xij_symbol_list[int_i][int_j]
                    state_dict[xij_symbol_list[int_i][int_j]] = 2
                else:  # 如果因为ijx变化，说明此轮fij会从等于0变成大于0，需要加上方程组
                    int_i = int(head_fij[0])
                    int_j = int(head_fij[1])
                    expr_f = str(int_i) + str(int_j)
                    if int(int_j) == change_cs:
                        expression_dict[expr_f] = (jp + q_symbol_list[int_j] + dist[int_j][int_i] + xij_symbol_list[int_i][int_j] * vehicle_num[int_i]) * vehicle_num[int_i] - lamuda_symbol_list[int_i]

                    else:
                        expression_dict[expr_f] = (copy_price[int_j] + q_symbol_list[int_j] + dist[int_j][int_i] + xij_symbol_list[int_i][int_j] * vehicle_num[int_i]) * vehicle_num[int_i] - lamuda_symbol_list[int_i]

                    state_dict[xij_symbol_list[int_i][int_j]] = 1
        else:  # 发生变化的fij有多个，需要判断是否有存在矛盾的fij
            while copy_last_min_p_fij_queue:
                copy_expression_dict = copy.deepcopy(expression_dict)  # 保存上一轮的方程组
                head_fij = copy_last_min_p_fij_queue.popleft()  # fij
                if p_equal_flag: # 如果出现了跳跃点波动的情况
                    if str(head_fij)[0] == 'x':  # 如果是因为fij从大于0变成等于0引起的波动
                        expr_f = str(head_fij)[1] + str(head_fij)[2]
                        int_i = int(str(head_fij)[1])
                        int_j = int(str(head_fij)[2])
                        expression_dict[expr_f] = xij_symbol_list[int_i][int_j]  # 直接置为0，且不参与从0变成大于0的计算
                        state_dict[xij_symbol_list[int_i][int_j]] = 2
                    else:  # 如果是因为fij从等于0变成大于0引起的波动，同样的处理方式
                        int_i = int(head_fij[0])
                        int_j = int(head_fij[1])
                        expr_f = str(int_i) + str(int_j)
                        expression_dict[expr_f] = xij_symbol_list[int_i][int_j]
                        state_dict[xij_symbol_list[int_i][int_j]] = 2
                else:  # 没有出现波动，正常处理方程组
                    if str(head_fij)[0] == 'x':  # # 如果因为fij变化，说明此轮fij会从大于0变成等于0，需要去掉该方程
                        expr_f = str(head_fij)[1] + str(head_fij)[2]
                        int_i = int(str(head_fij)[1])
                        int_j = int(str(head_fij)[2])
                        expression_dict[expr_f] = xij_symbol_list[int_i][int_j]
                        state_dict[xij_symbol_list[int_i][int_j]] = 2
                    else:  # 如果因为ijx变化，说明此轮fij会从等于0变成大于0，需要加上方程组
                        int_i = int(head_fij[0])
                        int_j = int(head_fij[1])
                        expr_f = str(int_i) + str(int_j)
                        if int(int_j) == change_cs:
                            expression_dict[expr_f] = (jp + q_symbol_list[int_j] + dist[int_j][int_i] + xij_symbol_list[int_i][int_j] * vehicle_num[int_i]) * vehicle_num[int_i] - lamuda_symbol_list[int_i]

                        else:
                            expression_dict[expr_f] = (copy_price[int_j] + q_symbol_list[int_j] + dist[int_j][int_i] + xij_symbol_list[int_i][int_j] * vehicle_num[int_i]) * vehicle_num[int_i] - lamuda_symbol_list[int_i]

                        state_dict[xij_symbol_list[int_i][int_j]] = 1

                        # 加入后检查是否存在矛盾
                        if check_if_conflict(expression_dict,  lamuda_symbol_list, state_dict, copy_price, jp, change_cs,head_fij):
                            expression_dict = copy.deepcopy(copy_expression_dict)
                            state_dict[xij_symbol_list[int_i][int_j]] = 2
                        else:
                            print("x", int_i, int_j, "没有矛盾")

    for item in state_dict.items():
        if item[1] == 1:
            int_i = int(str(item[0])[1])
            int_j = int(str(item[0])[2])
            flag[int_i][int_j] = 1
        elif item[1] == 2:
            int_i = int(str(item[0])[1])
            int_j = int(str(item[0])[2])
            flag[int_i][int_j] = 0
    flag_each_jump_step.append(copy.deepcopy(flag))

    # ans = solve(expression_dict.values(), (xij_symbol_list[0][0], xij_symbol_list[0][1], lamuda_symbol_list))
    ans = solve(expression_dict.values(),
        xij_symbol_list[0][0], xij_symbol_list[0][1], xij_symbol_list[0][2], xij_symbol_list[0][3],xij_symbol_list[0][4],xij_symbol_list[0][5],
        #         xij_symbol_list[0][3],xij_symbol_list[0][4],
        # xij_symbol_list[0][5], xij_symbol_list[0][6], xij_symbol_list[0][7],
                xij_symbol_list[1][0], xij_symbol_list[1][1], xij_symbol_list[1][2], xij_symbol_list[1][3],xij_symbol_list[1][4], xij_symbol_list[1][5],
                xij_symbol_list[2][0], xij_symbol_list[2][1], xij_symbol_list[2][2], xij_symbol_list[2][3],xij_symbol_list[2][4], xij_symbol_list[2][5],
                # xij_symbol_list[1][3], xij_symbol_list[1][4], xij_symbol_list[1][5], xij_symbol_list[1][6], xij_symbol_list[1][7],
                # xij_symbol_list[2][0], xij_symbol_list[2][1], xij_symbol_list[2][2],
        # xij_symbol_list[2][3],xij_symbol_list[2][4], xij_symbol_list[2][5], xij_symbol_list[2][6], xij_symbol_list[2][7],
        xij_symbol_list[3][0], xij_symbol_list[3][1], xij_symbol_list[3][2],
                xij_symbol_list[3][3], xij_symbol_list[3][4], xij_symbol_list[3][5],
        # xij_symbol_list[3][6], xij_symbol_list[3][7], xij_symbol_list[4][0], xij_symbol_list[4][1],
        # xij_symbol_list[4][2],
        # xij_symbol_list[4][3], xij_symbol_list[4][4], xij_symbol_list[4][5], xij_symbol_list[4][6],
        # xij_symbol_list[4][7],
        lamuda_symbol_list[0], lamuda_symbol_list[1], lamuda_symbol_list[2], lamuda_symbol_list[3],
                # lamuda_symbol_list[2], lamuda_symbol_list[3], lamuda_symbol_list[4],
                jp)

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



    for item in state_dict.items():
        if item[1] == 2:  # 说明存在当前为负的值
            lower_0_flag = True
            break
    ineq_expression_dict = {}
    fluctuate_i_j = []
    lower_0_fij_list = []

    if lower_0_flag == True:  # 如果存在fij=0，需要计算什么时候fij从=0开始有流量
        for j in range(config.cs_num):  # 将qj表示成关于p的形式
            expr = 0
            for i in range(config.region_num):
                expr += (ans[xij_symbol_list[i][j]] * vehicle_num[i])
            q_symbol_list[j] = expr

        if p_equal_flag:  # 如果存在跳跃点振荡的情况，确定是哪个fij
            copy_last_min_p_fij_queue = last_min_p_fij_queue.copy()
            fluctuate_fij = copy_last_min_p_fij_queue.popleft()
            if str(fluctuate_fij)[0] == 'x':  #
                int_i = int(str(fluctuate_fij)[1])
                int_j = int(str(fluctuate_fij)[2])
                fluctuate_i_j.append(int_i)
                fluctuate_i_j.append(int_j)
            else:
                int_i = int(fluctuate_fij[0])
                int_j = int(fluctuate_fij[1])
                fluctuate_i_j.append(int_i)
                fluctuate_i_j.append(int_j)

        # 求解fij=0何时开始有流量
        for i in range(config.region_num):
            for j in range(config.cs_num):
                if ans[xij_symbol_list[i][j]] == 0:  # xij流量为0,且正好是此fij导致跳跃点振荡的情况，跳过此轮求解
                    if fluctuate_i_j != [] and i == fluctuate_i_j[0] and j == fluctuate_i_j[1]:
                            continue
                    else:
                        for k in range(config.region_num):  # 求解 qj+dij+p = lamudi_i
                            fij_name = str(i) + str(j) + str(k)  # ijx(x表示lamubai）
                            if j == change_cs:
                                expr = vehicle_num[i] * (jp + q_symbol_list[j] + dist[j][i]) - ans[lamuda_symbol_list[k]]
                            else:
                                expr = vehicle_num[i] * (copy_price[j] + q_symbol_list[j] + dist[j][i]) - ans[lamuda_symbol_list[k]]
                            ineq_expression_dict[fij_name] = expr

        # 把fij何时等于0的点解出来
        for item in ineq_expression_dict.items():
            if 'jp' in str(item[1]):  # 这个地方说明是有些fij规定为某个值，不随p变化
                points_dict[item[0]] = list([str(solve(item[1])[0])])

    # 开始比较大小
    min_P_fij = collections.deque()
    min_P_fij.append(0)
    min_p = collections.deque()
    min_p.append(0)

    for item in points_dict.items():  # 有两种情况，分别是['-445']+['6789/4']
        if item[1][0] != []:  # 首先不能为空
            if '/' in item[1][0]:  # 如果待选值是分数，形式是['6789/4']
                list_item = list(item)  # 将待选值转为list，利于进行除法操作
                nums_arr = regular.findall(r'-?\d+', list_item[1][0])  # 提取分数中的数字
                item = tuple(list_item)  # 再从list转回去
                list_value = round(int(nums_arr[0]) / int(nums_arr[1]), 3)
                if list_value > copy_price[change_cs]:  # 待选值要大于当前的跳跃点
                    head = min_p.popleft()
                    head_fij = min_P_fij.popleft()
                    if '/' in str(head):  # 如果最小的值也是分数形式
                        nums_arr_head = regular.findall(r'-?\d+', str(head))
                        res_head = round(int(nums_arr_head[0]) / int(nums_arr_head[1]), 3)
                        if res_head == list_value:  # 相等将head和待选值都放进去
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                        elif res_head > list_value:
                            while min_p:  # 看看是不是有多个fij在此p发生变化，如果是，则全部pop出来
                                min_p.popleft()
                                min_P_fij.popleft()
                            min_p.append(item[1][0])  # 加入更新的值
                            min_P_fij.append(item[0])
                        elif res_head < list_value:
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                    else:  # 最小值不是分数形式，是['245']这种形式
                        int_head = int(head)  # 先转为int型
                        if int_head == 0:
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                        elif int_head  == list_value:
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                        elif int_head > list_value:
                            while min_p:  # 看看是不是有多个fij在此p发生变化，如果是，则全部pop出来
                                min_p.popleft()
                                min_P_fij.popleft()
                            min_p.append(item[1][0])  # 加入更新的值
                            min_P_fij.append(item[0])
                        elif int_head < list_value:
                            min_p.append(head)
                            min_P_fij.append(head_fij)
            else: # 待选值是整数，['345']形式
                int_item = int(item[1][0])  # 先转为int型
                if int_item > copy_price[change_cs]:  # 首先大于当前跳跃点
                    head = min_p.popleft()
                    head_fij = min_P_fij.popleft()
                    if '/' in str(head):  # 如果最小的head是分数形式
                        nums_arr_head = regular.findall(r'-?\d+', str(head))
                        res_head = round(int(nums_arr_head[0]) / int(nums_arr_head[1]), 3)
                        if res_head == int_item:  # 相等将head和待选值都放进去
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                        elif res_head > int_item:
                            while min_p:  # 看看是不是有多个fij在此p发生变化，如果是，则全部pop出来
                                min_p.popleft()
                                min_P_fij.popleft()
                            min_p.append(item[1][0])  # 加入更新的值
                            min_P_fij.append(item[0])
                        elif res_head < int_item:
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                    else:  # 都是整数
                        int_head = int(head)
                        if int_head == 0:
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                        elif int_head == int_item:
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                        elif int_head > int_item:
                            while min_p:  # 说明不止一个head相同
                                min_p.popleft()
                                min_P_fij.popleft()
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                        elif int_head < int_item:
                            min_p.append(head)
                            min_P_fij.append(head_fij)



    print("跳跃点为：", min_p)
    print("跳跃点发生变化的fij是：", min_P_fij)

    # 先检查是否和上个跳跃点一样, 一样说明发生振荡
    if price_section_index != 1:
        now_p = min_p.popleft()
        now_p_fij = min_P_fij.popleft()
        last_p = last_min_p_queue.popleft()
        last_p_fij = last_min_p_fij_queue.popleft()
        if '/' in str(now_p) and '/' in str(last_p):  # 俩都是分数
            nums_now_head = regular.findall(r'-?\d+', str(now_p))
            res_now_head = round(int(nums_now_head[0]) / int(nums_now_head[1]), 3)
            nums_last_head = regular.findall(r'-?\d+', str(last_p))
            res_last_head = round(int(nums_last_head[0]) / int(nums_last_head[1]), 3)
            if abs(res_now_head - res_last_head) < 1e-6:
                p_equal_flag = True
            else:
                p_equal_flag = False
        elif '/' not in str(now_p) and '/' not in str(last_p):  # 俩都是整数
            int_now_p = int(now_p)
            int_last_p = int(last_p)
            if int_now_p == int_last_p:
                p_equal_flag = True
            else:
                p_equal_flag = False
        elif '/' in str(now_p) and '/' not in str(last_p):
            nums_now_head = regular.findall(r'-?\d+', str(now_p))
            res_now_head = round(int(nums_now_head[0]) / int(nums_now_head[1]), 3)
            int_last_p = int(last_p)
            if res_now_head == int_last_p:
                p_equal_flag = True
            else:
                p_equal_flag = False
        elif '/' not in str(now_p) and '/' in str(last_p):
            nums_last_head = regular.findall(r'-?\d+', str(last_p))
            res_last_head = round(int(nums_last_head[0]) / int(nums_last_head[1]), 3)
            int_now_p = int(now_p)
            if res_last_head == int_now_p:
                p_equal_flag = True
            else:
                p_equal_flag = False

        min_p.append(now_p)
        min_P_fij.append(now_p_fij)
        last_min_p_queue.append(last_p)
        last_min_p_fij_queue.append(last_p_fij)

    # 保存这轮最终找出的跳跃点
    last_min_p_queue = min_p.copy()
    last_min_p_fij_queue = min_P_fij.copy()

    # 确定底层约束
    # for item in state_dict.items():
    #     if item[1] == 1:
    #         int_i = int(str(item[0])[1])
    #         int_j = int(str(item[0])[2])
    #         flag[int_i][int_j] = 1
    #     elif item[1] == 2:
    #         int_i = int(str(item[0])[1])
    #         int_j = int(str(item[0])[2])
    #         flag[int_i][int_j] = 0

    # 将此轮flag放入
    # flag_each_jump_step.append(flag)

    head_p = min_p.popleft()
    head_p_fij = min_P_fij.popleft()
    if '/' in str(head_p):  # 如果确定的跳跃点是分数
        nums_arr_head = regular.findall(r'-?\d+', str(head_p))
        res_head = round(int(nums_arr_head[0]) / int(nums_arr_head[1]), 3)
        if res_head > p_max:  # 定价超过最大价格
            return 0, last_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, flag, expression_dict, p_equal_flag, False
        jump_point_list.append(res_head)
    else:  # =如果确定的跳跃点是整数
        int_head = int(head_p)
        if int_head > p_max or int_head == 0:
            # logging.debug("当前所求跳跃点为整数且值大于p_max或者等于0，返回0，以及结束标志")
            return 0, last_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, flag, expression_dict, p_equal_flag, False
        # logging.debug("当前所求跳跃点为分数且值小于p_max")
        jump_point_list.append(int_head)


    copy_price[change_cs] = jump_point_list[price_section_index]
    last_lower_0_flag = lower_0_flag

    return jump_point_list[price_section_index], last_lower_0_flag,  last_min_p_queue, last_min_p_fij_queue, flag, \
           expression_dict, p_equal_flag, True





if __name__ == "__main__":
    change_cs = 0  # 当前求解最优p的充电站索引
    price = [10 for i in range(config.cs_num)]  # 充电站的定价
    copy_price = copy.deepcopy(price)
    p_max= 1000
    region_num = config.region_num
    cs_num = config.cs_num

    minimize_res_list = []  # 无用
    priceIndex = 1  # 公司控制priceIndex个充电站
    price_section_index = 1  # 价格区间索引
    jump_point_list = []  # 求得的跳跃点集合
    flag_each_jump_step = []  # 保存每个区间的flag

    # revenue_max_p_list = []  # 所有跳跃点区间内最大收益 对应的最优价格列表
    # lower_0_fij_list = []  # fij从等于0变成大于0的fij的集合
    # flow_list_list = []  # 用来接收每个区间内最优p对应的fij情况
    # Q_list = []  # 计算每个最优pj对应的Qj
    # revenue_list = []  # 所有跳跃点区间内最大收益列表


    lower_0_flag = False  # False：没有fij从等于0变成大于0， True：存在fij从等于0变成大于0
    last_lower_0_flag = False  # False：上一轮没有计算过fij何时从等于0变成大于0，True：计算过，q的表示需要恢复
    p_equal_flag = False  # False：没有出现跳跃点振荡的情况， True：出现了
    # great0_to_lower_0_flag = False  # 第一种情况，从大于0变成小于0
    # last_great0_to_lower_0_flag = False  # 上一轮是大于0变小于0的情况

    last_min_p_queue = collections.deque()  # 记录上一轮的跳跃点的价格
    last_min_p_fij_queue = collections.deque()  # 记录上一轮跳跃点变化的fij

    state_dict = {}  # 记录fij的状态, 1:fij>0, 2:fij=0
    flag = [[1 for j in range(cs_num)] for i in range(region_num)]  # 利用pyomo求解最优解时对应的拉格朗日约束




    # 首先进行初始化
    dist, vehicle_num, region, strategy_vector = evEquilibrium.initiation(region_num, cs_num)
    print("初始化dist集合, 车辆数， agent, 策略向量集合, 价格集合分别为：", dist, vehicle_num,
          region, strategy_vector, price)
    # 加入初始值
    jump_point_list.append(price[change_cs])
    # 对于单个充电桩，首先根据scipy库计算下层的均衡，得到当前的fij配置
    strategy, signal = evEquilibrium.best_response_simulation(region, dist, vehicle_num, price, minimize_res_list, strategy_vector)

    print("当前我公司控制的", change_cs, "号充电站的", "fij配置为：", strategy)

    # 接下来开始求跳跃点
    jp = sympy.symbols("jp")
    expression_dict = {}  # 方程组表示

    # fij的symbol表示
    xij_symbol_list = [[] for i in range(region_num)]
    for i in range(region_num):
        for j in range(cs_num):
            expr_f = "x"
            expr_f += str(i) + str(j)
            xij = sympy.symbols(expr_f)  # f00, f01, f02...
            xij_symbol_list[i].append(xij)
    # print("f_i_j的symbol表示：\n", fij_symbol_list)

    # q的symbol表示
    q_symbol_list = []
    for j in range(cs_num):
        expr_q = "Q"
        expr_q += str(j)
        qj = sympy.symbols(expr_q)  # Q1, Q2, ..
        q_symbol_list.append(qj)
    # print("\nQ_j的symbol表示：\n", q_symbol_list)

    # lamuda的symbol表示：
    lamuda_symbol_list = []
    for i in range(region_num):
        expr_lamuda = "l"
        expr_lamuda += str(i)
        lamuda = sympy.symbols(expr_lamuda)  # l1,l2,...
        lamuda_symbol_list.append(lamuda)
    # print("\nlamuda_i的symbol表示：\n", lamuda_symbol_list)

    # ni等式约束
    for i in range(region_num):
        expr_n = "n"
        expr_n += str(i)
        expression_dict[expr_n] = -vehicle_num[i]  # n0,n1...
        for j in range(config.cs_num):
            expression_dict[expr_n] += (xij_symbol_list[i][j] * vehicle_num[i])  # n0: f00+f01+.. - v[0] = 0

    # Q用fij表示：
    for j in range(cs_num):
        expr = 0
        for i in range(region_num):
            expr += (xij_symbol_list[i][j] * vehicle_num[i])
        q_symbol_list[j] = expr  # Q0 = f00+f10+f20+...

    # 方程表示：
    """
    {"00": f00+Q0+d00+p-l0 = 0}...
    """
    for i in range(region_num):
        for j in range(cs_num):
            expr_f = str(i) + str(j)
            if j == change_cs:
                expression_dict[expr_f] = (jp + q_symbol_list[j] + dist[j][i] + xij_symbol_list[i][j] * vehicle_num[i]) * vehicle_num[i] - lamuda_symbol_list[i]
            else:
                expression_dict[expr_f] = (copy_price[j] + q_symbol_list[j] + dist[j][i] + xij_symbol_list[i][j] * vehicle_num[i]) * vehicle_num[i] - lamuda_symbol_list[i]
            # 将状态全部初始化为1，表示fij>0
            state_dict[xij_symbol_list[i][j]] = 1

    # 循环求跳跃点
    result_p = 0
    end_flag = True
    while end_flag:
        result_p, last_lower_0_flag,  last_min_p_queue, last_min_p_fij_queue, flag, expression_dict, p_equal_flag, end_flag = get_next_jump_p(
            dist, vehicle_num, copy_price, expression_dict, xij_symbol_list, q_symbol_list, lamuda_symbol_list, jump_point_list,
            price_section_index, lower_0_flag, last_lower_0_flag,  last_min_p_queue, last_min_p_fij_queue,
            state_dict, flag, p_equal_flag, jp, strategy,  change_cs,  p_max, flag_each_jump_step)
        price_section_index += 1





    # 求解每个区间内的最大收益、最优价格p、最大收益下的fij
    coppy_price = copy.deepcopy(price)
    optimal_revenue, optimal_p, optimal_fij, optimal_lamuda, all_revenue_list, all_p_list, all_fij_list, all_lamuda_list = get_optimal_p(dist,
                                                                            region_num, cs_num, priceIndex, coppy_price,
                                                                            vehicle_num, flag_each_jump_step, jump_point_list, change_cs)


    print("跳跃点区间：", jump_point_list)
    print("每个区间的最大收益列表：", all_revenue_list)
    print("整体最大收益：", optimal_revenue)

    print("每个区间的最优价格列表：", all_p_list)
    print("整体最大收益时的最优价格为：", optimal_p)

    print("每个区间的fij配置列表：", all_fij_list)
    print("整体最大收益时的fij配置为：", optimal_fij)

    print("每个区间的lamuda列表：", all_lamuda_list)
    print("整体最大收益时的lamuda为：", optimal_lamuda)

    print("每个区间的fij约束列表：", flag_each_jump_step)
