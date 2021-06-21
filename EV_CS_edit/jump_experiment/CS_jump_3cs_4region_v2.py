# -*- coding:utf-8 -*-
# @Time: 2021/2/18 12:16
# @Author: xxuanzhu
# @Email: xxuanzhu@seu.edu.cn
# @File: CS_jump_3cs_4region.py
# 此文件用于求解价格的跳跃点，已完成功能：在固定其他充电站价格后，可以确定我公司控制充电站的最优定价
import collections
import copy
import re as regular

import sympy
from matplotlib import pyplot as plt
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
        print("---------------下面开始求价格区间为：[", 0, ",", results[index], "]的最优价格和最大收益---------------")
        model.p = Var(model.priceIndex, bounds=(0, results[index]))
    else:
        print("---------------下面开始求价格区间为：[", results[index - 1], ",", results[index], "]的最大收益---------------")
        model.p = Var(model.priceIndex, bounds=(results[index - 1], results[index]))
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
    print('Decision Variables')
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
    print("\n")
    return revenue, flow_list, p_list, lamuda_list


def check_if_conflict(expression_dict, fij_symbol_list, q_symbol_list, lamuda_symbol_list, state_dict, price):
    ans = solve(expression_dict.values())
    # print("\n将其他变量表示为关于p的线性函数：\n", ans)

    linear_expression_dict = {}  # 其他变量关于p的线性函数表示 fij:f(p)
    points_dict = {}  # 跳跃点表示字典 fij:points

    # 提取跳跃点
    for item in ans.items():
        if item[0] not in lamuda_symbol_list:
            linear_expression_dict[item[0]] = str(solve([item[1] < 0, p - price[0] >= 0]))
    for item in linear_expression_dict.items():
        if item[0] not in lamuda_symbol_list and item[1] != "False":  # 只提取不为0的fij
            if '/' in str(item[1]):
                points_dict[item[0]] = list(regular.findall(r'-?\d+/-?\d+', str(linear_expression_dict[item[0]])))
            else:
                # points_dict[item[0]] = list(map(int, regular.findall(r'(-?\d+/-?\d+| -?\d+)', str(linear_expression_dict[item[0]]))))
                points_dict[item[0]] = list(regular.findall(r'-?\d+', str(linear_expression_dict[item[0]])))
    for item in points_dict.items():
        if state_dict[item[0]] == 1:  # 已经是正的，却又出现负的
            if len(item[1]) == 2:
                if item[1][0] < item[1][1]:  # 存在交叉，这个是不正常的
                    return False

    return True


def get_next_jump_p(dist, vehicle_num, price, expression_dict, fij_symbol_list, q_symbol_list, lamuda_symbol_list,
                    revenue_list,
                    result_jump_p, index, lower_0_flag, last_lower_0_flag, great0_to_lower_0_flag,
                    last_great0_to_lower_0_flag, revenue_max_p_list, lower_0_fij_list, last_min_p_queue,
                    last_min_p_fij_queue, state_dict, flag, p_equal_flag):
    print(p_equal_flag)
    # 恢复Q的表示
    if last_lower_0_flag:
        for j in range(config.cs_num):
            expr = 0
            for i in range(config.region_num):
                expr += fij_symbol_list[i][j]
            q_symbol_list[j] = expr

    # p0=0时，依靠迭代产生的strategy进行方程组的确定
    if index == 0:
        for i in range(config.region_num):
            for j in range(config.cs_num):
                if strategy[i][j] <= 0:
                    expr_f = str(i) + str(j)
                    lower_0_flag = True
                    expression_dict[expr_f] = fij_symbol_list[i][j]
                    # fij<0，将其表示为2
                    state_dict[fij_symbol_list[i][j]] = 2
        if check_if_conflict(expression_dict, fij_symbol_list, q_symbol_list, lamuda_symbol_list, state_dict, price):
            print("没有矛盾！！！")
        else:
            print("存在矛盾！！！")
    else:  # 其他时候依靠上一轮的结果确定方程组
        copy_last_min_p_fij_queue = last_min_p_fij_queue.copy()
        if len(copy_last_min_p_fij_queue) == 1:
            head_fij = copy_last_min_p_fij_queue.popleft()
            if p_equal_flag:
                if str(head_fij)[0] == 'f':  # 如果上一轮确定的p是因为fxx变化，说明上一轮求p时fij会从大于0变成小于0，当求下一个跳跃点时需要去掉
                    expr_f = str(head_fij)[1] + str(head_fij)[2]
                    int_i = int(str(head_fij)[1])
                    int_j = int(str(head_fij)[2])
                    expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                    state_dict[fij_symbol_list[int_i][int_j]] = 2
                else:  # 如果上一轮确定的p是因为xxx变化，考虑是否有多个变量变为从小于0变为大于0 说明上一轮求p时fij会从小于0变成大于0，当求下一个跳跃点时要加上
                    int_i = int(head_fij[0])
                    int_j = int(head_fij[1])
                    expr_f = str(int_i) + str(int_j)
                    expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                    state_dict[fij_symbol_list[int_i][int_j]] = 2
            else:
                if str(head_fij)[0] == 'f':  # 如果上一轮确定的p是因为fxx变化，说明上一轮求p时fij会从大于0变成小于0，当求下一个跳跃点时需要去掉
                    expr_f = str(head_fij)[1] + str(head_fij)[2]
                    int_i = int(str(head_fij)[1])
                    int_j = int(str(head_fij)[2])
                    expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                    state_dict[fij_symbol_list[int_i][int_j]] = 2
                else:  # 如果上一轮确定的p是因为xxx变化，考虑是否有多个变量变为从小于0变为大于0 说明上一轮求p时fij会从小于0变成大于0，当求下一个跳跃点时要加上
                    int_i = int(head_fij[0])
                    int_j = int(head_fij[1])
                    expr_f = str(int_i) + str(int_j)
                    if int(int_j) == 0:
                        expression_dict[expr_f] = fij_symbol_list[int_i][int_j] + q_symbol_list[int_j] + dist[int_j][
                            int_i] + p - lamuda_symbol_list[int_i]
                    else:
                        expression_dict[expr_f] = fij_symbol_list[int_i][int_j] + q_symbol_list[int_j] + dist[int_j][
                            int_i] + \
                                                  price[int_j] - lamuda_symbol_list[int_i]
                    state_dict[fij_symbol_list[int_i][int_j]] = 1
        else:
            # 此处应该加一个逻辑判断，有一个会有矛盾
            while copy_last_min_p_fij_queue:
                copy_expression_dict = copy.deepcopy(expression_dict)
                head_fij = copy_last_min_p_fij_queue.popleft()  # 先拿到可能的fij
                if p_equal_flag:
                    if str(head_fij)[0] == 'f':  # 如果上一轮确定的p是因为fxx变化，说明上一轮求p时fij会从大于0变成小于0，当求下一个跳跃点时需要去掉
                        expr_f = str(head_fij)[1] + str(head_fij)[2]
                        int_i = int(str(head_fij)[1])
                        int_j = int(str(head_fij)[2])
                        expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                        state_dict[fij_symbol_list[int_i][int_j]] = 2
                    else:  # 如果上一轮确定的p是因为xxx变化，考虑是否有多个变量变为从小于0变为大于0 说明上一轮求p时fij会从小于0变成大于0，当求下一个跳跃点时要加上
                        int_i = int(head_fij[0])
                        int_j = int(head_fij[1])
                        expr_f = str(int_i) + str(int_j)
                        expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                        state_dict[fij_symbol_list[int_i][int_j]] = 2
                else:
                    if str(head_fij)[0] == 'f':  # 如果上一轮确定的p是因为fxx变化，说明上一轮求p时fij会从大于0变成小于0，当求下一个跳跃点时需要去掉
                        expr_f = str(head_fij)[1] + str(head_fij)[2]
                        int_i = int(str(head_fij)[1])
                        int_j = int(str(head_fij)[2])
                        expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                        state_dict[fij_symbol_list[int_i][int_j]] = 2
                    else:  # 如果上一轮确定的p是因为xxx变化，考虑是否有多个变量变为从小于0变为大于0 说明上一轮求p时fij会从小于0变成大于0，当求下一个跳跃点时要加上
                        int_i = int(head_fij[0])
                        int_j = int(head_fij[1])
                        expr_f = str(int_i) + str(int_j)
                        if int(int_j) == 0:
                            expression_dict[expr_f] = fij_symbol_list[int_i][int_j] + q_symbol_list[int_j] + dist[int_j][
                                int_i] + p - lamuda_symbol_list[int_i]
                        else:
                            expression_dict[expr_f] = fij_symbol_list[int_i][int_j] + q_symbol_list[int_j] + dist[int_j][
                                int_i] + \
                                                      price[int_j] - lamuda_symbol_list[int_i]
                        state_dict[fij_symbol_list[int_i][int_j]] = 1
                    # 加入后检查是否存在矛盾
                    if not check_if_conflict(expression_dict, fij_symbol_list, q_symbol_list, lamuda_symbol_list,
                                             state_dict, price):  # 如果有矛盾
                        print("f", int_i, int_j, "存在矛盾！")
                        # 复原之前的方程组
                        expression_dict = copy.deepcopy(copy_expression_dict)
                        state_dict[fij_symbol_list[int_i][int_j]] = 2
                    else:
                        print("f", int_i, int_j, "没有矛盾！")

    # 求解方程组，将其他变量表示为关于p的线性函数
    ans = solve(expression_dict.values())
    print("\n将其他变量表示为关于p的线性函数：\n", ans)

    linear_expression_dict = {}  # 其他变量关于p的线性函数表示 fij:f(p)
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

    print("\nfij在此区间内小于0：\n", linear_expression_dict)
    print("\n对应的跳跃点：\n", points_dict)

    # 第二种情况：看state_dict有没有==2的，说明有小于0的，需要计算什么时候大于0
    for item in state_dict.items():
        if item[1] == 2:  # 说明存在当前为负的值
            lower_0_flag = True
            break
    ineq_expression_dict = {}
    if lower_0_flag == True and not p_equal_flag:
        for j in range(config.cs_num):
            expr = 0
            for i in range(config.region_num):
                expr += ans[fij_symbol_list[i][j]]
            q_symbol_list[j] = expr

        lower_0_fij_list = []
        for i in range(config.region_num):
            for j in range(config.cs_num):
                if (ans[fij_symbol_list[i][j]] == 0):
                    for k in range(config.region_num):
                        fij_name = str(i) + str(j) + str(k)  # ijx(x表示lamuba）
                        low_0_fij_name = str(i) + "," + str(j)
                        lower_0_fij_list.append(low_0_fij_name)
                        if j == 0:
                            expr = dist[j][i] + q_symbol_list[j] + p - ans[lamuda_symbol_list[k]]
                        else:
                            expr = dist[j][i] + q_symbol_list[j] + price[j] - ans[lamuda_symbol_list[k]]
                        ineq_expression_dict[fij_name] = expr

        for item in ineq_expression_dict.items():
            points_dict[item[0]] = solve(item[1])

    # 开始比较大小，找出最小的p和对应的fij
    min_P_fij = collections.deque()
    min_P_fij.append(0)
    min_p = collections.deque()
    min_p.append(0)
    for item in points_dict.items():
        # 一种情况是['-445']+['6789/4'], 还有一种是[140]
        # 第一种情况
        if str(item[0])[0] == 'f':  # 如果是从大于0变成小于0
            list_item = list(item)
            if list_item[1] != [] and '/' in list_item[1][0]:  # 如果是分数形式
                nums_arr = regular.findall(r'-?\d+', list_item[1][0])  # 提取数字
                item = tuple(list_item)
                if round(int(nums_arr[0]) / int(nums_arr[1]), 6) > price[0]:  # 比较大小
                    head = min_p.popleft()
                    head_fij = min_P_fij.popleft()
                    if '/' in str(head):  # 如果之前的head也是分数形式
                        nums_arr_head = regular.findall(r'-?\d+', str(head))
                        res_head = int(nums_arr_head[0]) / int(nums_arr_head[1])
                        if res_head == 0:  # head==0,将该值放进去
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                        elif res_head == (int(nums_arr[0]) / int(nums_arr[1])):  # 值相同的话，都放进去
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                        elif res_head > (int(nums_arr[0]) / int(nums_arr[1])):  # 小的话，则加入新的fij
                            if min_p:  # 说明不止一个head相同
                                min_p.popleft()
                                min_P_fij.popleft()
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                        elif res_head < (int(nums_arr[0]) / int(nums_arr[1])):  # 大的话，继续放进去head
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                    else:  # 如果之前的head不是分数形式
                        if head == 0:
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                        elif head == (int(nums_arr[0]) / int(nums_arr[1])):
                            min_p.append(head)
                            min_P_fij.append(head_fij)
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                        elif head > (int(nums_arr[0]) / int(nums_arr[1])):
                            if min_p:  # 说明不止一个head相同
                                min_p.popleft()
                                min_P_fij.popleft()
                            min_p.append(item[1][0])
                            min_P_fij.append(item[0])
                        elif head < (int(nums_arr[0]) / int(nums_arr[1])):
                            min_p.append(head)
                            min_P_fij.append(head_fij)

                continue
            elif list_item[1] != []:  # 不是分数行驶，转为int
                list_item[1] = list([int(list_item[1][0])])
                item = tuple(list_item)
        # 剩下int类型的
        if item[1] != []:
            if item[1][0] > price[0]:
                head = min_p.popleft()
                head_fij = min_P_fij.popleft()
                if '/' in str(head):  # 如果最小的head也是分数形式
                    nums_arr_head = regular.findall(r'-?\d+', str(head))
                    res = round(int(nums_arr_head[0]) / int(nums_arr_head[1]), 6)
                    if res == 0:
                        min_p.append(item[1][0])
                        min_P_fij.append(item[0])
                    elif res == item[1][0]:
                        min_p.append(head)
                        min_P_fij.append(head_fij)
                        min_p.append(item[1][0])
                        min_P_fij.append(item[0])
                    elif res > item[1][0]:
                        if min_p:  # 说明不止一个head相同
                            min_p.popleft()
                            min_P_fij.popleft()
                        min_p.append(item[1][0])
                        min_P_fij.append(item[0])
                    elif res < item[1][0]:
                        min_p.append(head)
                        min_P_fij.append(head_fij)
                else:
                    if head == 0:
                        min_p.append(item[1][0])
                        min_P_fij.append(item[0])
                    elif head == item[1][0]:
                        min_p.append(head)
                        min_P_fij.append(head_fij)
                        min_p.append(item[1][0])
                        min_P_fij.append(item[0])
                    elif head > item[1][0]:
                        if min_p:  # 说明不止一个head相同
                            min_p.popleft()
                            min_P_fij.popleft()
                        min_p.append(item[1][0])
                        min_P_fij.append(item[0])
                    elif head < item[1][0]:
                        min_p.append(head)
                        min_P_fij.append(head_fij)

    print(min_p)
    print(min_P_fij)

    # 先检查是否和上个p一样,直接转到最新的情况
    if index != 0:
        now_p = min_p.popleft()
        now_p_fij = min_P_fij.popleft()
        last_p = last_min_p_queue.popleft()
        if '/' in str(now_p) and '/' in str(last_p):
            nums_now_head = regular.findall(r'-?\d+', str(now_p))
            res_now_head = round(int(nums_now_head[0]) / int(nums_now_head[1]), 6)
            nums_last_head = regular.findall(r'-?\d+', str(last_p))
            res_last_head = round(int(nums_last_head[0]) / int(nums_last_head[1]), 6)
            # 如果相等的话
            if abs(res_now_head - res_last_head) < 1e-6:
                p_equal_flag = True
            else:
                p_equal_flag = False
        else:
            if now_p == last_p:
                p_equal_flag = True
            else:
                p_equal_flag = False


        # 这个地方存疑
        min_p.append(now_p)
        min_P_fij.append(now_p_fij)
        last_min_p_queue.append(last_p)

    # 拷贝这轮的跳跃点价格及对应的fij变化
    last_min_p_queue = min_p.copy()
    last_min_p_fij_queue = min_P_fij.copy()

    # 求区间的最大revenue
    # 先指定求解包的方程规则
    for item in state_dict.items():
        if item[1] == 1:
            int_i = int(str(item[0])[1])
            int_j = int(str(item[0])[2])
            flag[int_i][int_j] = 1
        elif item[1] == 2:
            int_i = int(str(item[0])[1])
            int_j = int(str(item[0])[2])
            flag[int_i][int_j] = 0

    print(state_dict)

    # 结束条件判断
    head_p = min_p.popleft()

    # if head_p == 0:
    #     return 0, last_lower_0_flag, last_great0_to_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, flag, expression_dict, p_equal_flag, False
    if '/' in str(head_p):
        nums_arr_head = regular.findall(r'-?\d+', str(head_p))
        res = round(int(nums_arr_head[0]) / int(nums_arr_head[1]), 5)
        if res > 1000:
            return 0, last_lower_0_flag, last_great0_to_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, flag, expression_dict, p_equal_flag, False
        result_jump_p.append(res)
    else:
        if head_p > 1000 or head_p == 0:
            return 0, last_lower_0_flag, last_great0_to_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, flag, expression_dict, p_equal_flag, False
        result_jump_p.append(head_p)

    if not p_equal_flag:
        revenue, flow_list, optimal_p, lamuda_list = get_max_revenue(dist, config.region_num, config.cs_num, priceIndex,
                                                                     price,
                                                                     vehicle_num,
                                                                     flag, result_jump_p, index)

        # 把最优p对应的revenue加入revenue_list
        revenue_list.append(revenue)
        # 把最优p加入revenue_max_p_list
        revenue_max_p_list.append(optimal_p[0])

    price[0] = result_jump_p[index]
    # 记录上一轮是何种情况导致产生的跳跃点
    last_lower_0_flag = lower_0_flag
    last_great0_to_lower_0_flag = great0_to_lower_0_flag

    return result_jump_p[
               index], last_lower_0_flag, last_great0_to_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, flag, expression_dict, p_equal_flag, True


if __name__ == "__main__":
    p_min = 0
    p_max = 1000
    price = [0, 280, 300, 300, 300, 300, 300, 300]
    # price = [0, 358, 246, 125, 280, 345, 125, 245]  # 公司只控制1个充电站（0号），其余充电站定价不变
    minimize_res_list = []

    region_num = config.region_num
    cs_num = config.cs_num
    priceIndex = 1  # 公司控制priceIndex个充电站
    revenue_list = []  # 最优价格list
    index = 0  # 价格区间坐标
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
    state_dict = {}  # 记录fij的状态，0:fij==0, 1:fij>0, 2:fij<0
    flag = [[1 for j in range(config.cs_num)] for i in range(config.region_num)]  # 求解最优解时对应的方程

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
    Q_j = [0 for j in range(config.cs_num)]
    for cs in range(config.cs_num):
        for item in range(config.region_num):
            Q_j[cs] = round(Q_j[cs] + vehicle_num[item] * strategy[item][cs])
    print("区域到充电站的车辆数：", Q_j)

    print("开始求跳跃点：\n")
    p = sympy.symbols("p")  # price
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
    print("f_i_j的symbol表示：\n", fij_symbol_list)

    # q的symbol表示
    q_symbol_list = []
    for j in range(config.cs_num):
        expr_q = "Q"
        expr_q += str(j)
        qj = sympy.symbols(expr_q)
        q_symbol_list.append(qj)
    print("\nQ_j的symbol表示：\n", q_symbol_list)

    # lamuda的symbol表示：
    lamuda_symbol_list = []
    for i in range(config.region_num):
        expr_lamuda = "l"
        expr_lamuda += str(i)
        lamuda = sympy.symbols(expr_lamuda)
        lamuda_symbol_list.append(lamuda)
    print("\nlamuda_i的symbol表示：\n", lamuda_symbol_list)

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
            if j == 0:
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
        result_p, last_lower_0_flag, last_great0_to_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, flag, expression_dict, p_equal_flag, end_flag, = get_next_jump_p(
            dist, vehicle_num, price,
            expression_dict,
            fij_symbol_list,
            q_symbol_list,
            lamuda_symbol_list,
            revenue_list,
            result_jump_p, index,
            lower_0_flag,
            last_lower_0_flag,
            great0_to_lower_0_flag,
            last_great0_to_lower_0_flag,
            revenue_max_p_list,
            lower_0_fij_list,
            last_min_p_queue,
            last_min_p_fij_queue, state_dict, flag, p_equal_flag)
        index += 1

    print(revenue_max_p_list)
    print(revenue_list)
    plt.title("max_p with optimal revenue")
    plt.xlabel("max_p")
    plt.ylabel("revenue")

    plt.plot(revenue_max_p_list, revenue_list, color='red')
    # plt.legend()  # 显示图例
    plt.show()
