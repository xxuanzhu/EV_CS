#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：EV_CS
@File    ：single_xij.py
@Author  ：xxuanZhu
@Date    ：2021/6/9 11:27 
@Purpose : 利用跳跃点方法求解单个CS的最大收益及最优定价
'''



import collections
import copy
import re as regular
import numpy as np
import sympy
from pyomo.environ import *
from sympy import *

from config import config
from EV_Equilibrium_only_equal import EvEquilibrium

evEquilibrium = EvEquilibrium()




def check_if_conflict(expression_dict, lamuda_symbol_list, state_dict, copy_price, p, change_cs, head_fij):
    """
    检查如果多个fij在该点发生变化，是否存在矛盾的地方
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


def get_next_jump_p( dist, vehicle_num, copy_price, expression_dict, fij_symbol_list, q_symbol_list, lamuda_symbol_list, jump_point_list,
                     price_section_index, lower_0_flag, last_lower_0_flag,  last_min_p_queue,
                     last_min_p_fij_queue, state_dict,  p_equal_flag, jp, strategy,  change_cs, p_max, ans_each_jump_step, q_each_jump_step, end_flag):
    """
    获得下一个跳跃点
    :param dist: 区域到充电站的距离
    :param vehicle_num: 每个区域的车辆数
    :param copy_price: 充电站当前的价格
    :param expression_dict: 方程组表示
    :param fij_symbol_list: fij的符号表示, 表示为具体的车辆数量
    :param q_symbol_list: q的符号表示
    :param lamuda_symbol_list: lamuda的符号表示
    :param jump_point_list: 计算出的跳跃点列表
    :param price_section_index: 跳跃点的索引
    :param lower_0_flag: True：fij==0， False：fij>0
    :param last_lower_0_flag: True: 上一轮计算过fij何时从等于0变成大于0，要恢复Q的表示； False：没算过
    :param last_min_p_queue: 上一轮求得的跳跃点价格
    :param last_min_p_fij_queue: 上一轮求得的跳跃点所对应的fij
    :param state_dict: fij的配置，1：fij>0, 2:fij=0
    :param p_equal_flag: True:跳跃点出现振荡，False：没有出现
    :param jp: 价格
    :param strategy: 第一轮所需要确定的fij配置
    :param change_cs: 要求最优p的充电站编号
    :param ans_each_jump_step: 每个区间fij以及lamuda关于p的表示
    :param q_each_jump_step: 每个区间q关于p的表示
    :return:
        result_p：求解出的下一个跳跃点
        last_lower_0_flag：是否进行了fij等于0变成大于0的求解
        last_min_p_queue：上一轮求出的价格
        last_min_p_fij_queue：上一轮变化的fij
        expression_dict：方程组表示
        p_equal_flag：跳跃点振荡
        end_flag：求解是否结束
        start_flag：是否是第一轮
    """

    print("是否出现波动：", p_equal_flag)
    # if last_lower_0_flag:  # 如果上轮有算过小于0的fij什么时候大于0，则要恢复Q的表示，然后计算方程组
    # 恢复Q的关于fij的表示

    for j in range(config.cs_num):
        expr = 0
        for i in range(config.region_num):
            expr += fij_symbol_list[i][j]
        q_symbol_list[j] = expr

    if price_section_index == 1: # 如果是刚开始迭代，通过scipy确定的底层fij确定方程组
        for i in range(config.region_num):
            for j in range(config.cs_num):
                if strategy[i][j] <= 0:
                    expr_f = str(i) + str(j)
                    lower_0_flag = True  # 如果有小于0，置True，后面进行第二种情况的计算
                    expression_dict[expr_f] = fij_symbol_list[i][j]
                    state_dict[fij_symbol_list[i][j]] = 2  # fij=0，将其表示为2
    else:  # 如果不是，则通过上轮解出来的跳跃点情况进行方程组的操作
        copy_last_min_p_fij_queue = last_min_p_fij_queue.copy()
        if len(copy_last_min_p_fij_queue) == 1:  # 发生变化的fij只有1个
            head_fij = copy_last_min_p_fij_queue.popleft() # 拿到上一轮变化的fij
            if p_equal_flag:  # 如果出现了跳跃点波动的情况
                if str(head_fij)[0] == 'f':  # 如果是因为fij从大于0变成等于0引起的波动
                    expr_f = str(head_fij)[1] + str(head_fij)[2]
                    int_i = int(str(head_fij)[1])
                    int_j = int(str(head_fij)[2])
                    expression_dict[expr_f] = fij_symbol_list[int_i][int_j]  # 直接置为0，且不参与从0变成大于0的计算
                    state_dict[fij_symbol_list[int_i][int_j]] = 2
                else:  # 如果是因为fij从等于0变成大于0引起的波动，同样的处理方式
                    int_i = int(head_fij[0])
                    int_j = int(head_fij[1])
                    expr_f = str(int_i) + str(int_j)
                    expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                    state_dict[fij_symbol_list[int_i][int_j]] = 2
            else:  # 没有出现波动，正常处理方程组
                if str(head_fij)[0] == 'f':  # 如果因为fij变化，说明此轮fij会从大于0变成等于0，需要去掉该方程
                    expr_f = str(head_fij)[1] + str(head_fij)[2]
                    int_i = int(str(head_fij)[1])
                    int_j = int(str(head_fij)[2])
                    expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                    state_dict[fij_symbol_list[int_i][int_j]] = 2
                else:  # 如果因为ijx变化，说明此轮fij会从等于0变成大于0，需要加上方程组
                    int_i = int(head_fij[0])
                    int_j = int(head_fij[1])
                    expr_f = str(int_i) + str(int_j)
                    if int(int_j) == change_cs:
                        expression_dict[expr_f] = jp + q_symbol_list[int_j] + dist[int_j][int_i] + fij_symbol_list[int_i][int_j] - lamuda_symbol_list[int_i]

                    else:
                        expression_dict[expr_f] = copy_price[int_j] + q_symbol_list[int_j] + dist[int_j][int_i] + fij_symbol_list[int_i][int_j] - lamuda_symbol_list[int_i]

                    state_dict[fij_symbol_list[int_i][int_j]] = 1
        else:  # 发生变化的fij有多个，先检查是不是同一个ijxf发生变化需要判断是否有存在矛盾的fij
            while copy_last_min_p_fij_queue:
                copy_expression_dict = copy.deepcopy(expression_dict)  # 保存上一轮的方程组
                head_fij = copy_last_min_p_fij_queue.popleft()  # fij
                if p_equal_flag: # 如果出现了跳跃点波动的情况
                    if str(head_fij)[0] == 'f':  # 如果是因为fij从大于0变成等于0引起的波动
                        expr_f = str(head_fij)[1] + str(head_fij)[2]
                        int_i = int(str(head_fij)[1])
                        int_j = int(str(head_fij)[2])
                        expression_dict[expr_f] = fij_symbol_list[int_i][int_j]  # 直接置为0，且不参与从0变成大于0的计算
                        state_dict[fij_symbol_list[int_i][int_j]] = 2
                    else:  # 如果是因为fij从等于0变成大于0引起的波动，同样的处理方式
                        int_i = int(head_fij[0])
                        int_j = int(head_fij[1])
                        expr_f = str(int_i) + str(int_j)
                        expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                        state_dict[fij_symbol_list[int_i][int_j]] = 2
                else:  # 没有出现波动，正常处理方程组
                    if str(head_fij)[0] == 'f':  # # 如果因为fij变化，说明此轮fij会从大于0变成等于0，需要去掉该方程
                        expr_f = str(head_fij)[1] + str(head_fij)[2]
                        int_i = int(str(head_fij)[1])
                        int_j = int(str(head_fij)[2])
                        expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
                        state_dict[fij_symbol_list[int_i][int_j]] = 2
                    else:  # 如果因为ijx变化，说明此轮fij会从等于0变成大于0，需要加上方程组
                        int_i = int(head_fij[0])
                        int_j = int(head_fij[1])
                        expr_f = str(int_i) + str(int_j)
                        if int(int_j) == change_cs:
                            expression_dict[expr_f] = jp + q_symbol_list[int_j] + dist[int_j][int_i] + fij_symbol_list[int_i][int_j] - lamuda_symbol_list[int_i]

                        else:
                            expression_dict[expr_f] = copy_price[int_j] + q_symbol_list[int_j] + dist[int_j][int_i] + fij_symbol_list[int_i][int_j]  - lamuda_symbol_list[int_i]

                        state_dict[fij_symbol_list[int_i][int_j]] = 1

                        # 加入后检查是否存在矛盾
                        if check_if_conflict(expression_dict,  lamuda_symbol_list, state_dict, copy_price, jp, change_cs,head_fij):
                            expression_dict = copy.deepcopy(copy_expression_dict)
                            state_dict[fij_symbol_list[int_i][int_j]] = 2
                        else:
                            print("f", int_i, int_j, "没有矛盾")


    # ans = solve(expression_dict.values())
    # ans = solve(expression_dict.values(),
    #             # (fij_symbol_list[i][j] for j in range(cs_num) for i in range(region_num)), (lamuda_symbol_list[i] for i in range(region_num)),
    #             fij_symbol_list[0][0], fij_symbol_list[0][1], fij_symbol_list[0][2], fij_symbol_list[0][3],
    #             fij_symbol_list[1][0], fij_symbol_list[1][1], fij_symbol_list[1][2], fij_symbol_list[1][3],
    #             fij_symbol_list[2][0], fij_symbol_list[2][1], fij_symbol_list[2][2], fij_symbol_list[2][3],
    #             fij_symbol_list[3][0], fij_symbol_list[3][1], fij_symbol_list[3][2], fij_symbol_list[3][3],
    #             fij_symbol_list[4][0], fij_symbol_list[4][1], fij_symbol_list[4][2], fij_symbol_list[4][3],
    #             fij_symbol_list[5][0], fij_symbol_list[5][1], fij_symbol_list[5][2], fij_symbol_list[5][3],
    #             lamuda_symbol_list[0], lamuda_symbol_list[1], lamuda_symbol_list[2], lamuda_symbol_list[3], lamuda_symbol_list[4], lamuda_symbol_list[5],
    #             jp
    #             )


    # zhx重构，solve的第二个参数是动态传参，可以打包成列表传进去
    arg_list_1 = [fij_symbol_list[i][j] for j in range(cs_num) for i in range(region_num)]
    arg_list_2 = [lamuda_symbol_list[i] for i in range(region_num)]
    arg_list = arg_list_1 + arg_list_2 + [jp]
    ans = solve(expression_dict.values(), arg_list)

    print(ans)
    ans_each_jump_step.append(copy.deepcopy(ans))
    linear_expression_dict = {}  # fij在此区间内小于0
    points_dict = {}  # 跳跃点表示字典 fij:points

    # 提取跳跃点
    for item in ans.items():
        linear_expression_dict[item[0]] = str(solve([item[1] < 0]))
    for item in ans.items():
        if item[0] not in lamuda_symbol_list and item[1] != 0:  # 只提取不为0的fij
            if 'jp' in str(item[1]):
                ans_item0 = str(solve([item[1] < 0]))
                if '/' in ans_item0:
                    points_dict[item[0]] = list(regular.findall(r'(-?\d+/-?\d+)', ans_item0))
                else:
                    points_dict[item[0]] = list(regular.findall(r'(-?\d+)', ans_item0))
    # print(expression_dict)
    # print("在此区间内小于0：", linear_expression_dict)

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
                expr += ans[fij_symbol_list[i][j]]
            q_symbol_list[j] = expr

        if p_equal_flag:  # 如果存在跳跃点振荡的情况，确定是哪个fij
            copy_last_min_p_fij_queue = last_min_p_fij_queue.copy()
            fluctuate_fij = copy_last_min_p_fij_queue.popleft()
            if str(fluctuate_fij)[0] == 'f':  #
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
                if ans[fij_symbol_list[i][j]] == 0:  # xij流量为0,且正好是此fij导致跳跃点振荡的情况，跳过此轮求解
                    if fluctuate_i_j != [] and i == fluctuate_i_j[0] and j == fluctuate_i_j[1]:
                            continue
                    else:
                        fij_name = str(i) + str(j)
                        if j == change_cs:
                            expr = jp + q_symbol_list[j] + dist[j][i] - ans[lamuda_symbol_list[i]]
                        else:
                            expr = copy_price[j] + q_symbol_list[j] + dist[j][i] - ans[lamuda_symbol_list[i]]
                        # for k in range(config.region_num):  # 求解 qj+dij+p = lamudi_i

                        #     fij_name = str(i) + str(j) + str(k)  # ijx(x表示lamubai）
                        #     if j != change_cs:
                                # expr = jp + q_symbol_list[j] + dist[j][i] - ans[lamuda_symbol_list[k]]
                            # else:
                            #     expr =
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


    if lower_0_flag == False:  # 如果是False，将Q表示成关于p的函数
        for j in range(config.cs_num):  # 将qj表示成关于p的形式
            expr = 0
            for i in range(config.region_num):
                # expr += (ans[xij_symbol_list[i][j]] * vehicle_num[i])
                expr += ans[fij_symbol_list[i][j]]
            q_symbol_list[j] = expr

    q_each_jump_step.append(copy.deepcopy(q_symbol_list))
    if check_Q(jump_point_list, price_section_index, change_cs, q_each_jump_step, jp):  # 如果该跳跃点小于0，加入后停止寻找
        end_flag = False


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

    head_p = min_p.popleft()
    head_p_fij = min_P_fij.popleft()
    if '/' in str(head_p):  # 如果确定的跳跃点是分数
        nums_arr_head = regular.findall(r'-?\d+', str(head_p))
        res_head = round(int(nums_arr_head[0]) / int(nums_arr_head[1]), 3)
        if res_head > p_max:
            end_flag = False
            return 0, last_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, expression_dict, p_equal_flag, end_flag
        print("最终加入的值为：", res_head)
        jump_point_list.append(res_head)
    else:  # =如果确定的跳跃点是整数
        int_head = int(head_p)
        if int_head == 0 or int_head > p_max:
            # logging.debug("当前所求跳跃点为整数且值大于p_max或者等于0，返回0，以及结束标志")
            end_flag = False
            return 0, last_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, expression_dict, p_equal_flag, end_flag
        print("最终加入的值为：", int_head)
        jump_point_list.append(int_head)

    copy_price[change_cs] = jump_point_list[price_section_index]  # 把新确定cs的值更新，以确保下一轮的跳跃点首先大于此值
    last_lower_0_flag = lower_0_flag

    return jump_point_list[
               price_section_index], last_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, expression_dict, p_equal_flag, end_flag


def get_max_revenue(dist, region_num, cs_num, priceIndex, vehicle_num, jump_point_list,
                    price_section_index, change_cs, ans_each_jump_step, q_each_jump_step, jp):
    """
    获得每个区间的最大收益，最优定价，最优下的fij，lamuda，以及Q
    :param dist: m*n, 距离数组
    :param region_num: 区域数量
    :param cs_num: 充电站数量
    :param priceIndex: 控制priceIndex个桩
    :param coppy_price:
    :param vehicle_num:
    :param jump_point_list: 求得的跳跃点区间
    :param price_section_index: 跳跃点区间的索引
    :param change_cs: 要求最优p的cs的索引
    :param ans_each_jump_step: 每个区间内fij，lamuda，q关于p的表示
    :param q_each_jump_step: 每个区间内q关于p的表示
    :param jp: 价格p
    :return:
        round(revenue_, 3)：某区间最优收益
        round(p_, 3)：某区间最优收益对应的定价
        fij_：某区间最优收益对应的fij
        lamuda_：某区间最优收益对应的lamuda
        q_：某区间最优收益对应的q
    """
    pleft = jump_point_list[price_section_index-1]  # 区间左端点
    pright = jump_point_list[price_section_index]  # 区间右端点
    print("在区间[", pleft, ",", pright, "]求最大收益：")

    revenue_ = 0
    p_ = 0
    fij_ = [[0 for j in range(cs_num)] for i in range(region_num)]
    lamuda_ = [ 0 for i in range(region_num)]
    q_ = [0 for j in range(cs_num)]

    q_expression = q_each_jump_step[price_section_index - 1][change_cs]
    quadratic_expression = q_each_jump_step[price_section_index-1][change_cs] * jp
    print("最优收益表示：", quadratic_expression)
    fd = diff(quadratic_expression)  # 求出一阶导数
    dRoots = solveset(fd, jp)  # 一阶导数为0的点
    list_dRoots = list(dRoots)
    # print(list_dRoots)
    if len(list_dRoots) != 0 and '/' in str(list_dRoots[0]):
        nums_arr = list(regular.findall(r'-?\d+', str(list_dRoots[0])))  # 提取分数中的数字
        list_value = round(int(nums_arr[0]) / int(nums_arr[1]), 3)
        if list_value > pleft and list_value < pright:  # 如果极值点在区间内
            p_ = list_value
        else:  # 如果极值点不在区间内，带入求两端的值比较大小
            PQ_left = quadratic_expression.subs(jp, pleft)
            PQ_right = quadratic_expression.subs(jp, pright)
            if PQ_left > PQ_right:
                p_ = pleft
            elif PQ_right > PQ_left:
                p_ = pright
        revenue_ = quadratic_expression.subs(jp, p_)
    elif len(list_dRoots) != 0:  # 如果不是分数
        int_value = int(list_dRoots[0])
        if int_value > pleft and int_value < pright:  # 如果极值点在区间内
            p_ = int_value
        else:  # 如果极值点不在区间内，带入求两端的值
            PQ_left = quadratic_expression.subs(jp, pleft)
            PQ_right = quadratic_expression.subs(jp, pright)
            if PQ_left > PQ_right:
                p_ = pleft
            elif PQ_right > PQ_left:
                p_ = pright
        revenue_ = quadratic_expression.subs(jp, p_)
    elif len(list_dRoots) == 0:
        PQ_left = quadratic_expression.subs(jp, pleft)
        PQ_right = quadratic_expression.subs(jp, pright)
        if PQ_left > PQ_right:
            p_ = pleft
        elif PQ_right > PQ_left:
            p_ = pright
    revenue_ = quadratic_expression.subs(jp, p_)

    anss = ans_each_jump_step[price_section_index - 1]
    # if len(list_dRoots) != 0:
    for item in anss.items():
        if 'f' in str(item[0]):
            int_i = int(str(item[0])[1])
            int_j = int(str(item[0])[2])
            if item[1] != 0:
                fij_[int_i][int_j] = item[1].subs(jp, p_)
                if abs(fij_[int_i][int_j]) < 1e-4:
                    fij_[int_i][int_j] = 0
                    # print("出现错误的fij：f", int_i, int_j, "出错的fij关于p的表示：", item[1], "值为：", fij_[int_i][int_j])
                    # print("当前为第", price_section_index, "个跳跃点")
                    # print("所有fij关于p的表示：", anss)
                    # print("当前的价格：", p_)
            else:
                fij_[int_i][int_j] = 0
        elif 'l' in str(item[0]):
            int_i = int(str(item[0])[1])
            lamuda_[int_i] = round(item[1].subs(jp, p_))
    # else:



    # 还要处理q
    q_index = 0
    for item in q_each_jump_step[price_section_index-1]:
        q_[q_index] = round(item.subs(jp, p_))
        q_index += 1

    # Q_about_p = q_each_jump_step[price_section_index-1][change_cs]
    # Q_value = Q_about_p.subs(jp, p_)
    # print("在该区间内最优价格为：", p_, "时，Q为：", Q_value)

    return round(revenue_, 3), round(p_, 3), fij_, lamuda_, q_, q_expression, quadratic_expression



def get_optimal_p(dist, region_num, cs_num, priceIndex, vehicle_num, jump_point_list, change_cs,
                  ans_each_jump_step, q_each_jump_step, jp, copy_price):
    """
    获得pmin-pmax之间求出的跳跃点区间中的最大收益，最优定价，fij，lamuda，q以及所有区间的信息
    :param dist: m*n，距离数组
    :param region_num: 区域数量
    :param cs_num: 充电站数量
    :param priceIndex: 公司控制priceIndex个cs
    :param vehicle_num: 1*n，n个区域的车辆数
    :param jump_point_list: 求出的跳跃点list
    :param change_cs: 求解最优p的充电站索引
    :param ans_each_jump_step: 每个跳跃点区间，fij，lamuda，q关于p的表示
    :param q_each_jump_step: 每个跳跃点区间，q关于p的表示f
    :return:
        optimal_revenue： 整个[p_origin, p_max]内的最大收益
        optimal_p：[p_origin, p_max]中最大收益对应的最优定价
        optimal_fij： 最优定价下的fij配置
        optimal_lamuda ： 最优定价下的lamuda
        optimal_q：最优定价下所有q
        all_revenue_list：[p_origin, p_max]中每个跳跃点区间的最大收益
        all_p_list：[p_origin, p_max]中每个跳跃点区间的最优定价
        all_fij_list：[p_origin, p_max]中每个跳跃点区间的最大收益时的fij配置
        all_lamuda_list：[p_origin, p_max]中每个跳跃点区间的最大收益时的lamuda
        all_q_list：[p_origin, p_max]中每个跳跃点区间最大收益时的q

    """

    optimal_revenue = 0
    optimal_p = 0
    optimal_fij = []
    optimal_lamuda = []
    optimal_q = []

    all_revenue_list = []
    all_p_list = []
    all_fij_list = []
    all_lamuda_list = []
    all_q_list= []
    all_q_expression = []
    all_pq_expression = []

    price_section_index = 1

    while price_section_index < len(jump_point_list):
        revenue_, p_, fij_, lamuda_, q_, q_expression, pq_expression = get_max_revenue(dist, region_num, cs_num,
                                                                                       priceIndex, vehicle_num,
                                                                                       jump_point_list,
                                                                                       price_section_index,
                                                                                       change_cs, ans_each_jump_step,
                                                                                       q_each_jump_step, jp)
        test_price = copy.deepcopy(copy_price)
        test_price[change_cs] = p_
        region = []
        for i in range(region_num):
            region.append(int(i))
        strategy_vector = []
        for i in range(config.region_num):
            s = np.random.rand(config.cs_num)
            strategy_vector.append(s)

        new_strategy, signal = evEquilibrium.best_response_simulation(region, dist, vehicle_num,
                                                                      test_price, minimize_res_list, strategy_vector)
        new_strategy = list(new_strategy)
        for i in range(region_num):
            new_strategy[i] = list(new_strategy[i])
            for j in range(cs_num):
                new_strategy[i][j] = vehicle_num[i] * new_strategy[i][j]

        print("迭代得到的策略为： ", new_strategy)
        print("数学方法求得的策略为：", fij_)



        # 记录求解过程中的值
        all_revenue_list.append(revenue_)
        all_p_list.append(p_)
        all_fij_list.append(fij_)
        all_lamuda_list.append(lamuda_)
        all_q_list.append(q_)
        all_q_expression.append(copy.deepcopy(q_expression))
        all_pq_expression.append(copy.deepcopy(pq_expression))
        price_section_index += 1
        # 取最优的值
        if optimal_revenue < revenue_:
            optimal_revenue = revenue_
            optimal_p = p_
            optimal_fij = fij_
            optimal_lamuda = lamuda_
            optimal_q = q_

    return optimal_revenue, optimal_p, optimal_fij, optimal_lamuda, optimal_q, all_revenue_list, all_p_list, all_fij_list, all_lamuda_list, all_q_list, all_q_expression, all_pq_expression

def check_Q(jump_point_list, price_section_index, change_cs, q_each_jump_step,
            jp):
    '''
    计算某一个区间内，Q_change_cs是否为0了
    :param region_num:
    :param cs_num:
    :param jump_point_list:
    :param price_section_index:
    :param change_cs:
    :param ans_each_jump_step:
    :param q_each_jump_step:
    :param jp:
    :return:
    '''
    # pleft = jump_point_list[price_section_index - 1]  # 区间左端点
    pright = jump_point_list[price_section_index-1]  # 区间右端点


    Q_about_jp = q_each_jump_step[price_section_index - 2][change_cs]  # Q关于p的表示

    Q_value = Q_about_jp.subs(jp, pright)
    if Q_value <= 0:
        return True
    else:
        return False



if __name__ == "__main__":
    change_cs = 0  # 当前求解最优p的充电站索引
    # price = [0 for i in range(config.cs_num)]  # 充电站的定价
    price = [0, 0]
    # list_price = copy.deepcopy(price)
    copy_price = copy.deepcopy(price)
    p_max = 100
    region_num = config.region_num
    cs_num = config.cs_num

    minimize_res_list = []  # 无用
    priceIndex = 1  # 公司控制priceIndex个充电站
    price_section_index = 1  # 价格区间索引
    jump_point_list = []  # 求得的跳跃点集合
    ans_each_jump_step = [] # 每个区间内fij以及lamuda关于p的表示
    q_each_jump_step = []  # 每个区间内q关于p的表示
    # flag_each_jump_step = []  # 保存每个区间的flag

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
    # flag = [[1 for j in range(cs_num)] for i in range(region_num)]  # 利用pyomo求解最优解时对应的拉格朗日约束




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
    fij_symbol_list = [[] for i in range(region_num)]
    for i in range(region_num):
        for j in range(cs_num):
            expr_f = "f"
            expr_f += str(i) + str(j)
            fij = sympy.symbols(expr_f)  # f00, f01, f02...
            fij_symbol_list[i].append(fij)
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
            # expression_dict[expr_n] += (xij_symbol_list[i][j] * vehicle_num[i])  # n0: f00+f01+.. - v[0] = 0
            expression_dict[expr_n] += fij_symbol_list[i][j]  # n0: f00+f01+.. - v[0] = 0

    # Q用fij表示：
    for j in range(cs_num):
        expr = 0
        for i in range(region_num):
            # expr += (xij_symbol_list[i][j] * vehicle_num[i])
            expr += fij_symbol_list[i][j]
        q_symbol_list[j] = expr  # Q0 = f00+f10+f20+...

    # 方程表示：
    """
    {"00": f00+Q0+d00+p-l0 = 0}...
    """
    for i in range(region_num):
        for j in range(cs_num):
            expr_f = str(i) + str(j)
            if j == change_cs:
                # expression_dict[expr_f] = (jp + q_symbol_list[j] + dist[j][i] + xij_symbol_list[i][j] * vehicle_num[i]) * vehicle_num[i] - lamuda_symbol_list[i]
                expression_dict[expr_f] = jp + q_symbol_list[j] + dist[j][i] + fij_symbol_list[i][j] - lamuda_symbol_list[i]
            else:
                # expression_dict[expr_f] = (copy_price[j] + q_symbol_list[j] + dist[j][i] + xij_symbol_list[i][j] * vehicle_num[i]) * vehicle_num[i] - lamuda_symbol_list[i]
                expression_dict[expr_f] = copy_price[j] + q_symbol_list[j] + dist[j][i] + fij_symbol_list[i][j] - lamuda_symbol_list[i]
            # 将状态全部初始化为1，表示fij>0
            state_dict[fij_symbol_list[i][j]] = 1

    # 循环求跳跃点
    result_p = 0
    end_flag = True
    while end_flag:
        result_p, last_lower_0_flag,  last_min_p_queue, last_min_p_fij_queue,  expression_dict, p_equal_flag, end_flag = get_next_jump_p(
            dist, vehicle_num, copy_price, expression_dict, fij_symbol_list, q_symbol_list, lamuda_symbol_list, jump_point_list,
            price_section_index, lower_0_flag, last_lower_0_flag,  last_min_p_queue, last_min_p_fij_queue,
            state_dict,  p_equal_flag, jp, strategy,  change_cs,  p_max,  ans_each_jump_step, q_each_jump_step, end_flag)
        price_section_index += 1





    # 求解每个区间内的最大收益、最优价格p、最大收益下的fij
    optimal_revenue, optimal_p, optimal_fij, optimal_lamuda, optimal_q, all_revenue_list, all_p_list, all_fij_list, all_lamuda_list, all_q_list, all_q_expression_list, all_pq_expression_list = get_optimal_p(
        dist,
        region_num, cs_num, priceIndex,
        vehicle_num,  jump_point_list, change_cs, ans_each_jump_step, q_each_jump_step, jp, copy_price)






    print("跳跃点区间：", jump_point_list)

    print("每个区间的最大收益列表：", all_revenue_list)
    print("整体最大收益：", optimal_revenue)

    print("每个区间的最优价格列表：", all_p_list)
    print("整体最大收益时的最优价格为：", optimal_p)

    print("每个区间的fij配置列表：", all_fij_list)
    print("整体最大收益时的fij配置为：", optimal_fij)

    print("每个区间的lamuda列表：", all_lamuda_list)
    print("整体最大收益时的lamuda为：", optimal_lamuda)

    print("每个区间的q列表：", all_q_list)
    print("整体最大收益时的q为：", optimal_q)

    print("q的表达式：", all_q_expression_list)
    print("pq的表达式：", all_pq_expression_list)
