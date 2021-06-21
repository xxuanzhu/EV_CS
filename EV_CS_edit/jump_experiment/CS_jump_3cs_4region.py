# -*- coding:utf-8 -*-
# @Time: 2021/2/18 12:16
# @Author: xxuanzhu
# @Email: xxuanzhu@seu.edu.cn
# @File: CS_jump_3cs_4region.py
# 此文件用于求解价格的跳跃点
import collections
import re as regular

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
        print("---------------下面开始求价格区间为：[", 0, ",", results[index], "]的最优价格和最大收益---------------")
        model.p = Var(model.priceIndex, bounds=(0, results[index]))
    else:
        print("---------------下面开始求价格区间为：[", results[index - 1] + 1, ",", results[index], "]的最大收益---------------")
        model.p = Var(model.priceIndex, bounds=(results[index - 1] + 1, results[index]))
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


def get_next_jump_p(dist, vehicle_num, price, expression_dict, fij_symbol_list, q_symbol_list, lamuda_symbol_list,
                    revenue_list,
                    result_jump_p, index, lower_0_flag, last_lower_0_flag, great0_to_lower_0_flag,
                    last_great0_to_lower_0_flag, revenue_max_p_list, lower_0_fij_list, last_min_p_queue,
                    last_min_p_fij_queue):

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
    else:  # 其他时候依靠上一轮的结果确定方程组
        copy_last_min_p_fij_queue = last_min_p_fij_queue.copy()
        # while copy_last_min_p_fij_queue:
        head_fij = copy_last_min_p_fij_queue.popleft()
        if str(head_fij)[0] == 'f':  # 如果上一轮确定的p是因为fxx变化，说明上一轮求p时fij会从大于0变成小于0，当求下一个跳跃点时需要去掉
            expr_f = str(head_fij)[1] + str(head_fij)[2]
            int_i = int(str(head_fij)[1])
            int_j = int(str(head_fij)[2])
            expression_dict[expr_f] = fij_symbol_list[int_i][int_j]
        else:  # 如果上一轮确定的p是因为xxx变化，说明上一轮求p时fij会从小于0变成大于0，当求下一个跳跃点时要加上
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
            points_dict[item[0]] = list(map(int, regular.findall('-?\d+', str(solve([item[1] < 0])))))

    print("\nfij在此区间内小于0：\n", linear_expression_dict)
    print("\n对应的跳跃点：\n", points_dict)

    # 第二种情况：有小于0的fij，看何时从小于0变成大于0
    ineq_expression_dict = {}
    if lower_0_flag == True:
        for j in range(config.cs_num):
            expr = 0
            for i in range(config.region_num):
                expr += ans[fij_symbol_list[i][j]]
            q_symbol_list[j] = expr

        # lower_0_fij_list = []
        for i in range(config.region_num):
            for j in range(config.cs_num):
                if (ans[fij_symbol_list[i][j]] == 0):
                    for k in range(config.region_num):
                        fij_name = str(i) + str(j) + str(k) # ijx(x表示lamuba）
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
        if len(item[1]) == 2:
            item[1][0] = item[1][0] / item[1][1]
        if item[1][0] > price[0]:
            head = min_p.popleft()
            head_fij = min_P_fij.popleft()
            if head == 0:
                min_p.append(item[1][0])
                min_P_fij.append(item[0])
            elif head == item[1][0]:
                min_p.append(head)
                min_P_fij.append(head_fij)
                min_p.append(item[1][0])
                min_P_fij.append(item[0])
            elif head > item[1][0]:
                min_p.append(item[1][0])
                min_P_fij.append(item[0])
            elif head < item[1][0]:
                min_p.append(head)
                min_P_fij.append(head_fij)
    print(min_p)
    print(min_P_fij)

    # 拷贝这轮的跳跃点价格及对应的fij变化
    last_min_p_queue = min_p.copy()
    last_min_p_fij_queue = min_P_fij.copy()

    # 求区间的最大revenue
    # 先指定求解包的方程规则
    copy_last_min_p_fij_queue = last_min_p_fij_queue.copy()
    flag = [[1 for j in range(config.cs_num)] for i in range(config.region_num)]
    if lower_0_flag == True:
        while copy_last_min_p_fij_queue:
            head_fij = copy_last_min_p_fij_queue.popleft()
            int_i = int(head_fij[0])
            int_j = int(head_fij[1])
            flag[int_i][int_j] = 0
    elif last_great0_to_lower_0_flag:
        # 这个地方还有问题，还没想明白
        while copy_last_min_p_fij_queue:
            head_fij = copy_last_min_p_fij_queue.popleft()
            int_i = int(head_fij[1])
            int_j = int(head_fij[2])
            flag[int_i][int_j] = 0

    # 结束条件判断
    head_p = min_p.popleft()
    if head_p == 0:
        return 0, last_lower_0_flag, last_great0_to_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, False
    result_jump_p.append(head_p)
    # 求解该区间内的最优price和revenue
    revenue, flow_list, optimal_p, lamuda_list = get_max_revenue(dist, config.region_num, config.cs_num, priceIndex,
                                                                 price,
                                                                 vehicle_num,
                                                                 flag, result_jump_p, index)


    # 把最优p对应的revenue加入revenue_list
    revenue_list.append(revenue)
    # 把最优p加入revenue_max_p_list
    revenue_max_p_list.append(optimal_p)





    price[0] = result_jump_p[index]
    # 记录上一轮是何种情况导致产生的跳跃点
    last_lower_0_flag = lower_0_flag
    last_great0_to_lower_0_flag = great0_to_lower_0_flag

    return result_jump_p[
               index], last_lower_0_flag, last_great0_to_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, True


if __name__ == "__main__":
    p_min = 0
    p_max = 1000
    price = [0, 300, 300, 300]  # 公司只控制1个充电站（0号），其余充电站定价不变
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
    revenue_max_p_list = []  # revenue最大所对应的p
    lower_0_fij_list = []  # 从小于0变成大于0的fij的集合
    last_min_p_queue = collections.deque()  # 记录上一轮的跳跃点
    last_min_p_fij_queue = collections.deque()  # 记录上一轮跳跃点 变化的fij

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

    # 循环求跳跃点
    result_p = 0
    end_flag = True
    while end_flag:
        result_p, last_lower_0_flag, last_great0_to_lower_0_flag, last_min_p_queue, last_min_p_fij_queue, end_flag = get_next_jump_p(
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
            last_min_p_fij_queue)
        index += 1

    # # fij < 0, 则不加入方程组
    # for i in range(config.region_num):
    #     for j in range(config.cs_num):
    #         expr = str(i) + str(j)
    #         if strategy[i][j] <= 0:
    #             lower_0_flag = True
    #             expression_dict[expr] = fij_symbol_list[i][j]
    # ans = solve(expression_dict.values())
    # print("\n将其他变量表示为关于p的线性函数：\n", ans)
    #
    #
    # linear_expression_dict = {}
    # points_dict = {}
    #
    # for item in ans.items():
    #     linear_expression_dict[item[0]] = str(solve([item[1] < 0]))
  # for item in ans.items():
    #     if item[0] not in lamuda_symbol_list and item[1] != 0:
    #         points_dict[item[0]] = list(map(int, regular.findall('-?\d+', str(solve([item[1] < 0])))))
    #
    # print("\nfij在此区间内小于0：\n", linear_expression_dict)
    # print("\n对应的跳跃点：\n", points_dict)
    # # print("\nfij<0：\n", lower_0_set)
    #
    # ineq_expression_dict = {}
    # if lower_0_flag == True:
    #     for j in range(config.cs_num):
    #         expr = 0
    #         for i in range(config.region_num):
    #             expr += ans[fij_symbol_list[i][j]]
    #         q_symbol_list[j] = expr
    #
    #     lower_0_fij_list = []
    #     for i in range(config.region_num):
    #         for j in range(config.cs_num):
    #             if(ans[fij_symbol_list[i][j]] == 0):
    #                 for k in range(config.region_num):
    #                     fij_name = str(i) + str(j) + str(k)
    #                     low_0_fij_name = str(i) + "," + str(j)
    #                     lower_0_fij_list.append(low_0_fij_name)
    #                     if j == 0:
    #                         expr = dist[j][i] + q_symbol_list[j] + p - ans[lamuda_symbol_list[k]]
    #                     else:
    #                         expr = dist[j][i] + q_symbol_list[j] + price[j] - ans[lamuda_symbol_list[k]]
    #                     ineq_expression_dict[fij_name] = expr
    #
    #     possible_p_set = set()
    #     for item in ineq_expression_dict.items():
    #         points_dict[item[0]] = solve(item[1])
    #
    #
    # min_P_fij =  Queue()
    # min_P_fij.put(0)
    # min_p = Queue()
    # min_p.put(0)
    # for item in points_dict.items():
    #     if item[1][0] > price[0]:
    #         head = min_p.get()
    #         head_fij = min_P_fij.get()
    #         if head == 0:
    #             min_p.put(item[1][0])
    #             min_P_fij.put(item[0])
    #         elif head == item[1][0]:
    #             min_p.put(head)
    #             min_P_fij.put(head_fij)
    #             min_p.put(item[1][0])
    #             min_P_fij.put(item[0])
    #         elif head > item[1][0]:
    #             min_p.put(item[1][0])
    #             min_P_fij.put(item[0])
    #         elif head < item[1][0]:
    #             min_p.put(head)
    #             min_P_fij.put(head_fij)
    # print(min_p.queue)
    # print(min_P_fij.queue)
    #
    # flag = [[1 for j in range(config.cs_num)] for i in range(config.region_num)]
    # if lower_0_flag == True:
    #     for i in lower_0_fij_list:
    #         nums = i.split(",")
    #         flag[int(nums[0])][int(nums[1])] = 0
    #
    # result_jump_p.append(min_p.get())
    #
    #
    # # 求解该区间内的最优price和revenue
    # revenue, flow_list, optimal_p, lamuda_list = get_max_revenue(dist, region_num, cs_num, priceIndex, price, vehicle_num,
    #                                                         flag, result_jump_p, index)
    # revenue_list.append(revenue)
    # revenue_max_p_list.append(optimal_p)
    #
    # price[0] = result_jump_p[0]
    #
    #
    # # p0=130，求下一个跳跃点
    # # Q用fij表示：
    # for j in range(config.cs_num):
    #     expr = 0
    #     for i in range(config.region_num):
    #         expr += fij_symbol_list[i][j]
    #     q_symbol_list[j] = expr
    #
    # if lower_0_flag == True:
    #     for ex in lower_0_fij_list:
    #         nums = ex.split(",")
    #         int_i = int(nums[0])
    #         int_j = int(nums[1])
    #         expr_f = nums[0] + nums[1]
    #         if int(nums[1]) == 0:
    #             expression_dict[expr_f] = fij_symbol_list[int_i][int_j] + q_symbol_list[int_j] + dist[int_j][int_i] + p - lamuda_symbol_list[int_i]
    #         else:
    #             expression_dict[expr_f] = fij_symbol_list[int_i][int_j] + q_symbol_list[int_j] + dist[int_j][int_i] + price[int_j] - lamuda_symbol_list[int_i]
    # lower_0_flag = False
    # ans = solve(expression_dict.values())
    # print("\n将其他变量表示为关于p的线性函数：\n", ans)
    #
    # linear_expression_dict = {}
    # points_dict = {}
    #
    # for item in ans.items():
    #     linear_expression_dict[item[0]] = str(solve([item[1] < 0]))
    # for item in ans.items():
    #     if item[0] not in lamuda_symbol_list and item[1] != 0:
    #         points_dict[item[0]] = list(map(int, regular.findall('-?\d+', str(solve([item[1] < 0])))))
    #
    # print("\nfij在此区间内小于0：\n", linear_expression_dict)
    # print("\n对应的跳跃点：\n", points_dict)
    #
    # ineq_expression_dict = {}
    # if lower_0_flag == True:
    #     for j in range(config.cs_num):
    #         expr = 0
    #         for i in range(config.region_num):
    #             expr += ans[fij_symbol_list[i][j]]
    #         q_symbol_list[j] = expr
    #
    #     lower_0_fij_list = []
    #     for i in range(config.region_num):
    #         for j in range(config.cs_num):
    #             if (ans[fij_symbol_list[i][j]] == 0):
    #                 for k in range(config.region_num):
    #                     fij_name = str(i) + str(j) + str(k)
    #                     low_0_fij_name = str(i) + "," + str(j)
    #                     lower_0_fij_list.append(low_0_fij_name)
    #                     if j == 0:
    #                         expr = dist[j][i] + q_symbol_list[j] + p - ans[lamuda_symbol_list[k]]
    #                     else:
    #                         expr = dist[j][i] + q_symbol_list[j] + price[j] - ans[lamuda_symbol_list[k]]
    #                     ineq_expression_dict[fij_name] = expr
    #
    #     for item in ineq_expression_dict.items():
    #         points_dict[item[0]] = solve(item[1])
    #
    # min_P_fij = Queue()
    # min_P_fij.put(0)
    # min_p = Queue()
    # min_p.put(0)
    # for item in points_dict.items():
    #     if len(item[1]) == 2:
    #         points_dict[item[0]] = item[1][0] / item[1][1]
    #     if item[1][0] > price[0]:
    #         head = min_p.get()
    #         head_fij = min_P_fij.get()
    #         if head == 0:
    #             min_p.put(item[1][0])
    #             min_P_fij.put(item[0])
    #         elif head == item[1][0]:
    #             min_p.put(head)
    #             min_P_fij.put(head_fij)
    #             min_p.put(item[1][0])
    #             min_P_fij.put(item[0])
    #         elif head > item[1][0]:
    #             min_p.put(item[1][0])
    #             min_P_fij.put(item[0])
    #         elif head < item[1][0]:
    #             min_p.put(head)
    #             min_P_fij.put(head_fij)
    # print(min_p.queue)
    # print(min_P_fij.queue)
    #
    # flag = [[1 for j in range(config.cs_num)] for i in range(config.region_num)]
    # if lower_0_flag == True:
    #     for i in lower_0_fij_list:
    #         nums = i.split(",")
    #         flag[int(nums[0])][int(nums[1])] = 0
    #
    # result_jump_p.append(min_p.get())
    #
    # # 求解该区间内的最优price和revenue
    # revenue, flow_list, optimal_p, lamuda_list = get_max_revenue(dist, region_num, cs_num, priceIndex, price,
    #                                                              vehicle_num,
    #                                                              flag, result_jump_p, index+1)
    # revenue_list.append(revenue)
    # revenue_max_p_list.append(optimal_p)
    #
    # price[0] = result_jump_p[0]
