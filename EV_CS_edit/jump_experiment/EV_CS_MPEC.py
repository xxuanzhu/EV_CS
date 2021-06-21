#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/4 22:44
# @Author : wbw
# @Version：V 0.1
# @File : EV_CS_MPEC.py
# @desc : 这里用pyomo来求解MPEC问题

from config import config
from pyomo.environ import *
from pyomo.mpec import *

N = config.region_num
M = config.cs_num
dist_vector = config.dist_vector
vehicle_vector = config.vehicle_vector
price_list = config.total_price

model = ConcreteModel()
model.priceIndex = range(4)  # 表示有四个需要研究的充电站
model.region = range(N)
model.CS = range(M)

model.p = Var(model.priceIndex, bounds=(40.0, 90.0))
model.f = Var(model.region, model.CS, bounds=(0.0, 1.0))  # 车流量
model.v = Var(model.region, model.CS, bounds=(0, None))  # 不等式约束的乘子，互补条件之一
model.lamuda = Var(model.region)  # 等式约束的乘子

model.obj = Objective(expr=sum(model.p[k]*sum(vehicle_vector[i]*model.f[i, k] for i in model.region)
                               for k in model.priceIndex), sense=maximize)

# 原问题的等式线性约束，一共有n个
model.single_f = ConstraintList()
for i in model.region:
    model.single_f.add(sum(model.f[i, j] for j in model.CS) == 1.0)

# 拉格朗日乘子约束，一共有m * n个
model.lagrange = ConstraintList()
for i in model.region:
    for j in model.CS:
        if j < 4:
            model.lagrange.add((model.p[j] + sum(model.f[k, j] for k in model.region) + dist_vector[i][j] +
                               model.f[i, j]*vehicle_vector[i])*vehicle_vector[i] - model.v[i, j] - model.lamuda[i] == 0.0)
        else:
            model.lagrange.add((price_list[j] + sum(model.f[k, j] for k in model.region) + dist_vector[i][j] +
                               model.f[i, j] * vehicle_vector[i])*vehicle_vector[i] - model.v[i, j] - model.lamuda[i] == 0.0)


# 互补约束，一共有m * n个
model.compl = ComplementarityList()
for i in model.region:
    for j in model.CS:
        model.compl.add(complements(model.f[i, j] >= 0, model.v[i, j] >= 0))

opt = SolverFactory('ipopt')
opt.solve(model)

flow_list = []
p_list = []
v_list = []
lamuda_list = []
print('\nDecision Variables')
for i in model.region:
    temp_flow_list = []
    for j in model.CS:
        temp_flow_list.append(round(model.f[i, j](), 3))
    flow_list.append(temp_flow_list)

for j in model.priceIndex:
    p_list.append(round(model.p[j](), 3))


for i in model.region:
    temp_v_list = []
    for j in model.CS:
        temp_v_list.append(round(model.v[i, j](), 3))
    v_list.append(temp_v_list)

for i in model.region:
    lamuda_list.append(round(model.lamuda[i](), 3))

print("profits:")
print(model.obj())
print("f_ij:")
for i in model.region:
    print(flow_list[i])
print("p_j:")
print(p_list)
print("v_ij:")
for i in model.region:
    print(v_list[i])
print("lamuda:")
print(lamuda_list)