from __future__ import division
import pyomo.environ as pyo

model = pyo.AbstractModel()

model.m = pyo.Param(within=pyo.NonNegativeIntegers)  # cs nums
model.n = pyo.Param(within=pyo.NonNegativeIntegers) # region nums

model.J = pyo.RangeSet(1, model.m)
model.I = pyo.RangeSet(1, model.n)

model.f_i_j = pyo.Param(model.I, model.J)
model.f_j = pyo.Param(model.J)


# j桩的价格是变量
model.p = pyo.Var(model.J, domain=[0, 213.5])



