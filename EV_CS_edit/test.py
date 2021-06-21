import sympy
from matplotlib import pyplot as plt

from sympy import *

jp = Symbol('jp')
# expr = jp * (7999523/6290 - 1897 * jp / 3145)
expr = jp * (4 - 2 * jp)
fd = diff(expr)
fdd = diff(fd)
print(fd)
print(fdd)
print(fd.subs(jp, 2))
polyRoots = solveset(expr, jp)
print(polyRoots)
dRoots = solveset(fd, jp)
print(dRoots)
ddRoots = solveset(fdd, jp)
print(dRoots)

