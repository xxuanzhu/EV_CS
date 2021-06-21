from EV_Equilibrium import EvEquilibrium
import numpy as np
from matplotlib import pyplot as plt
from config import config

evEquilibrium = EvEquilibrium()


if __name__ == "__main__":
    # p_1 = 1
    p_2 = 50
    region_num = config.region_num
    minimize_res_list = []
    vehicle_vector = config.vehicle_vector

    dist1, dist2, vehicle_num, region, strategy_vector1, strategy_vector2 = evEquilibrium.initiation(region_num)
    print("初始化dist1, dist2, 车辆数， agent, 策略向量1， 策略向量2分别为：", dist1, dist2, vehicle_num,
          region, strategy_vector1, strategy_vector2)

    # 下面是算出Q和p的关系图
    flow_list = []
    flow_list_second = []
    p1_min = 30
    p1_max = 70
    price_list = np.arange(p1_min, p1_max)
    # 计算CS1的收益 V_1 = p_1 * f_1
    for p_1 in np.arange(p1_min, p1_max):
        print("当前定价", p_1)
        # 求得此定价下底层的策略
        strategy1, strategy2 = evEquilibrium.best_response_simulation(region, dist1, dist2, vehicle_num,
                                                                      p_1, p_2, minimize_res_list, strategy_vector1,
                                                                      strategy_vector2)
        print("CS1的策略：", strategy1)
        # revenue_temp = 0
        f_i_1 = 0
        f_i_2 = 0
        for item in range(region_num):
            f_i_1 = f_i_1 + vehicle_vector[item] * strategy1[item]  # 所有区域派到充电站1的车辆数量
            f_i_2 = f_i_2 + vehicle_vector[item] * strategy2[item]  # 所有区域派到充电站2的车辆数量

        flow_list.append(f_i_1)
        flow_list_second.append(f_i_2)

    print("距离向量1：", dist1)
    print("距离向量2：", dist2)

    plt.title("Q and P curve")
    plt.xlabel("price")
    plt.ylabel("queue")
    plt.plot(price_list, flow_list, color='red', label='CS1 Q')
    plt.plot(price_list, flow_list_second, color='green', label='CS2 Q')
    plt.legend()  # 显示图例
    plt.show()
