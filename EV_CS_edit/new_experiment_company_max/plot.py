from matplotlib import pyplot as plt
import json
own_cs_revenue_list = []
with open('change_cs.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip('\n')
        list_change = json.loads(line)
        own_cs_revenue_list.append(list_change)

# list_revenue = json.loads(own_cs_revenue_list)



color = ['red', 'green', 'blue', 'black', 'yellow', 'gray', 'chocolate', 'orange', 'purple', 'pink', 'lightgreen']
marker = ['o', '.', '1', '+']

plt.title("company revenue curve under different cs nums")
plt.xlabel("num")
plt.ylabel("revenue")

plt.plot(range(len(own_cs_revenue_list[0])), own_cs_revenue_list[0], color=color[0],
         label='cs nums is 5 revenue curve', marker=marker[0], markersize=1 )
plt.plot(range(len(own_cs_revenue_list[1])), own_cs_revenue_list[1], color=color[1],
         label='cs nums is 10 revenue curve', marker=marker[1], markersize=1 )
plt.plot(range(len(own_cs_revenue_list[2])), own_cs_revenue_list[2], color=color[2],
         label='cs nums is 15 revenue curve', marker=marker[2], markersize=1 )
plt.plot(range(len(own_cs_revenue_list[3])), own_cs_revenue_list[3], color=color[3],
         label='cs nums is 20 revenue curve', marker=marker[3], markersize=1 )
plt.legend()  # 显示图例
plt.show()
