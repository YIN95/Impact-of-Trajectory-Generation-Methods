import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

def draw_graph(xs, ys, index):
    # plt.clf()
    # for i in range(len(xs)):   
    #     plt.plot(xs[i], ys[i], "-g")
    plt.plot(xs, ys)
    plt.savefig('logs/visualize/'+str(index)+'.png')
    pass

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
if __name__ == "__main__":        
    file = open('data/CongreG8/CongreG8.pkl', 'rb')
    data = pkl.load(file)
    file.close()

    group_size = 4
    sample_nums = len(data)
    
    r_list = []
    for sample_index in range(sample_nums):
        sample_data = data[sample_index]
        print(sample_index)
        for t in range(sample_data.shape[2]):
            p1 = Point(sample_data[0, 0, t, 5, 0], sample_data[0, 2, t, 5, 0])
            p2 = Point(sample_data[0, 0, t, 5, 1], sample_data[0, 2, t, 5, 1])
            p3 = Point(sample_data[0, 0, t, 5, 2], sample_data[0, 2, t, 5, 2])
            r_x = (p1.x + p2.x + p3.x) / 3.0
            r_y = (p1.y + p2.y + p3.y) / 3.0
            d_1 = ((p1.x - r_x) ** 2 + (p1.y - r_y) ** 2)
            d_2 = ((p2.x - r_x) ** 2 + (p2.y - r_y) ** 2)
            d_3 = ((p3.x - r_x) ** 2 + (p3.y - r_y) ** 2)
            d_max = max(d_1, d_2, d_3)
            r_list.append(d_max)
    
    r = sum(r_list)/(len(r_list))
    print(r)
    # 

    group_agents_data = []
    for sample_index in range(sample_nums):
        group_agents_x = []        
        group_agents_y = []
        sample_data = data[sample_index]

        for agent_index in range(group_size):
            
            group_agents_x.append(sample_data[0, 0,:, 5, agent_index])
            group_agents_y.append(sample_data[0, 2,:, 5, agent_index])
        
        plt.figure()
        for agent_index in range(group_size):  
            draw_graph(group_agents_x[agent_index], group_agents_y[agent_index], sample_index)
        
        pass

    
