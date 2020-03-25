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

if __name__ == "__main__":        
    file = open('data/CongreG8/CongreG8.pkl', 'rb')
    data = pkl.load(file)
    file.close()

    group_size = 4
    sample_nums = len(data)

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

    
