import scipy.io
from os import listdir
from os.path import isfile, join
import csv
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt

def colorselect(num):
    if num==0:
        return 'blue'
    elif num==1:
        return 'green'
    elif num==2:
        return 'black'
    elif num==3:
        return 'violet'
    elif num==4:
        return 'orange'
    elif num==5:
        return 'grey'
    elif num==6:
        return 'purple'
    elif num==7:
        return 'crimson'
    elif num==8:
        return 'lime'
    elif num==9:
        return 'yellow'
    elif num==10:
        return 'indigo'
    else:
        return 'pink'


def loadsalsa(isprint=False, dataset=2):
    filepath = "data/Salsa/salsa"+str(dataset)+"/geometryGT/"
    groupfilepath = "data/Salsa/salsa"+str(dataset)+"/fformationGT.csv"

    data = [filepath + f for f in listdir(filepath) if isfile(join(filepath, f))]
    totaldata = []
    groupdata = []

    colors = ['blue', 'green', 'black', 'violet', 'orange', 'grey', 'purple', 'crimson', 'lime', 'yellow', 'indigo',
              'pink',
              'olive', 'gold', 'darkgreen', 'navy', 'maroon', 'teal', 'cyan']

    with open(groupfilepath) as csvfile:
        rowreader = csv.reader(csvfile, delimiter=',')
        for row in rowreader:
            row = [float(irow) for irow in row]
            groupdata.append(row)

    for idata in data:
        with open(idata) as csvfile:
            rowreader = csv.reader(csvfile, delimiter=',')
            agentdata=[]
            for row in rowreader:
                row = [float(irow) for irow in row]
                agentdata.append(row)
            totaldata.append(agentdata)

    previousframe=0.2
    currentframe=0
    groupnum = 0
    for i in range(len(groupdata)):
        currentframe=groupdata[i][0]
        for k in range(len(groupdata[i])-1):
            k=k+1
            idata=totaldata[int(groupdata[i][k])-1]
            for j in range(len(idata)):
                if idata[j][0] == currentframe:
                    idata[j].append(groupnum)
                    break
        # for ind, data in enumerate(totaldata):
        #     for j in range(len(data)):
        #         if data[j][0]==currentframe and ind in groupdata[i]:
        #             data[j].append(groupnum)
        #             break
        if currentframe==previousframe:
            groupnum+=1
        else:
            previousframe=currentframe
            groupnum=0

    frame_num = len(totaldata[0])
    xmaxlist=[]
    xminlist=[]
    ymaxlist=[]
    yminlist=[]
    for agent in totaldata:
        xlist=[row[1] for row in agent]
        ylist=[row[2] for row in agent]
        xmaxlist.append(max(xlist))
        xminlist.append(min(xlist))
        ymaxlist.append(max(ylist))
        yminlist.append(min(ylist))

    xmax=max(xmaxlist)
    ymax=max(ymaxlist)
    xmin=min(xminlist)
    ymin=min(yminlist)

    boundary=[xmax,ymax,xmin,ymin]
    if isprint:
        for i in range(frame_num):
            plt.figure(i)
            axes = plt.gca()
            axes.set_xlim([xmin, xmax])
            axes.set_ylim([ymin, ymax])
            points = []
            for ind, agent in enumerate(totaldata):
                points.append([agent[i][1],agent[i][2],agent[i][4],agent[i][-1],ind])
            for id, point in enumerate(points):
                color =colorselect(point[3])
                plt.plot(float(point[0]), float(point[1]), marker='o', color=colors[point[4]], markersize=12)
                bodypos=(point[0]+0.2*math.cos(point[2]), point[1]+0.2*math.sin(point[2]))
                plt.plot(bodypos[0], bodypos[1], 'ro', markersize=8)
                ax=plt.gca()
                ax.annotate(str(id), (float(point[0]), float(point[1])))
            plt.savefig('figures/salsa'+str(dataset)+'/figure%d.png' % i)
            plt.close()

    return totaldata, boundary
