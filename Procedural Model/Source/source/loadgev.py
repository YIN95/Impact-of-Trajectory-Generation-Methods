import scipy.io
from os import listdir
from os.path import isfile, join
import csv
from math import *
import numpy as np
import h5py
import matplotlib.pyplot as plt
import string
import copy
import math
import random

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

def loadgve(isprint=False):
    filepath = "data/GVE/trajectories.txt"
    groupfilepath = "data/GVE/clusters.txt"
    f=open(filepath,'r')
    rows=f.readlines()
    totaldata=[]
    groupdata=[]
    currentind = 1
    previousind = 1
    frame = []

    colors = ['aqua', 'green', 'azure', 'violet', 'orange', 'grey', 'purple', 'crimson', 'lime', 'yellow', 'indigo',
              'pink',
              'olive', 'gold', 'darkgreen', 'navy', 'maroon', 'teal', 'cyan', 'ivory','orchid','plum','yellowgreen','silver','lightgreen',
              'orangered','tan','turquoise','wheat','salmon','sienna','goldenrod','fuchsia','coral','brown']

    for row in rows:
        row=row.strip().split(',')
        row=row[0].split(' ')
        row=[float(x) for x in row]
        currentind = row[0]
        if previousind != currentind:
            previousind = currentind
            copyframe=copy.deepcopy(frame)
            totaldata.append(copyframe)
            frame.clear()
            frame.append(row)
        else:
            frame.append(row)
    f.close()

    f=open(groupfilepath,'r')
    rows=f.readlines()
    for row in rows:
        row=row.strip().split(',')
        row=row[0].split(' ')
        row=[float(x) for x in row]
        groupdata.append(row)
    f.close()

    for ind, frame in enumerate(totaldata):
        agents=[agent[1] for agent in frame]
        maxagentid=max(agents)
        groupid=1
        for grow in groupdata:
            if max(grow)<=maxagentid and min(grow)>=min(agents):
                if len(grow)==1 and id in agents:
                    continue
                foundgroup = True
                for id in grow:
                    if id not in agents:
                        foundgroup=False
                if foundgroup==True:
                    for id in grow:
                        totaldata[ind][agents.index(id)].append(groupid)
                    groupid+=1

    for ind, frame in enumerate(totaldata):
        for id, row in enumerate(frame):
            if len(row)==8:
                totaldata[ind][id].append(0)

    for ind in np.arange(len(totaldata)-1):
        currentagents=[agent[1] for agent in totaldata[ind+1]]
        previousagents=[agent[1] for agent in totaldata[ind]]
        commonagents=[]
        for id in currentagents:
            if id in previousagents:
                commonagents.append(id)
        if commonagents:
            for icommon in commonagents:
                curid=currentagents.index(icommon)
                previd=previousagents.index(icommon)
                curpos=[totaldata[ind+1][curid][4],totaldata[ind+1][curid][2]]
                prevpos=[totaldata[ind][previd][4],totaldata[ind][previd][2]]
                vec=[curpos[0]-prevpos[0],curpos[1]-prevpos[1]]
                totaldata[ind+1][curid].append(atan2(vec[1],vec[0]))

    for ind, frame in enumerate(totaldata):
        for id, row in enumerate(frame):
            if len(row) == 9:
                totaldata[ind][id].append(-10)

    # print(np.array(totaldata[0]))

    frame_num = len(totaldata)
    xmaxlist=[]
    xminlist=[]
    ymaxlist=[]
    yminlist=[]
    for framedata in totaldata:
        xlist=[row[4] for row in framedata]
        ylist=[row[2] for row in framedata]
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
            if i%3!=0:
                continue
            plt.figure(i)
            axes = plt.gca()
            axes.set_xlim([xmin, xmax])
            axes.set_ylim([ymin, ymax])
            points = []
            for ind, agent in enumerate(totaldata[i]):
                points.append([agent[4],agent[2],agent[9],int(agent[1])])
            for id, point in enumerate(points):
                # color =colorselect(point[3])
                pointcolor=point[3]
                if point[3]>=len(colors):
                    pointcolor=point[3]%len(colors)

                plt.plot(float(point[0]), float(point[1]), marker='o', color=colors[pointcolor], markersize=12)
                pointorient=point[2]
                if pointorient==-10:
                    pointorient=random.uniform(pi, -pi)
                bodypos=(point[0]+0.2*math.cos(pointorient), point[1]+0.2*math.sin(pointorient))
                plt.plot(bodypos[0], bodypos[1], 'ro', markersize=8)
                ax=plt.gca()
                ax.annotate(str(point[3]), (float(point[0]), float(point[1])))
            plt.savefig('figures/gev/figure%d.png' % i)
            plt.close()


    return totaldata


def main():
    loadgve(isprint=True)

if __name__=='__main__':
    main()