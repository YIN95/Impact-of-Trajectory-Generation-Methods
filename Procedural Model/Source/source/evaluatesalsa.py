import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
import numpy as np
from math import *
from utilities.pointInside import *
from utilities.gradient import *
from loadsalsa import *
import vectormath as vmath
from rtree import index
from shapely.geometry.polygon import LinearRing, Polygon
import skfmm
from scipy.optimize import minimize
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import statistics


def get_dis(humanpos, pos):
    return sqrt((humanpos.x - pos.x) ** 2 + (humanpos.y - pos.y) ** 2)


def get_closet_dis(point, line):
    a = point
    b, c = line
    t = (a[0] - b[0]) * (c[0] - b[0]) + (a[1] - b[1]) * (c[1] - b[1])
    t = t / ((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2)
    a = vmath.Vector2(a[0], a[1])
    b = vmath.Vector2(b[0], b[1])
    c = vmath.Vector2(c[0], c[1])

    if 0 < t < 1:
        pcoords = vmath.Vector2(t * (c.x - b.x) + b.x, t * (c.y - b.y) + b.y)
        dmin = get_dis(a, pcoords)
        return pcoords, dmin
    elif t <= 0:
        return b, get_dis(a, b)
    elif 1 <= t:
        return c, get_dis(a, c)


def get_ang(humanpos, pos):
    return atan2(pos.y - humanpos.y, pos.x - humanpos.x)


def human_hand_pose(humanpos, lhandpos, rhandpos):
    dlh = get_dis(humanpos, lhandpos)
    drh = get_dis(humanpos, rhandpos)
    alh = get_ang(humanpos, lhandpos)
    arh = get_ang(humanpos, rhandpos)
    return dlh, drh, alh, arh


def get_speed_id(x, y, pos, dx, size):
    for ix in range(size):
        for iy in range(size):
            posx = x[0][ix]
            posy = y[iy][0]
            if abs(posx - pos[0]) <= dx and abs(posy - pos[1]) <= dx:
                return [ix, iy]
    return [0, 0]


class Group:
    def __init__(self, humans, stride=0.3, radiusstep=0.25, A=255, fg=0.0):
        self.humans = humans
        self.stride = stride
        self.groupcenter = self.get_group_center()
        self.radiusstep = radiusstep
        self.A = A
        self.fg = fg

    def get_group_center(self):
        initpos = self.humans[0].hpos
        result = minimize(self.group_center_dis_sum, initpos)
        if result.success:
            groupcenter = result.x
            return groupcenter
        else:
            raise ValueError(result.message)

    def group_center_dis_sum(self, pos):
        res = 0
        for ihuman in self.humans:
            x = ihuman.hpos.x + self.stride * cos(ihuman.htheta)
            y = ihuman.hpos.y + self.stride * sin(ihuman.htheta)
            res += (pos[0] - x) ** 2 + (pos[1] - y) ** 2
        return res

    def group_interaction(self, pos):
        d = sqrt((self.groupcenter[0] - pos.x) ** 2 + (self.groupcenter[1] - pos.y) ** 2)
        sigma = 3 * self.radiusstep * (1 + self.fg)
        f = self.A * exp(-(d / (sqrt(2) * sigma)) ** 2)
        return f


class Wall:
    def __init__(self, linepoints, boundary, opt=True, minsize=2.0, theta=0.3, A=255):
        self.linepoints = linepoints
        self.minsize = minsize
        self.theta = theta
        self.opt = opt
        self.boundary = boundary
        if self.opt:
            self.idx = self.get_rtree()
        self.A = A

    def get_rtree(self):
        def generate_items():
            sindx = 0
            for idx, l in enumerate(self.linepoints):
                if idx == 0:
                    continue
                a, b = self.linepoints[idx - 1]
                c, d = l
                segment = ((a, b), (c, d))
                box = (min(a, c), min(b, d), max(a, c), max(b, d))
                # box = left, bottom, right, top
                yield (sindx, box, (idx, segment))
                sindx += 1

        return index.Index(generate_items())

    def test_point_in_wall(self, point):
        # return is_inside_sm(self.linepoints, point)
        return cn_PnPoly(point, self.linepoints)

    def get_close_point_nonopt(self, point):
        d = inf
        for idx, line in enumerate(self.linepoints):
            if idx == 0:
                continue
            nearest_p, new_d = get_closet_dis(point, (self.linepoints[idx - 1], line))
            if d >= new_d:
                d = new_d

        return d

    def get_close_point(self, point):
        pbox = (point[0] - self.minsize, point[1] - self.minsize, point[0] + self.minsize, point[1] + self.minsize)
        hits = self.idx.intersection(pbox, objects='raw')
        d = inf
        for h in hits:
            nearest_p, new_d = get_closet_dis(point, h[1])
            if d >= new_d:
                d = new_d
        return d

    def get_wall_interaction(self, point):
        # p=(point.x,point.y)
        if self.opt:
            s = self.get_close_point(point)
        else:
            s = self.get_close_point_nonopt(point)
        if s == inf:
            return 0
        f = self.A * exp(-s / self.theta)
        return f

    def get_boundary_interaction(self, point):
        s = min(abs(point[0] - self.boundary[0]), abs(point[0] - self.boundary[2]),
                abs(point[1] - self.boundary[1]), abs(point[1] - self.boundary[3]))
        return self.A * exp(-s / self.theta)


class Human:
    def __init__(self, hpos, hgaze, objpos=None, hmotion=None, A=255, hsigmax=0.45, hsigmay=0.45, handsigmax=0.28,
                 handsigmay=0.28,
                 Tdis=3.6, Tang=10 * pi / 180, fov2ang=140 * pi / 180, fv=0.8, fh=1.68, ffront=0.2, ffov=0.0, fsit=1,
                 fnorm=0.6, hlength=0.2, w1=1.0, w2=1.0, hv=0):
        self.hpos = hpos
        self.objpos = objpos
        self.A = A
        self.hsigmax = hsigmax
        self.hsigmay = hsigmay
        self.hv = hv
        self.Tdis = Tdis
        self.Tang = Tang
        self.fov2ang = fov2ang
        self.fv = fv
        self.ffront = ffront
        self.ffov = ffov
        self.fsit = fsit
        self.fh = fh
        self.hlength = hlength
        self.handsigmax = handsigmax
        self.handsigmay = handsigmay
        self.w1 = w1
        self.w2 = w2
        self.fnorm = fnorm
        if self.hv == 0:
            self.htheta = hgaze
        elif self.hv > 0:
            self.htheta = hmotion

    def basic_personal_space(self, pos):
        d = get_dis(pos, self.hpos)
        ang = get_ang(self.hpos, pos)
        posvec = [pos[0] - self.hpos[0], pos[1] - self.hpos[1]]
        headvec = [cos(self.htheta), sin(self.htheta)]
        dir = headvec[0] * posvec[1] - headvec[1] * posvec[0]
        angle = np.arccos(np.dot(posvec, headvec) / (np.linalg.norm(posvec) * np.linalg.norm(headvec)))

        if abs(angle) <= pi / 2:
            if self.hv == 0:
                if abs(angle) <= self.fov2ang / 2:
                    sigmaynew = (1 + self.hv * self.fv + self.ffront * self.fsit + self.ffov) * self.hsigmay
                    sigmaxnew = self.hsigmax
                else:
                    sigmaynew = (1 + self.hv * self.fv + self.ffront * self.fsit) * self.hsigmay
                    sigmaxnew = self.hsigmax
            elif self.hv > 0:
                if abs(angle) <= self.fov2ang / 2:
                    if dir > 0:
                        sigmaynew = (1 + self.hv * self.fv + self.ffront * self.fsit + self.ffov) * self.hsigmay
                        sigmaxnew = (1 + self.fnorm) * self.hsigmax
                    else:
                        sigmaynew = (1 + self.hv * self.fv + self.ffront * self.fsit + self.ffov) * self.hsigmay
                        sigmaxnew = self.hsigmax
                else:
                    if dir > 0:
                        sigmaynew = (1 + self.hv * self.fv + self.ffront * self.fsit) * self.hsigmay
                        sigmaxnew = (1 + self.fnorm) * self.hsigmax
                    else:
                        sigmaynew = (1 + self.hv * self.fv + self.ffront * self.fsit) * self.hsigmay
                        sigmaxnew = self.hsigmax
        else:
            if self.hv == 0:
                sigmaynew = self.hsigmay
                sigmaxnew = self.hsigmax
            elif self.hv > 0:
                # if -pi/2>ang - self.htheta >-pi:
                if dir > 0:
                    sigmaynew = self.hsigmay
                    sigmaxnew = (1 + self.fnorm) * self.hsigmax
                else:
                    sigmaynew = self.hsigmay
                    sigmaxnew = self.hsigmax

        try:
            f = self.A * exp(
                -((d * cos(ang - self.htheta) / (sqrt(2) * sigmaynew)) ** 2 + (
                            d * sin(ang - self.htheta) / (sqrt(2) * sigmaxnew)) ** 2))
        except:
            print(sigmaynew, pos)

        lhpos = self.get_handpos('left')
        rhpos = self.get_handpos('right')
        lhang = get_ang(pos, lhpos)
        rhang = get_ang(pos, rhpos)
        dl = get_dis(pos, lhpos)
        dr = get_dis(pos, rhpos)

        flh = self.A * exp(
            -((dl * sin(lhang - self.htheta) / (sqrt(2) * self.handsigmax)) ** 2 +
              (dl * cos(lhang - self.htheta) / (sqrt(2) * (1 + self.fh * self.hlength) * self.handsigmay)) ** 2))

        frh = self.A * exp(
            -((dr * sin(rhang - self.htheta) / (sqrt(2) * self.handsigmax)) ** 2 +
              (dr * cos(rhang - self.htheta) / (sqrt(2) * (1 + self.fh * self.hlength) * self.handsigmay)) ** 2))

        f = max(self.w1 * f, self.w2 * flh, self.w2 * frh)

        return f

    def object_interaction(self, pos):
        d = get_dis(pos, self.hpos)
        ang = get_ang(self.hpos, pos)

        # if self.objpos.all() != None:
        # if pos is in the frontal area of human
        if abs(ang - self.htheta) <= pi / 2:
            dobj = get_dis(self.objpos, self.hpos)
            angobj = get_ang(self.hpos, self.objpos)
            diffang = abs(angobj - self.htheta)
            sigmaynew = 0.0
            if dobj < self.Tdis and diffang < self.Tang:
                # sigmaynew = self.hsigmay
                sigmaynew = dobj / 2
        else:
            sigmaynew = 0.0

        if sigmaynew == 0.0:
            return 0.0
        else:
            f = self.A * exp(
                -((d * cos(ang - self.htheta) / (sqrt(2) * sigmaynew)) ** 2 + (
                            d * sin(ang - self.htheta) / (sqrt(2) * self.hsigmax)) ** 2))
        return f

    def get_handpos(self, hand='right'):
        forward = vmath.Vector2(cos(self.htheta), sin(self.htheta))

        if hand == 'right':
            hdir = vmath.Vector2(forward.y, -forward.x)
            hdir = hdir / hypot(hdir.x, hdir.y)
            hand = self.hpos + hdir * self.hlength
        elif hand == 'left':
            hdir = vmath.Vector2(-forward.y, forward.x)
            hdir = hdir / hypot(hdir.x, hdir.y)
            hand = self.hpos + hdir * self.hlength

        return hand


def get_group_index(grouplist):
    x = []
    for a in grouplist:
        if a not in x:
            x.append(a)
    groups = []
    for ix in x:
        group = []
        for ind, ig in enumerate(grouplist):
            if ig == ix:
                group.append(ind)
        groups.append(group)
    return groups


def get_pos_id(x, y, pos, dx):
    value = [0, 0]
    for ind, ix in enumerate(x[0]):
        if abs(ix - pos[0]) <= dx:
            value[0] = ind
            break
    for ind, iy in enumerate(np.transpose(y)[0]):
        if abs(iy - pos[1]) <= dx:
            value[1] = ind
            break
    return value


def make_obstacles(grid, pos, band, x, y, dx):
    size = len(x)
    posid = get_pos_id(x, y, pos, dx)
    ids = [posid[0], posid[0], posid[1], posid[1]]
    if posid[0] - band >= 0:
        ids[0] = posid[0] - band
    else:
        ids[0] = 0
    if posid[0] + band < size - band:
        ids[1] = posid[0] + band
    else:
        ids[1] = size - 1
    if posid[1] - band >= 0:
        ids[2] = posid[1] - band
    else:
        ids[2] = 0
    if posid[1] + band < size - band:
        ids[3] = posid[1] + band
    else:
        ids[3] = size - 1

    idx = np.arange(ids[0], ids[1] + 1)
    idy = np.arange(ids[2], ids[3] + 1)
    for ix in idx:
        for iy in idy:
            grid[iy][ix] = 0


def truncate(px, py, newstart):
    for ind, ix in enumerate(px):
        if ind - 1 >= 0:
            if px[ind] == px[ind - 1] and py[ind] == py[ind - 1]:
                return ind - 1


def readextractedpaths(dataset=1):
    filepath = "data/Salsa/extractedpaths/salsa" + str(dataset) + "paths.csv"
    with open(filepath) as csvfile:
        rowreader = csv.reader(csvfile, delimiter=',')
        trajectorydata = []
        for row in rowreader:
            row = [float(irow) for irow in row]
            trajectorydata.append(row)

    return trajectorydata


def main():
    dataset = 2
    totaldata, boundary = loadsalsa(isprint=False, dataset=dataset)
    trajectorydata = readextractedpaths(dataset=dataset)
    N = 200

    obstacles1 = [(8, 1), (8, 2.5), (8.5, 2.5), (8.5, 1), (8, 1)]
    obstacles2 = [(8, 3), (8, 4.5), (8.5, 4.5), (8.5, 3), (8, 3)]
    obstacles3 = [(9.2, 6.9), (10.2, 6.9), (10.2, 6.7), (9.2, 6.7), (9.2, 6.9)]
    obstacles4 = [(10.8, 6.9), (11.8, 6.9), (11.8, 6.7), (10.8, 6.7), (10.8, 6.9)]
    obstacles5 = [(12.4, 6.9), (13.4, 6.9), (13.4, 6.7), (12.4, 6.7), (12.4, 6.9)]
    obstacles6 = [(12.5, 1.1), (12.4, 1.2), (13.1, 1.9), (13.2, 1.8), (12.5, 1.1)]

    newboundary = [boundary[2] - 1, boundary[3] - 1, boundary[0] + 1, boundary[1] + 1]
    obstacles = [obstacles1, obstacles2, obstacles3, obstacles4, obstacles5, obstacles6]
    wall1 = Wall(obstacles1, newboundary)
    wall2 = Wall(obstacles2, newboundary)
    wall3 = Wall(obstacles3, newboundary)
    wall4 = Wall(obstacles4, newboundary)
    wall5 = Wall(obstacles5, newboundary)
    wall6 = Wall(obstacles6, newboundary)

    x, y = np.meshgrid(np.linspace(boundary[2] - 1, boundary[0] + 1, N),
                       np.linspace(boundary[3] - 1, boundary[1] + 1, N))
    dx = min((boundary[0] - boundary[2] + 2) / (N - 1), (boundary[1] - boundary[3] + 2) / (N - 1))
    size = x.shape[0]

    zwall = [[0.0] * size for i in range(size)]
    mask = np.zeros_like(zwall)

    for ix in range(size):
        for iy in range(size):
            posx = x[0][ix]
            posy = y[iy][0]
            if wall1.test_point_in_wall((posx, posy)) == 0 and wall2.test_point_in_wall((posx, posy)) == 0 and \
                    wall3.test_point_in_wall((posx, posy)) == 0 and wall4.test_point_in_wall((posx, posy)) == 0 and \
                    wall5.test_point_in_wall((posx, posy)) == 0 and wall6.test_point_in_wall((posx, posy)) == 0 and \
                    ix != 0 and ix != size - 1 and iy != 0 and iy != size - 1:
                zwall[ix][iy] = max(wall1.get_wall_interaction((posx, posy)), wall2.get_wall_interaction((posx, posy)),
                                    wall3.get_wall_interaction((posx, posy)), wall4.get_wall_interaction((posx, posy)),
                                    wall5.get_wall_interaction((posx, posy)), wall6.get_wall_interaction((posx, posy)),
                                    wall1.get_boundary_interaction((posx, posy)),
                                    wall2.get_boundary_interaction((posx, posy)),
                                    wall3.get_boundary_interaction((posx, posy)),
                                    wall4.get_boundary_interaction((posx, posy)),
                                    wall5.get_boundary_interaction((posx, posy)),
                                    wall6.get_boundary_interaction((posx, posy)))
            else:
                mask[ix][iy] = 1

    totalSdis = []
    for id in range(len(trajectorydata)):
        itrajectory = trajectorydata[id]
        startframenum = int(itrajectory[1])
        endframenum = int(itrajectory[2])
        selectedagentid = itrajectory[0]
        totalframes = int(endframenum - startframenum)
        newstart = [0, 0]
        target = [0, 0]
        Groundtruthx = []
        Groundtruthy = []
        StraveledPathx = []
        StraveledPathy = []
        Sdis = 0
        if totalframes == 1:
            continue
        for iframe in range(totalframes):
            z = [[0.0] * size for i in range(size)]
            z = np.array(z)
            humans = []
            startframe = startframenum * 3 + 0.2
            endframe = endframenum * 3 + 0.2
            nextframepos = [0, 0]
            traveledlen = 0
            grouplist = []

            for ind, idata in enumerate(totaldata):
                for k in range(len(idata)):
                    if startframe == idata[k][0]:
                        if selectedagentid != ind:
                            pos = vmath.Vector2(idata[k][1], idata[k][2])
                            angle = idata[k][4]
                            grouplist.append(idata[k][-1])
                            human = Human(pos, angle)
                            humans.append(human)
                        else:
                            newstart = [idata[k][1], idata[k][2]]
                            if iframe == 0:
                                Groundtruthx.append(newstart[0])
                                Groundtruthy.append(newstart[1])
                                StraveledPathx.append(newstart[0])
                                StraveledPathy.append(newstart[1])
                            nextframepos = [idata[k + 1][1], idata[k + 1][2]]
                            Groundtruthx.append(nextframepos[0])
                            Groundtruthy.append(nextframepos[1])
                            traveledlen = math.sqrt(
                                (newstart[0] - nextframepos[0]) ** 2 + (newstart[1] - nextframepos[1]) ** 2)
                    if endframe == idata[k][0] and selectedagentid == ind:
                        target = [idata[k][1], idata[k][2]]
                        break
            if iframe == 0:
                Snewstart = [newstart[0], newstart[1]]
            else:
                Snewstart = [StraveledPathx[-1], StraveledPathy[-1]]
            target = (target[0], target[1])
            groups = []
            groupslist = get_group_index(grouplist)
            for igroup in groupslist:
                if len(igroup) > 1:
                    group = [humans[ind] for ind in igroup]
                    groupinst = Group(group, stride=0.2)
                    groups.append(groupinst)

            for ix in range(size):
                for iy in range(size):
                    posx = x[0][ix]
                    posy = y[iy][0]
                    newpos = vmath.Vector2(posx, posy)
                    if mask[ix][iy] == 1:
                        z[ix, iy] = 0
                    else:
                        humanvalue = [ihuman.basic_personal_space(newpos) for ihuman in humans]
                        z[ix, iy] = max(humanvalue)
                        groupvalue = [igroup.group_interaction(newpos) for igroup in groups]
                        z[ix, iy] = max(max(humanvalue), max(groupvalue), zwall[ix][iy])

            z = np.transpose(z)

            mask = np.transpose(mask)
            maxvalue = max(max(x) for x in z)

            r = 0.1
            phi = (x - Snewstart[0]) ** 2 + (y - Snewstart[1]) ** 2 - r ** 2
            phi = np.ma.MaskedArray(phi, mask)
            speed = np.zeros_like(x)
            for ix in range(size):
                for iy in range(size):
                    speed[ix][iy] = (maxvalue - z[ix][iy]) / maxvalue * 1.4

            try:
                t = skfmm.travel_time(phi, speed, dx, order=1)
            except:
                mask = np.transpose(mask)
                print("no zero contour")
                break

            px, py = minimal_path(t, target, dx, boundary=newboundary, steps=N, N=1000)

            pxy = []
            newpxy = []
            for ind, ix in enumerate(px):
                pxy.append([ix, py[ind]])
            for ixy in pxy:
                if ixy not in newpxy:
                    newpxy.append(ixy)

            px = [ixy[0] for ixy in newpxy]
            py = [ixy[1] for ixy in newpxy]

            px.reverse()
            py.reverse()

            Straveledlen = 0

            Sdiffx = np.zeros_like(px)
            Sdiffy = np.zeros_like(py)
            Sdiffx[1:] = np.asarray(px[1:]) - np.asarray(px[:-1])
            Sdiffy[1:] = np.asarray(py[1:]) - np.asarray(py[:-1])

            Straveli = 1

            while Straveledlen < traveledlen and Straveli < len(Sdiffx):
                lens = math.sqrt((Sdiffx[Straveli]) ** 2 + (Sdiffy[Straveli]) ** 2)
                Straveledlen += lens
                Straveli += 1

            px = px[:Straveli - 1]
            py = py[:Straveli - 1]

            StraveledPathx += px[1:]
            StraveledPathy += py[1:]
            if iframe != 0 and iframe != totalframes - 1:
                if len(px) != 0:
                    Sdis += math.sqrt((nextframepos[0] - px[-1]) ** 2 + (nextframepos[1] - py[-1]) ** 2)

            z_min, z_max = -np.abs(z).max(), np.abs(z).max()
            fig, ax = plt.subplots()
            plt.axis('equal')
            c = ax.contourf(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
            # set the limits of the plot to the limits of the data
            ax.axis([x.min(), x.max(), y.min(), y.max()])
            fig.colorbar(c, ax=ax)
            for obs in obstacles:
                obstaclesx, obstaclesy = Polygon(obs).exterior.xy
                ax.plot(obstaclesx, obstaclesy, color='blue')
            plt.plot(Groundtruthx[0], Groundtruthy[0], 'oc')
            plt.plot(target[0], target[1], 'om')
            plt.plot(StraveledPathx, StraveledPathy, '-r', linewidth=2)
            plt.plot(Groundtruthx, Groundtruthy, '-g', linewidth=2)
            plt.savefig('figures/salsa' + str(dataset) + 'paths/' + str(id) + '_' + str(startframenum) + str(
                endframenum) + 'simulated.png')
            plt.close()
            startframenum += 1
            mask = np.transpose(mask)

        totalSdis.append(Sdis)

        with open(Sdissavepath, 'a', newline='') as Scsvfile:
            writer = csv.writer(Scsvfile)
            towrite = [Sdis]
            writer.writerow(towrite)
            Scsvfile.close()
    print('Social-aware mean:', statistics.mean(totalSdis), 'Social-aware SD', statistics.stdev(totalSdis))


if __name__ == '__main__':
    main()
