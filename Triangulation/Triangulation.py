from sympy import solve, Symbol
import matplotlib.pyplot as plt
import numpy as np
import os


def circle(region=[4, 5, 2], color='red', alpha=0.5):
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / 1000)
    x = np.cos(theta) + region[0]
    y = np.sin(theta) + region[1]
    # v = np.linspace(0, region[2], 100)
    # v.shape = (100, 1)
    # x = v * x
    # y = v * y
    # figure = plt.figure(figsize=[10, 10])
    plt.plot(x, y, color=color, alpha=alpha)


def location(region1=[0, 1, 1],
             region2=[1, 0, 1],
             region3=[0, 0, 2 ** 0.5],
             savepath=None):
    # region=[x,y,distence]
    x = Symbol('x')
    y = Symbol('y')
    xy = solve([(region1[0] - x) ** 2 + (region1[1] - y) ** 2 - region1[2] ** 2,
                (region2[0] - x) ** 2 + (region2[1] - y) ** 2 - region2[2] ** 2,
                (region3[0] - x) ** 2 + (region3[1] - y) ** 2 - region3[2] ** 2])
    xy = xy[0]
    x_0, y_0 = round(xy[x], 2), round(xy[y], 2)
    figure = plt.figure(figsize=[10, 10])
    # 画圆
    circle(region=region1, color='red', alpha=0.5)
    circle(region=region2, color='green', alpha=0.5)
    circle(region=region3, color='blue', alpha=0.5)
    # 画圆心
    plt.scatter(x=region1[0], y=region1[1], color='red')
    plt.scatter(x=region2[0], y=region2[1], color='green')
    plt.scatter(x=region3[0], y=region3[1], color='blue')
    plt.scatter(x=x_0, y=y_0, color='black')
    # 画连线
    plt.plot([x_0, region1[0]], [y_0, region1[1]], linestyle='--')
    plt.plot([x_0, region2[0]], [y_0, region2[1]], linestyle='--')
    plt.plot([x_0, region3[0]], [y_0, region3[1]], linestyle='--')

    # 距离坐标
    plt.text(0.5 * (x_0 + region1[0]), 0.5 * (y_0 + region1[1]), str(round(region1[2], 2)))
    plt.text(0.5 * (x_0 + region2[0]), 0.5 * (y_0 + region2[1]), str(round(region2[2], 2)))
    plt.text(0.5 * (x_0 + region3[0]), 0.5 * (y_0 + region3[1]), str(round(region3[2], 2)))

    # 圆心坐标
    plt.text(x=region1[0], y=region1[1], color='red', s=str(region1[0:2]))
    plt.text(x=region2[0], y=region2[1], color='green', s=str(region2[0:2]))
    plt.text(x=region3[0], y=region3[1], color='blue', s=str(region3[0:2]))
    plt.text(x=x_0, y=y_0, color='black', s='locat:' + str([x_0, y_0]))
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    return x_0, y_0


if __name__ == '__main__':
    location(region1=[0, 1, 1],
             region2=[1, 0, 1],
             region3=[0, 0, 2 ** 0.5],
             savepath=os.path.dirname(__file__)+'/example.png')
