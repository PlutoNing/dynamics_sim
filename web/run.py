
import matplotlib.pyplot as plt
from pylab import *
import random as randd
# plt.rcParams['axes.unicode_minus']=False  #用于解决不能显示负号的问题
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure(figsize=(7,7))
#各博弈主体动态复制方程
def f(x,y):
    return x*(1-x)*(5*y-2.5)
def g(x,y):
    return y*(1-y)*(4*x-2)

def calculateValue(initX, initY, dt, epoch):
    x = []
    y = []
    
    #演化初始赋值
    x.append(initX)
    y.append(initY)
    
    #微量计算及append
    for index in range(epoch):
        tempx = x[-1] + (f(x[-1],y[-1])) * dt
        tempy = y[-1] + (g(x[-1],y[-1])) * dt
 
        x.append(tempx)
        y.append(tempy)
    return (x, y)

def main():
    print("Hello, world!")
    D=[]
#随机生成200个初始点
    for index in range(200):
        random_a = random.uniform(0, 1)
        random_b = random.uniform(0, 1)
        #步长dt为0.001 迭代次数1000
        d = calculateValue(random_a, random_b, 0.001, 1000)
        D.append(d)
    
    
    for n,m in D:
        plt.plot(n,m)
    
    
    plt.ylabel("$y$",fontsize=25)
    plt.xlabel("$x$",fontsize=25)  
    plt.tick_params(labelsize=25)
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.grid(linestyle=":",color="b",linewidth=1)
    plt.savefig("test",dpi=300,bbox_inches ="tight")

if __name__ == "__main__":
    print(randd.random(),randd.random())
    print("Hello, world!")
    
    D=[]
#随机生成200个初始点
    for index in range(200):
        print(type(randd))
        rand1 = randd.random()
        rand2 = randd.random()
        #步长dt为0.001 迭代次数1000
        d = calculateValue(rand1, rand2, 0.001, 1000)
        D.append(d)
    
    
    for n,m in D:
        plt.plot(n,m)
    
    
    plt.ylabel("$y$",fontsize=25)
    plt.xlabel("$x$",fontsize=25)  
    plt.tick_params(labelsize=25)
    plt.xticks([0,0.2,0.4,0.6,0.8,1])
    plt.grid(linestyle=":",color="b",linewidth=1)
    plt.savefig("test",dpi=300,bbox_inches ="tight")