import numpy as np
from pythonds3.graphs import Graph,Vertex
from pythonds3.basic import Queue
from myProj.mypso import pso
from pyswarms.discrete import BinaryPSO
from matplotlib import pyplot as plt
import sys

fault_zone = 1
bcz = [[1, 2, 3, 18], [3, 4, 5], [5, 6, 7, 8], [8, 9], [9, 10, 11, 12], [12, 13, 14], [14, 15, 16, 17], [17, 18]]
bczs = [[1, -1, -1, -1], [1, -1, 1], [-1, -1, 1, -1], [1, 1], [-1, 1, -1, -1], [1, -1, 1], [-1, -1, 1, -1],
        [1, 1]]  # the breaks connect to zone value 1 or -1
# break_status=[1,1,1,1,0,1,1,0,0,1,1,1,1,0,1,1,1,1]
break_status = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

loads=np.array([[2,4,6,11,13,15],
               [2,20,2,2,20,2],
               [1,2,1,1,2,1]])   # the first row is the breaker, second row is the load capacity, 3rd row is the load priority

gens=np.array([[1,7,10,16],
               [36, 4, 34, 4]])

szone = np.zeros([8, 18])
for i in range(len(szone)):
    for j, k in zip(bcz[i], bczs[i]):
        szone[i][j - 1] = k




def buildGraph(break_status,fault_zone):
    change_breakstatus(fault_zone)
    g = Graph()
    for i in range(1, 9):
        g.set_vertex(str(i))
    a={3:[1,2],5:[2,3],8:[3,4],9:[4,5],12:[5,6],14:[6,7],17:[7,8],18:[1,8]}
    for i in a.keys():
        if break_status[i-1]==1:
            g.add_edge(str(a[i][1]),str(a[i][0]))
            g.add_edge(str(a[i][0]),str(a[i][1]))
    # g.bfs(g.get_vertex('1'))
    # g.traverse('1', '8')
    return g

def update_szone(fault_zone,bcz,szone):
    fault_zone=fault_zone-1   # to become human readable
    break_num=len(bcz[fault_zone])
    szone=np.delete(szone,fault_zone,0)
    arr = [i-1 for i in bcz[fault_zone]]
    szone=np.delete(szone,arr,1)

    break_flow=np.array([20,0,2,4,0,0,36,2,20,20,0,2,4,-1])
    zone_balance = szone.dot(break_flow)
    return zone_balance.tolist()

# change break_status
def change_breakstatus(fault_zone):
    fault_zone = fault_zone-1
    for i in bcz[fault_zone]:
        break_status[i-1] = 0
    print(break_status)
# update_szone(1,bcz,szone)

def search_nectivezone(zone_balance,fault_zone,break_status):
    a=search_zone(zone_balance,fault_zone)
    g = buildGraph(break_status,fault_zone)
    if a!=True:
        g.bfs(g.get_vertex(str(a[0])))
        g.traverse(str(a[0]),str(a[1]))
    print("balance_power:",a[2])

def search_zone(zone_balance,fault_zone):
    fault_zone = fault_zone - 1

    z = list(range(1,9))
    z = z[:fault_zone]+z[fault_zone+1:]

    minval = min(zone_balance)

    if (fault_zone > 0):
        zone_balance_w = zone_balance[fault_zone - 1::-1] + zone_balance[:fault_zone:-1]
        z = z[fault_zone - 1::-1] + z[:fault_zone:-1]
    else:
        zone_balance_w = zone_balance
    index = zone_balance_w.index(minval)
    sum = 0
    sums={}
    for i in range(index,len(zone_balance_w)):
        sum += zone_balance_w[i]
        sums[sum] = i
    sum=0
    for i in range(index-1,-1,-1):
        sum += zone_balance_w[i]
        sums[sum] = i

    max = minval
    for i in sums.keys():
        if i >0:
            return True
        if i>max:
            max = i
    return  z[index],z[sums[max]],max


##pso
def load_shedding():
    lb = np.zeros(len(brk_load))
    ub = np.ones(len(brk_load))
    xopt, fopt = pso(fun1, lb, ub, ieqcons=[constraints])
    break_change = []
    print(xopt, old_brk_status)
    for i in range(len(xopt)):
        if xopt[i] != old_brk_status[i]:
            break_change.append(brk_load[i])
    return break_change,fopt

def test_pso():
    #before running
    # old_brk_sttus=[break_status[i - 1] for i in loads[0]]
    args=(1,0)
    lb = np.zeros(len(brk_load))
    ub = np.ones(len(brk_load))
    xopt, fopt = pso(fun2, lb, ub, ieqcons=[constraints],args=args)
    break_change=[]
    print(xopt,old_brk_status)
    for i in range(len(xopt)):
        if xopt[i]!=old_brk_status[i]:
            break_change.append(brk_load[i])
    print("the break {} should  open \n".format(break_change),"the total load power is:",fopt)

def cal_pgen():
    '''

    :return: the the generator power: pgen,
    brk_load: the break of the load connected
    load_in: the loads that the system has
    pl: priority * load
    '''
    brk_gen = np.array([break_status[i - 1] for i in gens[0]])
    pgen=brk_gen.dot(gens[1])
    brk_load =[]
    load_in=[]
    old_brk_status=[]
    pl=[]
    for i,j,k in zip(loads[0],loads[1],loads[2]):
        if break_status[i-1]==1:
            brk_load.append(i)
            load_in.append(j)
            old_brk_status.append(break_status[i-1])
            pl.append(j*k)

    return pgen,np.array(brk_load),np.array(load_in),old_brk_status,np.array(pl)

def fun1(x):
    x=np.array(x)
    return x.dot(load_in.T)
def fun2(x,*args):
    x=np.array(x)
    w1,w2=args
    return w1*x.dot(load_in.T)+w2*x.dot(pl.T)

def constraints(x,*args):
    # x=np.array(x)
    # gen_cap = gens[1]
    # p_gen = x.dot(gen_cap.T)
    w1,w2=args
    return [pgen-fun1(x)]





def function():
    # break_flow=np.array([23,2,20,20,0,2,2,0,0,22,2,20,20,0,2,1,-1,1])
    zone_balance=update_szone(fault_zone,bcz,szone)
    search_nectivezone(zone_balance,fault_zone,break_status)



################# run ##########
function()
################################
pgen,brk_load,load_in,old_brk_status,pl=cal_pgen()
test_pso()
