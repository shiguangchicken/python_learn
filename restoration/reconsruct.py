from myProj.mypso import pso
import numpy as np
# from pythonds3.graphs import Graph,Vertex
from myProj.Myadjacency_graph import Graph
class Recons:
    '''
    bcz:节点所连接的开关
    break_status:开关状态
    loads:负载矩阵，第一行为开关序号，第二行为负载容量，第三行为权重
    gens:发电机矩阵
    load_brk:负载开关状态，第一行为负载开关序号，第二行为开关状态
    gen_brk:发电机开关状态，第一行为负载开关序号，第二行为开关状态

    '''


    def _getloadbreak(self):
        '''

        :return: 返回负载开关序号及状态矩阵
        '''
        brk=[]
        s=[]
        for i in self.loads[0]:
           i=int(i)
           brk.append(i)
           s.append(break_status[1][i-1])
        return np.array([brk,s])
    def _getgenbreak(self):
        '''
        :return: 返回发电机开关序号及状态矩阵
        '''
        brk=[]
        s=[]
        for i in self.gens[0]:
           i=int(i)
           brk.append(i)
           s.append(break_status[1][i-1])
        return np.array([brk,s])

    def __brkstatus_init(self):
        '''初始化开关状态，4,12,22未连接'''
        # a=[4,12,22];
        a=[2,22,19,23];
        for i in a:
            self.break_status[1][i-1]=0

    def __init__(self,bcz,break_status,loads,gens,bus_brk,cons):
        self.break_status=break_status
        self.bcz=bcz
        self.loads=loads
        self.gens=gens
        self.__brkstatus_init()
        self.load_brk=self._getloadbreak()
        self.gen_brk=self._getgenbreak()
        self.fault_zone=None
        self.cons=cons
        self.bus_brk = bus_brk

    def set_fault(self,fault_zone):
        '''
        :param fault_zone: 短路节点，改变相应节点所连接的开关状态 ，输入：bus1  bus2   bus3.......
        :return:
        '''
        self.fault_zone=fault_zone
        for i in bcz[fault_zone]:
            self.break_status[1][i-1]=0
        self.gen_brk=self._getgenbreak()
        self.load_brk=self._getloadbreak()
        # self._update_con()

    def cal_pgen(self):
        '''
        :return: the the generator power: pgen,
        brk_load: the break of the load connected
        load_in: the loads that the system has
        pl: priority * load
        '''
        # brk_gen = np.array([break_status[1][i - 1] for i in gens[0]])
        # pgen = brk_gen.dot(gens[1])
        pgen = self.gen_brk[1].dot(gens[1].T)
        brk_load = []
        d=[]
        for i in bcz[self.fault_zone]:
            for j in range(len(loads[0])):
                if i==loads[0][j]:
                    d.append(j)
                    break
        load_in=np.delete(loads,d,1)
        return pgen,load_in



    def binary_pso(self,iw1,iw2):
        '''
        binary pso 优化负载，优化哪些负载应该切除
        :return:
        '''

        def fun2( x, *args):
            '''
            目标函数
            :param args:传递参数，w1 w2
            :return:
            '''
            x = np.array(x)
            w1, w2 = args
            pl=[]
            for i in range(len(load_in[0])):
                pl.append(load_in[1][i]*load_in[2][i])
            pl=np.array(pl)
            return w1 * x.dot(load_in[1].T) + w2 * x.dot(pl.T)

        def constraints(x, *args):
            # x=np.array(x)
            # gen_cap = gens[1]
            # p_gen = x.dot(gen_cap.T)
            w1, w2 = args
            return [pgen - x.dot(load_in[1].T)]
        #计算pgen
        pgen,load_in=self.cal_pgen()

        args = (iw1, iw2)
        lb = np.zeros(len(load_in[0]))
        ub = np.ones(len(lb))
        xopt, fopt = pso(fun2, lb, ub, ieqcons=[constraints], args=args)
        print(xopt,fopt)


    def build_graph(self,bus_brk,cons):
        '''
        建立电站的拓扑图
        :param bus_brk:
        :param cons:
        :return:
        '''
        g = Graph()
        for i in bcz.keys():
            if i != self.fault_zone:
                g.set_vertex(str(i))
        for i,j in zip(bus_brk,cons):
            if (len(i)==1) and (self.break_status[1][i[0]-1]==1):
                g.add_edge(j[0],j[1])
                g.add_edge(j[1],j[0])
            if len(i)==3:
                a=[break_status[1][i[0]-1],break_status[1][i[1]-1],break_status[1][i[2]-1]]
                if a==[1,1,1]:
                    g.add_edge(j[0], j[1])
                    g.add_edge(j[1], j[0])
                    g.add_edge(j[0], j[2])
                    g.add_edge(j[2], j[0])
                else:
                    if [a[0],a[1]] == [1,1]:
                        g.add_edge(j[0], j[1])
                        g.add_edge(j[1], j[0])
                    if [a[0],a[2]] == [1,1]:
                        g.add_edge(j[0], j[2])
                        g.add_edge(j[2], j[0])
                    if [a[1],a[2]] == [1,1]:
                        g.add_edge(j[1], j[2])
                        g.add_edge(j[2], j[1])
        return g

    def serch_path(self):
        '''
        改变除故障处外未接入的开关状态
        :param g:
        :return:
        '''
        start='bus1' #从哪一个节点bfs
        if start==self.fault_zone:
            start=='bus2'
        for i in self.bcz.keys():
            if i!=self.fault_zone:
                for j in self.bcz[self.fault_zone]:
                    if j in self.bcz[i]:
                        bcz[i].remove(j)
        del self.bcz[self.fault_zone]
        brk_changes=[]
        cc =set() #未接入的开关序号
        ##找到未接入的开关编号
        for bus in self.bcz.values():
            for i in bus:
                if self.break_status[1][i-1]==0:
                    cc.add(i)
        for i in range(1,2**len(cc)):
            bi = format(i,'b')
            cc_status=[]
            for j in range(len(bi)):
                cc_status.append(int(bi[j]))
            for i1 in range(len(cc)-len(cc_status)):
                cc_status.insert(0,0)
            #改变break_status
            for i1,j1 in zip(cc,cc_status):
                self.break_status[1][i1-1]=j1
            #build graph
            g=self.build_graph(self.bus_brk,self.cons)
            g.bfs(g.get_vertex(start))
            if g.is_allcon() and g.is_radial():
                brk_changes.append([list(cc),cc_status])
        return brk_changes






    def _update_con(self):
        '''
        设置故障之后，更新con字典
        :return:
        '''
        for i in self.con[self.fault_zone]:
            self.con[i].remove(self.fault_zone)
        del self.con[self.fault_zone]





bcz = {'bus1': [1, 2, 3, 4], 'bus2': [8, 10, 11, 12], 'bus3': [18, 20, 21, 22], 'bus4': [27, 28, 29, 30],
           'bus5': [2, 5, 7],
           'bus6': [4, 6, 9], 'bus7': [14, 15, 17], 'bus8': [13, 16, 19], 'bus9': [24, 25, 27], 'bus10': [23, 26, 29]}
break_status=np.array([np.arange(1,31),
                          np.ones(30)])
# loads=np.array([[3,5,6,11,15,16,21,25,26,30],
#                    [18,5,2,13,11,7,14,12,1,17],
#                    [3,1,3,2,2,1,3,1,2,3]])
loads=np.array([[3,5,6,11,15,16,21,25,26,30],
                   [18.5,6.5,2,13,10.5,7,12.5,9.5,3.5,17],
                   [3,1,3,2,2,1,3,1,2,3]])
gens=np.array([[1,8,18,28],
                   [25,25,25,25]])

# con = {'bus1': ['bus5', 'bus6'], 'bus2': ['bus5', 'bus6'], 'bus3': ['bus7', 'bus8'], 'bus4': ['bus9', 'bus10'],
#      'bus5': ['bus1', 'bus2', 'bus7'],
#      'bus6': ['bus1', 'bus2', 'bus8'], 'bus7': ['bus3', 'bus5', 'bus9'], 'bus8': ['bus6', 'bus3', 'bus10'],
#      'bus9': ['bus7', 'bus4'], 'bus10': ['bus4', 'bus8'], }
bus_brk=[[2],[7,10,14],[17,20,24],[27],[29],[19,22,23],[9,12,13],[4]]
cons=[['bus1','bus5'],
      ['bus5','bus2','bus7'],
      ['bus7','bus3','bus9'],
      ['bus9','bus4'],
      ['bus4','bus10'],
      ['bus8','bus3','bus10'],
      ['bus6','bus2','bus8'],
      ['bus1','bus6']
      ]

testrecon= Recons(bcz,break_status,loads,gens,bus_brk,cons)
testrecon.set_fault('bus10')
# print(testrecon.break_status)
# g=testrecon.build_graph(bus_brk,cons)
# g.bfs(g.get_vertex('bus2'))

print(testrecon.serch_path())
# testrecon.binary_pso(iw1=0.1,iw2=0.9)
