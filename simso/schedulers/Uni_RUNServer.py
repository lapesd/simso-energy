"""
This module is part of the RUN implementation (see RUN.py).
"""

from fractions import Fraction
from math import ceil, floor

INFINITO = 9000000000000

class _Server(object):
    """
    Abstract class that represents a Server.
    """
    next_id = 1

    def __init__(self, is_dual, task=None):
        self.parent = None
        self.is_dual = is_dual
        self.utilization = Fraction(0, 1)
        self.task = task
        self.job = None
        self.deadlines = [0]
        self.all_deadlines = [0]
        self.budget = 0
        self.next_deadline = 0
        self.last_release = 0
        self.last_deadline = 0

        self.dummyServer = False
        self.identifier = _Server.next_id
        _Server.next_id += 1
        if task:
            if hasattr(task, 'utilization'):
                self.utilization += task.utilization
            else:
                self.utilization += Fraction(task.wcet) / Fraction(task.period)

    def add_deadline(self, current_instant, deadline):
        """
        Add a deadline to this server.
        """
        if self.last_deadline != self.next_deadline:
            self.last_deadline = self.next_deadline
        self.deadlines.append(deadline)
        self.all_deadlines.append(deadline)

        self.deadlines = [d for d in self.deadlines if d > current_instant]
        self.next_deadline = min(self.deadlines)

    def create_job(self, sim, current_instant):
        """
        Replenish the budget.
        """
        self.budget = int(self.utilization * (self.next_deadline - current_instant))

class TaskServer(_Server):
    """
    A Task Server is a Server that contains a real Task.
    """
    def __init__(self, task):
        super(TaskServer, self).__init__(False, task)
        self.last_cpu = None


class EDFServer(_Server):
    """
    An EDF Server is a Server with multiple children scheduled with EDF.
    """
    def __init__(self):
        super(EDFServer, self).__init__(False)
        self.children = []

    def add_child(self, server):
        """
        Add a child to this EDFServer (used by the packing function).
        """
        self.children.append(server)
        self.utilization += server.utilization
        server.parent = self


class DualServer(_Server):
    """
    A Dual server is the opposite of its child.
    """
    def __init__(self, child):
        super(DualServer, self).__init__(True)
        self.child = child
        child.parent = self
        self.utilization = 1 - child.utilization


def add_job(sim, job, server):
    """
    Recursively update the deadlines of the parents of server.
    """
    server.job = job
    while server:
        server.add_deadline(sim.now(), job.absolute_deadline *
                            sim.cycles_per_ms)
        server.create_job(sim, sim.now())
        server.last_release = sim.now()
        server = server.parent

def update_deadline(sim, server, deadline):

    last = None
    while server:
        server.deadlines = []
        server.next_deadline = deadline*sim.cycles_per_ms
        server.create_job(sim, sim.now())
        #sim.logger.log("{}.budget = {}, deadline {}".format(server.identifier, server.budget, server.next_deadline))
        server.dummyServer = True
        last = server
        server = server.parent

def turnoff(root, server):
    while server != root:
        server.deadlines = []
        server.next_deadline = 0
        server.budget = 0
        server = server.parent
def update_deadline2(sim, root, server, deadline):

    last = None
    while server != root:
        server.deadlines = []
        server.next_deadline = deadline*sim.cycles_per_ms
        server.budget = INFINITO
        #sim.logger.log("{}.budget = {}, deadline {}".format(server.identifier, server.budget, server.next_deadline))
        server.dummyServer = True
        last = server
        server = server.parent

# next release
def release(self, server, t):
    if server.next_deadline > t:
        return server.last_release
    return server.next_deadline
    

# next deadline
def deadline(self, server, t):

    if server.next_deadline > t:
        return server.next_deadline

    else:
        return server.next_deadline + (server.task.period * self.sim.cycles_per_ms)

# budget on activation
def budget(self, server, t):
    budget = 0

    if release(self, server, t) >= t:
        budget = int(server.utilization * (deadline(self, server, t) - release(self, server,t)))

    return budget

def uni_algorithm_1(self, servers, t):
    #self.sim.logger.log("algorithm_1".format())

    #servers = update_times(self, servers)

    servers.sort(key=lambda s: deadline(self, s, t), reverse=False)
    e = 1
    
    """for s in servers:
        self.sim.logger.log("id={}, release {}, next_deadline {}, budget = {}".format(s.identifier, release(self, s, t),deadline(self, s, t), budget(self, s, t)))
    """

    delta = deadline(self, servers[len(servers)-1], t) - t 
    #delta = servers[len(servers)-1].next_deadline - t 
    
    #self.sim.logger.log("start delta = {}".format(delta))

    for j in range(0, len(servers)):
        
        d = deadline(self, servers[j], t)
        c = 0
        #self.sim.logger.log("j = {}, d = {}, t = {}".format(servers[j].identifier, d, t))
        #self.sim.logger.log("---------".format())
        for i in range(0, j+1):

            ri = D(self, servers[i], t)
            di = R(self, servers[i], d)
            #self.sim.logger.log("i = {}, d = {}, D = {}, R = {}".format(servers[i].identifier, d, ri, di))
            ci=0
            if ri > t:
                ci = budget(self, servers[i], t) 
            c = c + ci + (di - ri)*(servers[i].utilization)
            #self.sim.logger.log("c = {}, ci = {}".format(c, ci))
            #self.sim.logger.log("---------".format())

        delta = min(delta, (d - t - c))
        #self.sim.logger.log("delta = {}, d-t-c = {}".format(delta, (d - t - c)))
        if delta <= 0:
            return 0

    return delta

# get deadline on time
def D(self, server, time):

    if time < server.next_deadline:

        next_d = server.next_deadline
        last_r = server.last_release
        period = server.next_deadline - server.last_release

        while time < last_r:
            next_d = last_r
            last_r -= period

        return next_d
    else:
        next_d = deadline(self, server, time)
        period = deadline(self, server, time)-release(self, server, time)
        last_r = server.next_deadline
        
        while time > next_d:
            last_r = next_d
            next_d += period

        return next_d
   
    """
    if server.budget > 0:
        if server.last_release == t:
            return t
        else:
            return server.next_deadline
    else:
        return max([d for d in server.all_deadlines])
    """

# get release on time d 
def R(self, server, d):

    if d < server.next_deadline:
        return server.last_release
    elif d == server.next_deadline:
        return d
    else:
        if d == deadline(self, server, d):
            return d
        elif d > deadline(self, server, d):
            next_d = deadline(self, server, d)
            period = deadline(self, server, d)-release(self, server, d)
            last_r = server.next_deadline
            
            while d > next_d:
                last_r = next_d
                next_d += period

            if d == next_d:
                return d
            else:
                return last_r
        else:
            return server.next_deadline

def algorithm_2(self, servers, t):

    #self.sim.logger.log("deadlines: {}".format([s.next_deadline for s in servers]))

    # Get server with min deadline
    #D_t = deadline(self,(min(servers, key=lambda s: deadline(self, s))))

    D_t = min([D(self, s, t) for s in servers]) 


    #self.sim.logger.log("D_t: {}, t: {}".format(D_t, t))

    # Get min deadline at instant t_0
    D_t0 = min([D(self, s, self.t0) for s in servers]) 

    #self.sim.logger.log("D_t0: {}, t_0: {}".format(D_t0, self.t0))
    
    
    if D_t == D_t0:
        return -1
    
    U = sum([s.utilization for s in servers])

    #self.sim.logger.log("U: {}".format(float(U)))

    

    delta = (D_t - t)*(1-U) #+ (D_t0-t)
    self.t0 = t

    #self.sim.logger.log("delta: {}, D_t {}".format(delta, D_t))

    acumulated = 0
    for s in servers:
        # find active jobs on time t
        if s.budget > 0:
            acumulated += s.budget
        # find next activation
        elif s.next_deadline < D_t:
            #self.sim.logger.log("+{}, d {}".format(budget(self, s), s.next_deadline))
            acumulated += budget(self, s)

    acumulated = (D_t - t) - acumulated

    #self.sim.logger.log("acumulated: {}".format(acumulated))

    if acumulated > delta:
        delta = acumulated


    return delta #+ (self.root.next_deadline-t)

def get_child_tasks(server):
    """
    Get the tasks scheduled by this server.
    """
    if server.task:
        return [server]
    else:
        if server.is_dual:
            return get_child_tasks(server.child)
        else:
            tasks = []
            for child in server.children:
                tasks += get_child_tasks(child)
            return tasks

def release_dummy_job(name, deadline):
    from collections import namedtuple
    # pylint: disable-msg=C0103
    IdleTask = namedtuple('IdleTask', ['utilization', 'deadline', 'name'])


    task = IdleTask(0, deadline, name)
    t = TaskServer(task)

    return t

def select_jobs(self, server, virtual, execute=True):
    """
    Select the jobs that should run according to RUN. The virtual jobs are
    appended to the virtual list passed as argument.
    """
    jobs = []
    if execute:
        virtual.append(server)

    # Leafs
    if server.task:
        if execute and server.budget > 0:

            # select a dummy job
            if server.job is None:
                #self.sim.logger.log("add dummy".format())
                jobs.append(server)

            #select a real job
            elif server.job.is_active():
                #self.sim.logger.log("add job {}".format(server.task.name))
                jobs.append(server)

            # real job don't use your WCET
            else:
                #self.sim.logger.log("add wcet".format())
                pass
                #jobs.append(release_dummy_job("IdleTask", 0))
                """

                # get activated brothers            
                active_jobs = [s for s in server.parent.children if s.budget > 0 and not s.dummyServer and s.job.is_active() and s != server ]

                # take the earliest job
                if active_jobs:
                    jobs.append(min(active_jobs, key=lambda s: s.next_deadline))
                    #self.sim.logger.log("keep scheduling, next job".format())

                # no jobs, no way out. Release a dummy job to carry on the idle period
                else:
                    jobs.append(release_dummy_job("wcet", server.parent.next_deadline - self.sim.now()))
                    #self.sim.logger.log("no jobs, releasing dummy".format())
                """
    else:
        # Rule 2
        if server.is_dual:
            #if execute:
            #self.sim.logger.log("DualServer{}: BUDGET = {}, deadline = {}".format(server.identifier, server.budget, server.next_deadline))
            
            jobs += select_jobs(self, server.child, virtual, not execute)
        # Rule 1
        else:
            active_servers = [s for s in server.children if s.budget > 0]
            if active_servers:
                min_server = min(active_servers, key=lambda s: s.next_deadline)

                if self.keep:
                    min_server = [s for s in active_servers if s.dummyServer][0]
                    
                if min_server.dummyServer:
                    if not self.is_idle:
                        self.start_idle = self.sim.now()
                    self.is_idle = True
                    #self.sim.logger.log("set idle".format())
                    
            else:
                min_server = None
            
            for child in server.children:
                jobs += select_jobs(self, child, virtual,
                                    execute and child is min_server)
    return jobs


def next_activation(server):


    while server.parent.dummyServer:
       server = server.parent

    return server.parent.next_deadline
