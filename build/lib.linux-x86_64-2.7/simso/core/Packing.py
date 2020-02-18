"""
This module is part of the RUN implementation (see RUN.py).
"""

from simso.schedulers.RUNServer import EDFServer, TaskServer, DualServer, \
    select_jobs, add_job, get_child_tasks, update_deadline2
import traceback

INFINITO = 9000000000000

class Packing(object):
	def __init__(self, arg):
		super(Packing, self).__init__()
		self.arg = arg

def who_am_i():
   stack = traceback.extract_stack()
   filename, codeline, funcName, text = stack[-2]

   return funcName

def pack_BFD(servers):
    """
    Best-Fit with servers inversely sorted by their utilization.
    """
    return pack_BF(sorted(servers, key=lambda x: x.utilization, reverse=True))

def pack_WFD(servers):
    """
    Worst-Fit with servers inversely sorted by their utilization.
    """
    return pack_WF(sorted(servers, key=lambda x: x.utilization, reverse=True))

def pack_FFD(servers):
    """
    First-Fit with servers inversely sorted by their utilization.
    """
    return pack_FF(sorted(servers, key=lambda x: x.utilization, reverse=True))


def pack_FF(servers):
    """
    Create a list of EDF Servers by packing servers. First-Fit  
    packing algorithm.
    """
    packed_servers = []
    for server in servers:
        for p_server in packed_servers:
            if p_server.utilization + server.utilization <= 1:
                p_server.add_child(server)
                break
        else:
            p_server = EDFServer()
            p_server.add_child(server)
            packed_servers.append(p_server)

    return packed_servers

def pack_WF(servers):
    """
    Create a list of EDF Servers by packing servers. Worst-Fit  
    packing algorithm.
    """

    # Find packed servers if there is one (EDFServer)
    packed_servers = []#[s for s in servers if s is EDFServer()]
    for server in servers:
        #Try to place the item in the least full bin that will accommodate it, i.e., the one that will leave the most space remaining
        packed_servers.sort(key=lambda x: x.utilization, reverse=False)
        for p_server in packed_servers:
            if p_server.utilization + server.utilization <= 1:
                p_server.add_child(server)
                break
        else:
            p_server = EDFServer()
            p_server.add_child(server)
            packed_servers.append(p_server)
    
    return packed_servers

def pack_BF(servers):
    """
    Create a list of EDF Servers by packing servers. Best-Fit 
    packing algorithm.
    """

    # Find packed servers if there is one (EDFServer)
    packed_servers = []#[s for s in servers if s is EDFServer()]
    for server in servers:
        #Try to place the item in the fullest bin that will accommodate it, i.e., the one that will leave the least space remaining
        packed_servers.sort(key=lambda x: x.utilization, reverse=True)
        for p_server in packed_servers:
            if p_server.utilization + server.utilization <= 1:
                p_server.add_child(server)
                break
        else:
            p_server = EDFServer()
            p_server.add_child(server)
            packed_servers.append(p_server)

    return packed_servers


def decrease(self, servers, length, dummy):
    
    servers.sort(key=lambda x: x.utilization, reverse=True)

    from collections import namedtuple
    # pylint: disable-msg=C0103
    IdleTask = namedtuple('IdleTask', ['utilization', 'deadline', 'name'])

    idle = length - sum([s.utilization for s in servers])

    packed_servers = []
    for server in servers:
        #Try to place the item in the fullest bin that will accommodate it, i.e., the one that will leave the least space remaining
        packed_servers.sort(key=lambda x: x.utilization, reverse=True)
        for p_server in packed_servers:
            if p_server.utilization + server.utilization <= 1:
                p_server.add_child(server)

                # Add dummy to complete server utilization
                if p_server.utilization < 1 and idle > 0: 
                	d = IdleTask(min(1 - p_server.utilization, idle), 0, 'IdleTask')
                	t = TaskServer(d)
                	update_deadline2(self.sim, t, INFINITO) 
                	self.Tdummy.append(t)
                	p_server.add_child(t)
                	idle -= d.utilization
                break
        else:
            p_server = EDFServer()
            p_server.add_child(server)
            packed_servers.append(p_server)

    return packed_servers

def increase(self, servers, length, dummy):
    
    servers.sort(key=lambda x: x.utilization, reverse=False)

    from collections import namedtuple
    # pylint: disable-msg=C0103
    IdleTask = namedtuple('IdleTask', ['utilization', 'deadline', 'name'])

    idle = length - sum([s.utilization for s in servers])

    packed_servers = []
    for server in servers:
        packed_servers.sort(key=lambda x: x.utilization, reverse=False)
        for p_server in packed_servers:
            if p_server.utilization + server.utilization <= 1:
                p_server.add_child(server)

                # Add dummy to complete server utilization
                if p_server.utilization < 1 and idle > 0: 
                	d = IdleTask(min(1 - p_server.utilization, idle), 0, 'IdleTask')
                	t = TaskServer(d)
                	update_deadline2(self.sim, t, INFINITO) 
                	self.Tdummy.append(t)
                	p_server.add_child(t)
                	idle -= d.utilization
                break
        else:
            p_server = EDFServer()
            p_server.add_child(server)
            packed_servers.append(p_server)

    return packed_servers

