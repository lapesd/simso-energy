"""
Implementation of the RUN scheduler as introduced in RUN: Optimal
Multiprocessor Real-Time Scheduling via Reduction to Uniprocessor
by Regnier et al.

RUN is a global multiprocessors scheduler for periodic-preemptive-independent
tasks with implicit deadlines.
"""
from fractions import Fraction
from math import ceil
from simso.core import Scheduler, Timer
from simso.schedulers.Multi_RUNServer import EDFServer, TaskServer, DualServer, \
    select_jobs, add_job, get_child_tasks, update_deadline, \
    delta_t, omega_t, release_dummy_job, \
    next_activation
from simso.schedulers import scheduler
from simso.core.Packing import pack_BFD, pack_WFD, pack_FFD, \
    pack_BF, pack_FF, pack_WF

INFINITO = 9000000000000


@scheduler("simso.schedulers.teste")
class teste(Scheduler):
    """
    RUN scheduler. The offline part is done here but the online part is mainly
    done in the SubSystem objects. The RUN object is a proxy for the
    sub-systems.
    """

    def init(self):
        """
        Initialization of the scheduler. This function is called when the
        system is ready to run.
        """
        self.subsystems = []  # List of all the sub-systems.
        self.available_cpus = self.processors[:]  # Not yet affected cpus.
        self.task_to_subsystem = {}  # map: Task -> SubSystem
        self.Tdummy = []

        # Create the Task Servers. Those are the leaves of the reduction tree.
        list_servers = [TaskServer(task) for task in self.task_list]

        # map: Task -> TaskServer. Used to quickly find the servers to update.
        self.servers = dict(zip(self.task_list, list_servers))

        assert (sum([s.utilization for s in list_servers])
                <= len(self.processors)), "Load exceeds 100%!"

        # Instanciate the reduction tree and the various sub-systems.
        self.reduce_iterations(list_servers)

    def fill(self, servers, dummy):
        """
        Create IdleTasks in order to reach 100% system utilization.
        """
        from collections import namedtuple
        # pylint: disable-msg=C0103
        IdleTask = namedtuple('IdleTask', ['utilization', 'deadline', 'name'])

        idle = len(self.processors) - sum([s.utilization for s in servers])
        for server in servers:
            if server.utilization < 1 and idle > 0:
                task = IdleTask(min(1 - server.utilization, idle), 0, 'IdleTask')
                t = TaskServer(task)
                self.Tdummy.append(t)
                dummy.append(t)
                server.add_child(t)
                idle -= task.utilization
        while idle > 0:
            task = IdleTask(min(1, idle), 0, 'IdleTask')
            t = TaskServer(task)
            #self.Tdummy.append(t)
            #dummy.append(t)
            server = EDFServer()
            server.add_child(TaskServer(task))
            idle -= task.utilization
            servers.append(server)

    def independent(self, servers, dummy):
        """
        Add dummy sem misturar
        """
        from collections import namedtuple
        # pylint: disable-msg=C0103
        IdleTask = namedtuple('IdleTask', ['utilization', 'deadline', 'name'])

        idle = len(self.processors) - sum([s.utilization for s in servers])

        while idle > 0:
            task = IdleTask(min(1, idle), 0, 'IdleTask')
            server = EDFServer()
            server.dummyServer = True
            
            t = TaskServer(task)
            self.Tdummy.append(t)
            t.dummyServer = True
           
            server.add_child(t)
            
            servers.append(server)
            dummy.append(t)
        
            idle -= min(1, idle)

    def add_subsystem(self, server, level):
        """
        Create a proper sub-system from a unit server.
        """
        tasks_servers = get_child_tasks(server)
        subsystem_utilization = sum([t.utilization for t in tasks_servers])
        #u = sum([float(t.utilization) for t in tasks_servers if t.task.name != "IdleTask"])
        cpus = []

        while subsystem_utilization > 0:
            cpus.append(self.available_cpus.pop())
            subsystem_utilization -= 1

        subsystem = ProperSubsystem(self.sim, server, cpus, level)

        for cpu in self.available_cpus:
            if not cpu in cpus:
                cpu.set_idle(self.sim.duration)

        for server in tasks_servers:
            self.task_to_subsystem[server.task] = subsystem
        self.subsystems.append(subsystem)

    def reduce_iterations(self, servers):
        """
        Offline part of the RUN Scheduler. Create the reduction tree.
        """
        dummy = []
        #self.print_taks_in_servers(servers)
      
        servers = pack_BFD(servers)
        #self.print_servers(servers)
    
        self.independent(servers, dummy)
        #self.print_servers(servers)
        
        level = 1
        while len(servers) > 1:
            s = dual(servers)
            #self.print_dual(s)
            servers = pack_BFD(s)
            level += 1

        self.sim.logger.log("Levels = {}".format(level))
        self.add_subsystem(servers[0], level)


        for d in dummy:
            #set_budget(d)
            update_deadline(self.sim, servers[0], d)          

    def print_taks_in_servers(self, servers):
        self.sim.logger.log("------------".format())
        for s in servers:
            self.sim.logger.log("Server {} : {} utilization".format(s.identifier, (s.utilization)))
            self.sim.logger.log("{}, C = {}, P = {} ".format(s.task.name, s.task.wcet, s.task.period))
    def print_servers(self, servers):
        self.sim.logger.log("PACK------------".format())
        for server in servers:
            self.sim.logger.log("Server {} : {} utilization".format(server.identifier, (server.utilization)))
            for s in server.children:
                self.sim.logger.log("{} ".format(s.identifier))
    def print_dual(self, servers):
        self.sim.logger.log("DUAL------------".format())
        for server in servers:
            self.sim.logger.log("Server {} : {} utilization".format(server.identifier, (server.utilization)))
            s = server.child
            self.sim.logger.log("{} ".format(s.identifier))


    def on_activate(self, job):
        """
        Deal with a (real) task activation.
        """
        subsystem = self.task_to_subsystem[job.task]
        subsystem.update_budget()
        add_job(self.sim, job, self.servers[job.task])
        subsystem.resched(self.processors[0])

    def on_terminated(self, job):
        """
        Deal with a (real) job termination.
        """
        subsystem = self.task_to_subsystem[job.task]
        self.task_to_subsystem[job.task].update_budget()
        s = self.servers[job.task]

        subsystem.resched(self.processors[0])

    def schedule(self, _):
        """
        This method is called by the simulator. The sub-systems that should be
        rescheduled are also scheduled.
        """
        decisions = []
        
        for subsystem in self.subsystems:
            if subsystem.to_reschedule:
                decisions += subsystem.schedule()

        return decisions

def dual(servers):
    """
    From a list of servers, returns a list of corresponding DualServers.
    """
    return [DualServer(s) for s in servers]


class ProperSubsystem(object):
    """
    Proper sub-system. A proper sub-system is the set of the tasks belonging to
    a unit server (server with utilization of 1) and a set of processors.
    """

    def __init__(self, sim, root, processors, level):
        self.sim = sim
        self.root = root
        self.processors = processors
        self.level = level
        self.virtual = []
        self.last_update = 0
        self.to_reschedule = False
        self.timer = None

        self.utilization = sum([s.utilization for s in root.children if s.dummyServer is False])

        self.dummy_timer = None

        self.is_idle = True

        self.idleBegin = 0
        self.idleEnd = 0

        self.busyBegin = 0
        self.busyEnd =0
        
        
    def update_budget(self):
        """
        Update the budget of the servers.
        """
        
        time_since_last_update = self.sim.now() - self.last_update
        for server in self.virtual:
            server.budget -= time_since_last_update
        self.last_update = self.sim.now()


    def resched(self, cpu):
        """
        Plannify a scheduling decision on processor cpu. Ignore it if already
        planned.
        """

        for cpu in self.processors: 
            if any(x[0] == 2 or x[0] == 3 for x in cpu._evts):
                return
        #if not self.to_reschedule:
        self.to_reschedule = True
        cpu.resched()

    def virtual_event(self, cpu):
        """
        Virtual scheduling event. Happens when a virtual job terminates.
        """

        self.update_budget()
        self.resched(cpu)

    def end_dummy(self, cpu):
        #self.is_idle = False

        #self.sim.logger.log("end_dummy dual".format())
        
        self.update_budget()
        self.to_reschedule = True
        self.resched(cpu)
        
    def add_timer(self, wakeup_delay, CPU):

        #self.sim.logger.log("add_timer to {}".format(wakeup_delay))
        if self.dummy_timer:
            self.dummy_timer.stop()
            
        self.dummy_timer = Timer(self.sim, ProperSubsystem.end_dummy,
                           (self, CPU), wakeup_delay,
                           cpu=CPU, in_ms=False)
        self.dummy_timer.start()

    # cumulative slack computation
    def CSC(self, t):

        active_servers = [s for s in self.root.children if not s.dummyServer and s.budget > 0]
        servers = [s for s in self.root.children if not s.dummyServer]
        
        beta = beta = sum(s.budget for s in servers)
        omega = max(0, self.root.next_deadline - t - beta)
        
        if self.is_idle is True:
            #self.sim.logger.log("was IDLE before t".format())
            

            delta = delta_t(self, servers, t, t)

            #delta = omega

            #self.sim.logger.log("delta_t = {} t = {}".format(delta, self.root.next_deadline))

            if delta > 0:
                self.idleEnd = t + delta
                self.busyBegin = self.idleEnd
                self.busyEnd = self.idleBegin + (self.idleEnd - self.idleBegin)/ float(1-self.utilization)
                
                #self.sim.logger.log("idle [{},{}) -- busy [{},{})".format(self.idleBegin, self.idleEnd, self.busyBegin, self.busyEnd))
                #self.sim.logger.log("u = {}, 1-u = {}".format(self.utilization, (1-self.utilization)))
            else:
                #self.keep = False
                self.idleEnd = t
                self.busyBegin = self.idleEnd
                self.busyEnd = max(beta, self.busyEnd)
                
                #self.sim.logger.log("else idle [{},{}) -- busy [{},{})".format(self.idleBegin, self.idleEnd, self.busyBegin, self.busyEnd))
        else:
            if active_servers:
                #self.sim.logger.log("Servidores ativos, BUSY".format())
                self.busyEnd = max(t + beta, self.busyEnd)
                #self.keep = False
                #self.sim.logger.log("idle [{},{}) -- busy [{},{})".format(self.idleBegin, self.idleEnd, self.busyBegin, self.busyEnd))
            else:
                #self.sim.logger.log("Sem servidores ativos, IDLE".format())
                
                deltat = delta_t(self, servers, t, self.root.next_deadline)
                omegat = omega_t(self, servers, t, self.root.next_deadline)
                
                #self.sim.logger.log("delta = {} t = {}".format(deltat, self.root.next_deadline))

                #self.sim.logger.log("omega = {} t = {}".format(omegat, self.root.next_deadline))
                
                self.idleBegin = t
                # implementation choice :)
                self.idleEnd = self.root.next_deadline + deltat#omegat
                self.busyBegin = self.idleEnd
                self.busyEnd = self.idleBegin + (self.idleEnd - self.idleBegin)/ float(1-self.utilization)
                #self.keep = True
                
                #self.sim.logger.log("idle [{},{}) -- busy [{},{})".format(self.idleBegin, self.idleEnd, self.busyBegin, self.busyEnd))
            

    def schedule(self):
        """
        Schedule this proper sub-system.
        """
        
        self.to_reschedule = False
        self.virtual    = []
        decision       = []
        processors      = []            
        processors      = self.processors

        self.CSC(self.sim.now())

        t = self.sim.now()

        if t >= self.idleBegin and t < self.idleEnd:
            self.is_idle = True
            #self.sim.logger.log("DUAL IDLE [{},{}) -- busy [{},{})".format(self.idleBegin, self.idleEnd, self.busyBegin, self.busyEnd))

            # add schedule event - end of idle period
            #if self.keep is False:
            self.add_timer(self.idleEnd-t, self.processors[0])
        else:
            self.is_idle = False
            #self.sim.logger.log("DUAL BUSY [{},{})".format(self.busyBegin, self.busyEnd))

        selected = select_jobs(self, self.root, self.virtual)

        idle = [s for s in selected if s.task.name == 'IdleTask']
        jobs = [s.job for s in selected if s.task.name != 'IdleTask' and s.task.name != "wcet"]


        if idle:
            # CSC dual level 
            if self.level % 2 == 0:

                # use busyInterval to save energy
                self.processors[0].set_dummy(self.busyEnd-t)
                #self.sim.logger.log("active low-power state during [{},{})".format(self.busyBegin, self.busyEnd))

            # CSC primal level
            else:
                # use idleInterval to save energy
                self.processors[0].set_dummy(self.idleEnd-t)

            processors = self.processors[1:] # Refresh avaliable processors list
            decision.append((None, self.processors[0])) # Set dummy to first processor
        else:
            self.processors[0].stop_dummy()
                

        wakeup_delay = min([s.budget for s in self.virtual if s.budget > 0])
        
        if wakeup_delay > 0:
            self.timer = Timer(self.sim, ProperSubsystem.virtual_event,
                                   (self, self.processors[0]), wakeup_delay,
                                   cpu=self.processors[0], in_ms=False)
            self.timer.start()

        cpus = []

        #first, leave already executing tasks on their current processors;
        for cpu in processors:
            if cpu.running in jobs:
                #cpus.append(cpu)
                jobs.remove(cpu.running) # remove job and cpu
            else:
                cpus.append(cpu)
           
        # second, assign previously idle tasks to their last-used processor, when its available
        aux_jobs = list(jobs)
        for job in aux_jobs:
            if job.task.last_cpu in cpus:
                #if job.task.last_cpu.is_running():
                decision.append((job, job.task.last_cpu))
                jobs.remove(job)
                cpus.remove(job.task.last_cpu)

       
        # third, assign remaining tasks to free processors arbitrarily
        for cpu in cpus:
            if jobs:
                decision.append((jobs.pop(), cpu))
            else:
                decision.append((None, cpu))
       
        return decision
