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
from simso.schedulers.Uni_RUNServer import EDFServer, TaskServer, DualServer, \
    select_jobs, add_job, get_child_tasks, update_deadline, \
    update_deadline2, algorithm_1, algorithm_2, release_dummy_job, \
    next_activation
from simso.schedulers import scheduler
from simso.core.Packing import pack_BFD, pack_WFD, pack_FFD, \
    pack_BF, pack_FF, pack_WF, decrease, increase

INFINITO = 9000000000000


@scheduler("simso.schedulers.Uni_off_RUN")
class Uni_off_RUN(Scheduler):
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

    def before(self, servers, dummy):
        """
        Create IdleTasks in order to reach 100% system utilization.
        """
        from collections import namedtuple
        # pylint: disable-msg=C0103
        IdleTask = namedtuple('IdleTask', ['utilization', 'deadline', 'name'])

        idle = len(self.processors) - sum([s.utilization for s in servers])
        new_servers = []
        for server in servers:
            if server.utilization < 1 and idle > 0:
                dummy = IdleTask(min(1 - server.utilization, idle), 0, 'IdleTask')
                # Create a EDFserver to add dummy and the task server
                s = EDFServer()
                s.add_child(TaskServer(dummy))
                s.add_child(server)

                idle -= dummy.utilization
                new_servers.append(s)
            else:
                new_servers.append(server)

        return new_servers

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

        if min(1, idle) == 1:
            idle -= 1

        if min(1, idle) == 1:
            idle -= 1

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
            #self.sim.logger.log("add dummy {} to server {} idle = {}".format(t.identifier, server.identifier, Fraction(round(min(1, idle),2))))
            idle -= min(1, idle)

    def independent2(self, servers, dummy):
        """
        Add dummy sem misturar
        """
        from collections import namedtuple
        # pylint: disable-msg=C0103
        IdleTask = namedtuple('IdleTask', ['utilization', 'deadline', 'name'])

        idle = len(self.processors) - sum([s.utilization for s in servers])

        while idle > 0:
            task = IdleTask(min(1, idle), 0, 'IdleTask')
            
            t = TaskServer(task)
            self.Tdummy.append(t)
            t.dummyServer = True
           
            servers[0].add_child(t)
            
            dummy.append(t)
            #self.sim.logger.log("add dummy {} to server {} idle = {}".format(t.identifier, server.identifier, Fraction(round(min(1, idle),2))))
            idle -= min(1, idle)

    def add_proper_subsystem(self, server):
        """
        Create a proper sub-system from a unit server.
        """
        tasks_servers = get_child_tasks(server)
        subsystem_utilization = sum([t.utilization for t in tasks_servers])
        cpus = []
        while subsystem_utilization > 0:
            cpus.append(self.available_cpus.pop())
            subsystem_utilization -= 1

        subsystem = ProperSubsystem(self.sim, server, cpus, self.Tdummy, 10)
        for server in tasks_servers:
            self.task_to_subsystem[server.task] = subsystem
        self.subsystems.append(subsystem)

    def add_subsystem(self, server):
        """
        Create a proper sub-system from a unit server.
        """
        tasks_servers = get_child_tasks(server)
        subsystem_utilization = sum([t.utilization for t in tasks_servers])
        cpus = []
        while subsystem_utilization > 0:
            cpus.append(self.available_cpus.pop())
            subsystem_utilization -= 1

        subsystem = ProperSubsystem(self.sim, server, cpus, self.Tdummy)

        for cpu in self.available_cpus:
            if not cpu in cpus:
                cpu.set_idle()

        #subsystem = ProperSubsystem(self.sim, server, self.available_cpus, self.Tdummy)
        for server in tasks_servers:
            self.task_to_subsystem[server.task] = subsystem
        self.subsystems.append(subsystem)

    def remove_unit_servers(self, servers):
        """
        Remove all the unit servers for a list and create a proper sub-system
        instead.
        """
        for server in servers:
            if server.utilization == 1:
                self.add_proper_subsystem(server)
        servers[:] = [s for s in servers if s.utilization < 1]

    def reduce_iterations(self, servers):
        """
        Offline part of the RUN Scheduler. Create the reduction tree.
        """
        dummy = []
        #self.print_taks_in_servers(servers)
        #servers = increase(self, servers, len(self.processors), dummy)
        #servers = self.before(servers, dummy)
        servers = pack_BFD(servers)
        #self.print_servers(servers)
        #self.fill(servers, dummy)
        self.independent2(servers, dummy)
        #self.print_servers(servers)
        #self.remove_unit_servers(servers)
        level = 0
        while len(servers) > 1:
        #while servers:
            #self.print_servers(servers)
            s = dual(servers)
            #self.print_dual(s)
            servers = pack_BFD(s)
            #self.print_servers(servers)
            #self.remove_unit_servers(servers)
            level += 1

        self.sim.logger.log("Levels = {}".format(level))
        self.add_subsystem(servers[0])


        for d in dummy:
            #set_budget(d)
            update_deadline2(self.sim, servers[0], d, INFINITO)          

    def print_taks_in_servers(self, servers):
        self.sim.logger.log("------------".format())
        for s in servers:
            self.sim.logger.log("Server {} : {} utilization".format(s.identifier, float(s.utilization)))
            self.sim.logger.log("{}, C = {}, P = {} ".format(s.task.name, s.task.wcet, s.task.period))
    def print_servers(self, servers):
        self.sim.logger.log("PACK------------".format())
        for server in servers:
            self.sim.logger.log("Server {} : {} utilization".format(server.identifier, float(server.utilization)))
            for s in server.children:
                self.sim.logger.log("{} ".format(s.identifier))
    def print_dual(self, servers):
        self.sim.logger.log("DUAL------------".format())
        for server in servers:
            self.sim.logger.log("Server {} : {} utilization".format(server.identifier, float(server.utilization)))
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
        subsystem.resched(self.processors[0])

    def schedule(self, _):
        """
        This method is called by the simulator. The sub-systems that should be
        rescheduled are also scheduled.
        """
        decisions = []

        
        for subsystem in self.subsystems:
            if subsystem.to_reschedule:
                for cpu in self.sim.processors: 
                    if any(x[0] == 1 or x[0] == 2 or x[0] == 3 for x in cpu._evts):
                       #self.sim.logger.log("RESCHED IN {}".format(cpu.name))
                       subsystem.to_reschedule = False
                       return decisions

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

    def __init__(self, sim, root, processors, dummy):
        self.sim = sim
        self.root = root
        self.processors = processors
        self.dummyTask = dummy
        self.virtual = []
        self.last_update = 0
        self.to_reschedule = False
        self.timer = None

        self.dummy_timer2 = None

        self.dummy_is_running = False
        self.is_idle = False
        self.keep = False
        self.start_idle = 0

        self.slack = 0
        self.wcet = False
        
        self.dummy_last_cpu = None

        self.t0 = 0
        self.mode = 2

        # Execute real job first
        #self.dummy_activated(self.processors[0])
        #update_deadline2(self.sim, self.dummyTask, INFINITO)

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
        if not self.to_reschedule:
            self.to_reschedule = True
            cpu.resched()

    def virtual_event(self, cpu):
        """
        Virtual scheduling event. Happens when a virtual job terminates.
        """

        self.update_budget()
        self.resched(cpu)

    def dummy_activated(self, cpu):

        update_deadline2(self.sim, self.root, self.dummyTask[0], INFINITO)
        self.is_idle = False
        self.keep = False

        self.sim.logger.log("NOT KEEP NOT IDLE".format())
        
        self.update_budget()
        self.to_reschedule = True
        cpu.resched()
        
    def add_timer(self, wakeup_delay, CPU):
        if self.dummy_timer2:
            self.dummy_timer2.stop()
            
        self.dummy_timer2 = Timer(self.sim, ProperSubsystem.dummy_activated,
                           (self, CPU), wakeup_delay,
                           cpu=CPU, in_ms=False)
        self.dummy_timer2.start()

    def schedule(self):
        """
        Schedule this proper sub-system.
        """

        self.to_reschedule = False
        decision = []            

        self.virtual = []

        
        selected = select_jobs(self, self.root, self.virtual)

        idle = [s for s in selected if s.task.name == 'IdleTask']
        jobs = [s.job for s in selected if s.task.name != 'IdleTask' and s.task.name != "wcet"]
        wcet = [s for s in selected if s.task.name == "wcet"]

        min_s = min(self.virtual, key=lambda s: s.budget)
        wakeup_delay = min_s.budget
        
        
        old = self.virtual
        while wakeup_delay == 0:
            old.remove(min_s)
            min_s = min(old, key=lambda s: s.budget)
            wakeup_delay = min_s.budget
        
        if wakeup_delay > 0:
            self.timer = Timer(self.sim, ProperSubsystem.virtual_event,
                                   (self, self.processors[0]), wakeup_delay,
                                   cpu=self.processors[0], in_ms=False)
            self.timer.start()

        cpus = []

        #first, leave already executing tasks on their current processors;
        for cpu in self.processors:
            if cpu.running in jobs:
                #cpus.append(cpu)
                jobs.remove(cpu.running) # remove job and cpu
            else:
                cpus.append(cpu)

        # second, assign previously idle tasks to their last-used processor, when its available
        aux_jobs = list(jobs)
        for job in aux_jobs:
            if job.task.last_cpu in cpus:
                decision.append((job, job.task.last_cpu))
                jobs.remove(job)
                cpus.remove(job.task.last_cpu)

       
        # third, assign remaining tasks to free processors arbitrarily
        for cpu in cpus:
            if jobs:
                decision.append((jobs.pop(), cpu))
            else:
                
                period = 0
                
                if self.keep and self.slack > 0:
                    period = self.slack
                    #self.sim.logger.log("slack {}".format(self.slack))
                    self.slack = 0 
                    cpu.set_idle(period)
                elif not self.keep and idle:
                    period = next_activation(idle.pop()) - self.sim.now()
                    #self.sim.logger.log("next_activation {}".format(period))
                    cpu.set_idle(period)
                
                elif wcet:
                    t = wcet.pop()
                    period = t.task.deadline
                    cpu.set_idle(period)
                    #self.sim.logger.log("WCET NEXT {}".format(period))
                
                
                decision.append((None, cpu))
       
        return decision
