# coding=utf-8

from collections import deque
from SimPy.Simulation import Process, Monitor, hold, waituntil
from simso.core.ProcEvent import ProcRunEvent, ProcIdleEvent, \
    ProcOverheadEvent, ProcCxtSaveEvent, ProcCxtLoadEvent


RESCHED = 1
ACTIVATE = 2
TERMINATE = 3
TIMER = 4
PREEMPT = 5
SPEED = 6

class ProcInfo(object):
    def __init__(self, identifier, name, cs_overhead=0, cl_overhead=0,
                 migration_overhead=0, speed=1.0, data=None):
        self.identifier = identifier
        self.name = name
        self.penalty = 0
        self.caches = []
        self.cs_overhead = cs_overhead
        self.cl_overhead = cl_overhead
        self.migration_overhead = migration_overhead
        if data is None:
            data = {}
        self.data = data
        self.speed = speed

    def add_cache(self, cache):
        self.caches.append(cache)


class Processor(Process):
    """
    A processor is responsible of deciding whether the simulated processor
    should execute a job or execute the scheduler. There is one instance of
    Processor per simulated processor. Those are responsible to call the
    scheduler methods.

    When a scheduler needs to take a scheduling decision, it must invoke the
    :meth:`resched` method. This is typically done in the :meth:`on_activate
    <simso.core.Scheduler.Scheduler.on_activate>`, :meth:`on_terminated
    <simso.core.Scheduler.Scheduler.on_terminated>` or in a :class:`timer
    <simso.core.Timer.Timer>` handler.
    """
    _identifier = 0

    @classmethod
    def init(cls):
        cls._identifier = 0

    def __init__(self, model, proc_info, power):
        Process.__init__(self, name=proc_info.name, sim=model)
        self._model = model
        self._internal_id = Processor._identifier
        Processor._identifier += 1
        self.identifier = proc_info.identifier
        self._running = None
        self.was_running = None
        self._evts = deque([])
        self.sched = model.scheduler
        self.monitor = Monitor(name="Monitor" + proc_info.name, sim=model)
        self._caches = []
        self._penalty = proc_info.penalty
        self._cs_overhead = proc_info.cs_overhead
        self._cl_overhead = proc_info.cl_overhead
        self._migration_overhead = proc_info.migration_overhead
        self.set_caches(proc_info.caches)
        self.timer_monitor = Monitor(name="Monitor Timer" + proc_info.name,
                                     sim=model)
        self._speed = proc_info.speed

        self._power = power

        self.idle_time = None
        

    def resched(self):
        """
        Add a resched event to the list of events to handle.
        """
        self._evts.append((RESCHED,))

    def activate(self, job):
        self._evts.append((ACTIVATE, job))

    def terminate(self, job):
        self._evts.append((TERMINATE, job))
        self._running = None

    def preempt(self, job=None):
        self._evts = deque([e for e in self._evts if e[0] != PREEMPT])
        self._evts.append((PREEMPT,))
        self._running = job

    def timer(self, timer):
        self._evts.append((TIMER, timer))

    def set_speed(self, speed):
        assert speed >= 0, "Speed must be positive."
        self._evts.append((SPEED, speed))

    def set_idle(self, period):
        self._power.count_energy(self, float(period))

    def set_dummy(self, x):
        if self.idle_time == None:
            self.idle_time = self.sim.now()

            self._power.energyMode(self.sim.now(), self, x) 

    def stop_dummy(self):
        if self.idle_time != None:
            self.idle_time = None

            self._power.stop_idle(self, self.sim.now())       

    def set_busy(self):
        if self.idle_time != None:
            period = self.sim.now() - self.idle_time
            self._power.record_idle_time(self, period)
            self.idle_time = None
    
    def set_idle_extend(self, time):

        if self.idle_time == None:
            self.idle_time = self.sim.now()
            self._power.set_state(time)

    def stop_idle_extend(self):
        if self.idle_time != None:
            time = self.sim.now() - self.idle_time 
            self.idle_time = None

            self._power.stop_state(self, time)

    @property
    def speed(self):
        return self._speed

    def is_running(self):
        """
        Return True if a job is currently running on that processor.
        """
        return self._running is not None

    def set_caches(self, caches):
        self._caches = caches
        for cache in caches:
            cache.shared_with.append(self)

    def get_caches(self):
        return self._caches

    caches = property(get_caches, set_caches)

    @property
    def penalty_memaccess(self):
        return self._penalty

    @property
    def cs_overhead(self):
        return self._cs_overhead

    @property
    def cl_overhead(self):
        return self._cl_overhead

    @property
    def internal_id(self):
        """A unique, internal, id."""
        return self._internal_id

    @property
    def running(self):
        """
        The job currently running on that processor. None if no job is
        currently running on the processor.
        """
        return self._running

    @property
    def less_effic_BET(self):
        bet = 10000
        for s, BET, Pen, Co in self._power.states:
            if BET < bet:
                bet = BET
        return bet


    # Update the energy consumption at the end of simulation
    def update(self):
        total = 0
        
        duration = self._model.duration/self._model.cycles_per_ms
        duration_t = duration*len(self._model.processors)
        utilization = (self._model.utilization)/len(self._model.processors)
        exec_time = (duration_t*utilization)
        ideal = 0

        for cpu in self._model.processors:
            period = 0
            if cpu.idle_time != None:
                period = float(self._model.duration - cpu.idle_time) / self._model.cycles_per_ms
            total += cpu._power.total()

        idle_time = duration_t - exec_time
        consumo_min = cpu._power.consumption(idle_time) + exec_time

        final = (total/consumo_min)-1
        max_c = ((duration*len(self._model.processors))/consumo_min)-1
           
        self._model.logger.log("Energy consumption with idle = {}. {} of {} max = {}".format(final, total, consumo_min, max_c))
        
    def run(self):
        while True:
            if not self._evts:
                job = self._running
                if job:
                    yield waituntil, self, lambda: job.context_ok
                    self.monitor.observe(ProcCxtLoadEvent())
                    yield hold, self, self.cl_overhead  # overhead load context
                    self.monitor.observe(ProcCxtLoadEvent(terminated=True))
                    job.interruptReset()
                    self.sim.reactivate(job)
                    self.monitor.observe(ProcRunEvent(job))
                    job.context_ok = False
                else:
                    self.monitor.observe(ProcIdleEvent())

                # Wait event.
                yield waituntil, self, lambda: self._evts
                if job:
                    self.interrupt(job)
                    self.monitor.observe(ProcCxtSaveEvent())
                    yield hold, self, self.cs_overhead  # overhead save context
                    self.monitor.observe(ProcCxtSaveEvent(terminated=True))
                    job.context_ok = True

            evt = self._evts.popleft()
            if evt[0] == RESCHED:
                if any(x[0] != RESCHED for x in self._evts):
                    self._evts.append(evt)
                    continue

            if evt[0] == ACTIVATE:
                
                self.sched.on_activate(evt[1])
                self.monitor.observe(ProcOverheadEvent("JobActivation"))
                self.sched.monitor_begin_activate(self)
                yield hold, self, self.sched.overhead_activate
                self.sched.monitor_end_activate(self)
            elif evt[0] == TERMINATE:
                self.sched.on_terminated(evt[1])
                self.monitor.observe(ProcOverheadEvent("JobTermination"))
                self.sched.monitor_begin_terminate(self)
                yield hold, self, self.sched.overhead_terminate
                self.sched.monitor_end_terminate(self)
            elif evt[0] == TIMER:
                self.timer_monitor.observe(None)
                if evt[1].overhead > 0:
                    print(self.sim.now(), "hold", evt[1].overhead)
                    yield hold, self, evt[1].overhead
                evt[1].call_handler()
            elif evt[0] == SPEED:
                self._speed = evt[1]

            elif evt[0] == RESCHED:

                self.monitor.observe(ProcOverheadEvent("Scheduling"))
                self.sched.monitor_begin_schedule(self)

                
                yield waituntil, self, self.sched.get_lock
                decisions = self.sched.schedule(self)
                yield hold, self, self.sched.overhead  # overhead scheduling             


                if type(decisions) is not list:
                    decisions = [decisions]
                decisions = [d for d in decisions if d is not None]

                for job, cpu in decisions:
                    # If there is nothing to change, simply ignore:
                    if cpu.running == job:
                        continue

                    # If trying to execute a terminated job, warn and ignore:
                    if job is not None and not job.is_active():
                        print("Can't schedule a terminated job! ({})"
                              .format(job.name))
                        continue

                    # if the job was running somewhere else, stop it.
                    if job and job.cpu.running == job:# and job.cpu.running != cpu:
                        job.cpu.preempt()

                    # Send that job to processor cpu.
                    cpu.preempt(job)

                    if job:
                        job.task.cpu = cpu

                # Forbid to run a job simultaneously on 2 or more processors.
                running_tasks = [
                    cpu.running.name
                    for cpu in self._model.processors if cpu.running]
                assert len(set(running_tasks)) == len(running_tasks), \
                    "Try to run a job on 2 processors simultaneously!"
                                         
                self.sched.release_lock()
                self.sched.monitor_end_schedule(self)
                

