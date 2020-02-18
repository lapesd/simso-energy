from simso.core.etm.AbstractExecutionTimeModel \
    import AbstractExecutionTimeModel
import random


class WCET(AbstractExecutionTimeModel):
    def __init__(self, sim, _):
        self.sim = sim
        self.et = {}
        self.executed = {}
        self.on_execute_date = {}

    def init(self):
        pass

    def update_executed(self, job):
        if job in self.on_execute_date:
            self.executed[job] += (self.sim.now() - self.on_execute_date[job]
                                   ) * job.cpu.speed

            del self.on_execute_date[job]

    def on_activate(self, job):
        self.executed[job] = 0

        #acet = random.uniform(0.000001, job.wcet)

        #sd = random.uniform(0.000001, job.wcet - acet)

        #wcet1
        #et = min(job.wcet, random.normalvariate(acet, sd))

        #wcet2
        # job.wcet - 100
        # x - 90
        # x = float(job.wcet*90) / 100
        #et = random.uniform(float(job.wcet)/2, job.wcet)

        et = random.uniform(float(job.wcet*50) / 100, job.wcet) 

        self.et[job] = et * self.sim.cycles_per_ms


    def on_execute(self, job):
        self.on_execute_date[job] = self.sim.now()

    def on_preempted(self, job):
        self.update_executed(job)

    def on_terminated(self, job):
        self.update_executed(job)
        del self.et[job]

    def on_abort(self, job):
        self.update_executed(job)
        del self.et[job]

    def get_executed(self, job):
        if job in self.on_execute_date:
            c = (self.sim.now() - self.on_execute_date[job]) * job.cpu.speed
        else:
            c = 0
        return self.executed[job] + c

    def get_ret(self, job):
        #wcet_cycles = int(job.wcet * self.sim.cycles_per_ms)
        #return int(wcet_cycles - self.get_executed(job))
        return int(self.et[job] - self.get_executed(job))

    def update(self):
        for job in list(self.on_execute_date.keys()):
            self.update_executed(job)
