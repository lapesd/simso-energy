# coding=utf-8
from collections import deque, namedtuple
from SimPy.Simulation import *

CPU_CONSUME = 1

# State, Break-Event time, Energy penalty, energy consumption per ms
low_power_info = [	("SHUTDOWN", 10, 4.99995, 0.00001), 
					("DORMANT", 2, 0.9, 0.1), 
					("STANDBY", 0.1, 0.025, 0.5)]
		
class Energy(Process):
	_identifier = 1

	@classmethod
	def init(cls):
		cls._identifier = 1

	def __init__(self, model):
		Process.__init__(self, name="Energy "+str(Energy._identifier), sim=model)
		self.level = Level(name="Energy "+str(Energy._identifier), sim=model)

		self._identifier = Energy._identifier
		Energy._identifier += 1

		self.amount = 0

		self._evts = deque([])
		self.power_info = low_power_info

		self.start = -1
		self.start_sub = -1
		self.state = ("RUN", 0, 0, 1)
		self.old = ("RUN", 0, 0, 1)
		self.keep = 0

		self.idle_time = 0
	
	def count_energy(self, cpu, time):

		if self.sim.now()+time > self.sim.duration:
			time = self.sim.duration - self.sim.now()

		time = float(time) / self.sim.cycles_per_ms
	
		if time <= 0:
			return

		self.idle_time = 0

		# Look for the best low-power state
		for s, BET, Pen, Co in self.power_info:
			if time >  BET:
				self._evts.append((CPU_CONSUME, (time * Co + Pen), cpu, s, time))
				return

		# Short idle time
		self._evts.append((CPU_CONSUME, time, cpu, "RUN", time))

	def record_idle_time(self, cpu, time):

		time = float(time) / self.sim.cycles_per_ms
	
		if time <= 0:
			return

		# Look for the best low-power state
		for s, BET, Pen, Co in self.power_info:
			if time >  BET:
				self._evts.append((CPU_CONSUME, (time * Co + Pen), cpu, s, time))
				return

		# Short idle time
		self._evts.append((CPU_CONSUME, time, cpu, "RUN", time))

	def flag(self, idle_time):
		self.idle_time = idle_time

	def energyMode(self, t, p, x):

		x = float(x) / self.sim.cycles_per_ms

		new_state = ("RUN", 0, 0, 1)

		for s, BET, Pen, Co in self.power_info:
			if x > BET:
				new_state = (s, BET, Pen, Co)
				break

		# first activation
		if self.start == -1:
			self.start = t
			self.old = self.state
			self.state = new_state
			self.sim.logger.log("Active {} on {} during x = {} ms".format(self.state[0], p.name, x))
		return self.state[0]

		# if idle and new_state is more efficient
		if new_state[1] > self.state[1]:
		
			sub_len = 0
			# first sub-interval
			if self.start_sub == -1:
				sub_len = t - self.start
			else:
				sub_len = t - self.start_sub 

			# cycles_per_ms to ms
			sub_len = float(sub_len) / self.sim.cycles_per_ms

			# consumption of interval with state self.state
			self.add_consumption(self.old, self.state, sub_len)

			# update new state and the value to subinterval start
			self.start_sub = t
			self.old = self.state
			self.state = new_state

		return self.state[0]

			
	def add_consumption(self, old, state, time):
		consumption =0
		if time > 0:
			self.amount -= time

			pen = state[2]

			# 1/2 * T_s * (C_s - C_0)
			if old[0] != "RUN":	
				pen = ((old[3]-state[3])*state[1])/2

			consumption = state[3]*time + state[2] - old[2]
			self.amount += consumption
		return consumption

	def stop_idle(self, cpu, end_time):

		idle_len = end_time - self.start 

		if self.start_sub != -1:
			sub_len = end_time - self.start_sub
		else:
			sub_len = end_time - self.start

		idle_len = float(idle_len) / self.sim.cycles_per_ms
		sub_len = float(sub_len) / self.sim.cycles_per_ms

		con = self.add_consumption(self.old, self.state, sub_len)
		self.sim.logger.log(" {} : {} ms, {} consuming {} energy - Done".format(self.state[0], sub_len, cpu.name, con))

		self.start = self.start_sub = -1
		self.state = ("RUN", 0, 0, 1)
		self.old = ("RUN", 0, 0, 1)

	def set_state(self, time):
		
		self.start = self.sim.now()
		self.state = ("RUN", 0, 0, 1)

		time = float(time) / self.sim.cycles_per_ms

		for s in self.power_info:
			if time > s[1]:
				self.state = s
				break

	def stop_state(self, cpu, time):

		time = self.sim.now() - self.start
		time = float(time) / self.sim.cycles_per_ms


		self._evts.append((CPU_CONSUME, (time * self.state[3] + self.state[2]), cpu, self.state[0], time))

		self.start = 0
		self.keep = 0

	@property
	def states(self):
		return self.power_info

	# Update the total energy consumption at end of simulation time
	def total(self):
		add = 0
		
		if self.state[0] != "RUN":
			end_time = self.sim.duration
			idle_len = end_time - self.start 

			if self.start_sub != -1:
				sub_len = end_time - self.start_sub
			else:
				sub_len = end_time - self.start

			idle_len = float(idle_len) / self.sim.cycles_per_ms
			sub_len = float(sub_len) / self.sim.cycles_per_ms

			con = self.add_consumption(self.old, self.state, sub_len)
			self.sim.logger.log(" {} : {} ms, consuming {} energy - Done".format(self.state[0], sub_len, con))

		return self.amount 

	def time_amount(self):
		return self.amount

	def get_energy(self, time):

		if time <= 0:
			return 0

		return (time*self.state[3] + self.state[2])

	def consumption(self, time):

		# Look for the best low-power state
		for s, BET, Pen, Co in self.power_info:
			if time > BET:
				return (time * Co + Pen)

		return time

	def on(self):
		yield put, self, self.level, 0.0

		self.amount = self.sim.duration/self.sim.cycles_per_ms # Max energy consumption

		while True:
			# Wait event.
			yield waituntil, self, lambda:any(x[0] for x in self._evts)
			# So, get one
			evt = self._evts.popleft()

			# Check event type
			if evt[0] == CPU_CONSUME:
				self.amount -= evt[4] # remove idle period

				yield put, self, self.level, evt[1]
				self.sim.logger.log(" {} : {} ms, {} consuming {} energy".format(
            		evt[3], evt[4], evt[2].name, evt[1]))

