
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/

import random, numpy, math
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
	def __init__(self, NStates, Nactions):
		self.NStates = NStates
		self.Nactions = Nactions

		self.model = self._createModel()

	def _createModel(self):
		model = Sequential()

		# Simple Model with Two Hidden Layers and a Linear Output Layer. The Input layer is simply the State input. 
		model.add(Dense(units=64, activation='relu', input_dim=self.NStates))
		model.add(Dense(units=32, activation='relu'))
		model.add(Dense(units=self.Nactions, activation='linear'))				# Linear Output Layer as we are estimating a Function Q[S,A]

		model.compile(loss='mse', optimizer='adam')     # use adam as an alternative optimsiuer as per comment

		return model

	def train(self, x, y, epoch=1, verbose=0):
		self.model.fit(x, y, batch_size=64, epochs=epoch, verbose=verbose)

	def predict(self, s):
		return self.model.predict(s)

	def predictOne(self, s):
		return self.predict(s.reshape(1, self.NStates)).flatten()

# A simple Experience Replay memory

class ExpReplay:   # stored as ( s, a, r, s_ )
	samples = []

	def __init__(self, capacity):
		self.capacity = capacity

	def add(self, sample):
		self.samples.append(sample)        

		if len(self.samples) > self.capacity:
			self.samples.pop(0)

	def sample(self, n):
		n = min(n, len(self.samples))
		return random.sample(self.samples, n)


# DQN Reinforcement Learning Algorithm  Hyper Parameters
ExpReplay_CAPACITY = 2000
OBSERVEPERIOD = 750		# Period start real Training 
BATCH_SIZE = 128
GAMMA = 0.95				# Q Reward Discount Gamma
MAX_EPSILON = 1
MIN_EPSILON = 0.05
LAMBDA = 0.0005      		# Speed of Epsilon decay

class Agent:
	def __init__(self, NStates, Nactions):
		self.NStates = NStates
		self.Nactions = Nactions

		self.brain = Brain(NStates, Nactions)
		self.ExpReplay = ExpReplay(ExpReplay_CAPACITY)
		self.steps = 0
		self.epsilon = MAX_EPSILON
        

	def Act(self, s):
		if (random.random() < self.epsilon or self.steps < OBSERVEPERIOD):
			return random.randint(0, self.Nactions-1)						# Explore 
		else:
			return numpy.argmax(self.brain.predictOne(s))					


	def CaptureSample(self, sample): 
		self.ExpReplay.add(sample)        

		
		self.steps += 1
		if(self.steps>OBSERVEPERIOD):
			self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * (self.steps-OBSERVEPERIOD))

	def Process(self):    
		batch = self.ExpReplay.sample(BATCH_SIZE)
		batchLen = len(batch)

		no_state = numpy.zeros(self.NStates)

		states = numpy.array([ batchitem[0] for batchitem in batch ])
		states_ = numpy.array([ (no_state if batchitem[3] is None else batchitem[3]) for batchitem in batch ])

		predictedQ = self.brain.predict(states)						
		predictedNextQ = self.brain.predict(states_)				

		x = numpy.zeros((batchLen, self.NStates))
		y = numpy.zeros((batchLen, self.Nactions))

		
		for i in range(batchLen):
			batchitem = batch[i]
			state = batchitem[0]; a = batchitem[1]; reward = batchitem[2]; nextstate = batchitem[3]
			
			targetQ = predictedQ[i]
			if nextstate is None:
				targetQ[a] = reward												
			else:
				targetQ[a] = reward + GAMMA * numpy.amax(predictedNextQ[i])   	

			x[i] = state
			y[i] = targetQ

		self.brain.train(x, y)						