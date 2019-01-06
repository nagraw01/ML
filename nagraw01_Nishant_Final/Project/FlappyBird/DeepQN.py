import tensorflow as tf
import numpy as np
import random
from collections import deque

#following are the hyperparameters
FPS = 1
GAMMA = 0.99
OBSERVE_TIMESTEPS = 100000
EXPLORE = 200000
EPSILON1 = 0.1
EPSILON2 = 0.0001
MEMORY = 50000
BATCH = 32
UPDATE_TIME = 100

class DQN:
    def __init__(self, actions):
        self.replayMemory = deque()
        self.timeStep = 0
        self.epsilon = EPSILON1
        self.actions = actions
        self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_fc3, self.b_fc3 = self.createQNW()

        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T, self.W_fc3T, self.b_fc3T = self.createQNW()

        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                            self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),
                                            self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),
                                            self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2),
                                            self.W_fc3T.assign(self.W_fc3), self.b_fc3T.assign(self.b_fc3)]

        self.createTrainingMethod()

        # first save the Q network and then load it 
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Network not found")


    def createQNW(self):

        W_conv1 = self.weight_var([8, 8, 4, 32])
        b_conv1 = self.bias_var([32])

        W_conv2 = self.weight_var([4, 4, 32, 64])
        b_conv2 = self.bias_var([64])

        W_conv3 = self.weight_var([3, 3, 64, 64])
        b_conv3 = self.bias_var([64])

        W_fc1 = self.weight_var([1600, 512])
        b_fc1 = self.bias_var([512])

        W_fc2 = self.weight_var([512, 100])
        b_fc2 = self.bias_var([100])

        W_fc3 = self.weight_var([100, self.actions])
        b_fc3 = self.bias_var([self.actions])

        stateIP = tf.placeholder("float", [None, 80, 80, 4])

        h_conv1 = tf.nn.relu(self.conv2d(stateIP, W_conv1, 4) + b_conv1)

        h_pool1 = self.avg_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        QValue = tf.matmul(h_fc2, W_fc3) + b_fc3

        return stateIP, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3


    def getAction(self):
        QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState]})[0]
        action = np.zeros(self.actions)
        if self.timeStep % FPS == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1

        if self.epsilon > EPSILON2 and self.timeStep > OBSERVE_TIMESTEPS:
            self.epsilon -= (EPSILON1 - EPSILON2) / EXPLORE

        return action

    def setInitState(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis=2)

    def weight_var(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_var(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def avg_pool_2x2(self, x):
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


    def copyTargetQNW(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def trainQNW(self):

        # get random minibatch from memory
        minibatch = random.sample(self.replayMemory, BATCH)
        state1Batch = [data[0] for data in minibatch]
        actionBatch = [data[1] for data in minibatch]
        rewardBatch = [data[2] for data in minibatch]
        state2Batch = [data[3] for data in minibatch]

        # y calculation
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: state2Batch})
        for i in range(0, BATCH):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(rewardBatch[i])
            else:
                y_batch.append(rewardBatch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: actionBatch,
            self.stateInput: state1Batch
        })

        # save network for every 100000 iteration
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step=self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNW()

    def setPerception(self, nextObservation, action, reward, terminal):
        newState = np.append(self.currentState[:, :, 1:], nextObservation, axis=2)
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE_TIMESTEPS:
            # Train network
            self.trainQNW()

        state = ""
        if self.timeStep <= OBSERVE_TIMESTEPS:
            state = "observe"
        elif self.timeStep > OBSERVE_TIMESTEPS and self.timeStep <= OBSERVE_TIMESTEPS + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.timeStep, "/ STATE", state,
              "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

