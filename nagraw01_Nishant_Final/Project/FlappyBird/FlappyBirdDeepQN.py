import cv2
import sys
sys.path.append("game/")

import wrapped_flappy_bird as game

from DeepQN import DQN
import numpy as np

# pre-processing to a 80*80 gray scale image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (80, 80, 1))


def playFlappyBird():
    actions = 2
    deepqn = DQN(actions)
    flappyBird = game.GameState()
    action0 = np.array([1, 0])
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    deepqn.setInitState(observation0)

    while 1 != 0:
        action = deepqn.getAction()
        nextObservation, reward, terminal = flappyBird.frame_step(action)
        nextObservation = preprocess(nextObservation)
        deepqn.setPerception(nextObservation, action, reward, terminal)


def main():
    playFlappyBird()


if __name__ == '__main__':
    main()
