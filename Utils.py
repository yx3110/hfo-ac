from hfo import *
import numpy as np


class ExpBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.cur_size = 0

    def add(self, experience):
        if self.cur_size >= self.buffer_size:
            self.buffer.pop(0)
            self.cur_size -= 1
        self.buffer.append(experience)
        self.cur_size += 1

    def sample(self, size):
        res = []
        for i in range(size):
            res.append(self.buffer[np.random.randint(0, self.cur_size - 1)])
        return res


class Experience:
    def __init__(self, state0, state1, action, reward, done):
        self.state0 = state0
        self.state1 = state1
        self.action = action
        self.reward = reward
        self.done = done


class Action:
    def __init__(self):
        self.action = 0
        self.param1 = 0
        self.param2 = 0


def get_action(action_arr):
    res = Action()
    res.action = 0

    action_arr_copy = []
    for i in xrange(len(action_arr)):
        action_arr_copy.append(action_arr[i])
    cur_max = action_arr_copy[0]
    action_arr_copy[2] = -99999
    for i in xrange(0, 3):
        if action_arr_copy[i] >= cur_max:
            res.action = i
    if res.action == 0:
        res.param1 = action_arr[4]
        res.param2 = action_arr[5]
    elif res.action == 1:
        res.param1 = action_arr[6]
        res.param2 = 0
    elif res.action == 3:
        res.param1 = action_arr[8]
        res.param2 = action_arr[9]
    return res


def take_action(env, action):
    if action.action == 0 or action.action == 3:
        env.act(action.action, action.param1, action.param2)
    else:
        env.act(action.action, action.param1)
