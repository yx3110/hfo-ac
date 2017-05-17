import numpy as np
from hfo import *


class Experience:
    def __init__(self, prev_state, cur_state, action, reward, done):
        self.prev_state = prev_state
        self.cur_state = cur_state
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
    copy = np.copy(action_arr)
    copy[2] = float(-999999999999)
    res.action = 0
    cur_max = action_arr[0]
    for i in range(0, 3):
        if copy[i] > cur_max:
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


def calculate_reward(state1, state2, game_status, team_size=1, opponent_size=0):
    ball_dist1 = []
    ball_dist2 = []
    goal_dist1 = []
    goal_dist2 = []
    feature_size = 58 + 8 * (team_size - 1) + opponent_size * 8
    for i in range(team_size):
        ball_dist1.append(1.0 - state1[53 + i * feature_size])
        ball_dist2.append(1.0 - state2[53 + i * feature_size])
        goal_dist1.append(1.0 - state1[15 + i * feature_size])
        goal_dist2.append(1.0 - state2[15 + i * feature_size])

    ball_ang_sin_rad1 = state1[51]
    ball_ang_sin_rad2 = state2[51]
    ball_ang_cos_rad1 = state1[52]
    ball_ang_cos_rad2 = state2[52]

    ball_ang_rad1 = np.arccos(ball_ang_cos_rad1)
    ball_ang_rad2 = np.arccos(ball_ang_cos_rad2)
    if ball_ang_sin_rad1 < 0:
        ball_ang_rad1 *= -1.
    if ball_ang_sin_rad2 < 0:
        ball_ang_rad2 *= -1.

    goal_ang_sin_rad1 = state1[13]
    goal_ang_sin_rad2 = state2[13]
    goal_ang_cos_rad1 = state1[14]
    goal_ang_cos_rad2 = state2[14]

    goal_ang_rad1 = np.arccos(goal_ang_cos_rad1)
    goal_ang_rad2 = np.arccos(goal_ang_cos_rad2)
    if goal_ang_sin_rad1 < 0:
        goal_ang_rad1 *= -1.
    if goal_ang_sin_rad2 < 0:
        goal_ang_rad2 *= -1.

    alpha1 = max(ball_ang_rad1, goal_ang_rad1) - min(ball_ang_rad1, goal_ang_rad1)
    alpha2 = max(ball_ang_rad2, goal_ang_rad2) - min(ball_ang_rad2, goal_ang_rad2)

    ball_dist_goal1 = np.sqrt(
        ball_dist1[0] * ball_dist1[0] + goal_dist1[0] * goal_dist1[0] - 2. * ball_dist1[0] * goal_dist1[0] * np.cos(
            alpha1))
    ball_dist_goal2 = np.sqrt(
        ball_dist2[0] * ball_dist2[0] + goal_dist2[0] * goal_dist2[0] - 2. * ball_dist2[0] * goal_dist2[0] * np.cos(
            alpha2))
    able_to_kick1 = state1[5]
    able_to_kick2 = state2[5]
    kick_reward = 0
    goal_reward = 0
    if game_status == GOAL:
        goal_reward = 10
    if able_to_kick2 == 1 and able_to_kick1 == -1:
        kick_reward = 5

    # temp hack of reward
    return ball_dist2[0] - ball_dist1[0] + kick_reward + 3 * (ball_dist_goal2 - ball_dist_goal1) + goal_reward
