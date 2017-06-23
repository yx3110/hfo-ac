from hfo import *

kPassVelThreshold = -.5


class GameInfo:
    def __init__(self,unum):
        self.got_kickable_reward = False
        self.prev_ball_prox = 0
        self.ball_prox_delta = 0
        self.prev_kickable = 0
        self.kickable_delta = 0
        self.prev_ball_dist_goal = 0
        self.ball_dist_goal_delta = 0
        self.steps = 0
        self.total_reward = 0
        self.extrinsic_reward = 0
        self.status = IN_GAME
        self.episode_over = False
        self.our_unum = unum
        self.prev_player_on_ball = 0
        self.player_on_ball = 0
        self.pass_active = False

    def update(self, hfo):
        status = hfo.step()
        if status != IN_GAME:
            self.episode_over = True
        cur_state = hfo.getState()
        ball_prox = cur_state[53]
        goal_prox = cur_state[15]
        ball_dist = 1.0 - ball_prox
        goal_dist = 1.0 - goal_prox
        kickable = cur_state[12]
        ball_ang_sin_rad = cur_state[51]
        ball_ang_cos_rad = cur_state[52]
        ball_ang_rad = np.arccos(ball_ang_cos_rad)
        if ball_ang_sin_rad < 0:
            ball_ang_rad *= -1.
        goal_ang_sin_rad = cur_state[13]
        goal_ang_cos_rad = cur_state[14]
        goal_ang_rad = np.arccos(goal_ang_cos_rad)
        if goal_ang_sin_rad < 0:
            goal_ang_rad *= -1.
        alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
        ball_dist_goal = np.sqrt(ball_dist * ball_dist + goal_dist * goal_dist - 2. * ball_dist * goal_dist *
                                 np.cos(alpha))

        ball_vel_valid = cur_state[54]
        ball_vel = cur_state[55]
        if ball_vel_valid and ball_vel > kPassVelThreshold:
            self.pass_active = True

        if self.steps > 0:
            self.ball_prox_delta = ball_prox - self.prev_ball_prox
            self.kickable_delta = kickable - self.prev_kickable
            self.ball_dist_goal_delta = ball_dist_goal - self.prev_ball_dist_goal

        self.prev_ball_prox = ball_prox
        self.prev_kickable = kickable
        self.prev_ball_dist_goal = ball_dist_goal
        if self.episode_over:
            self.ball_prox_delta = 0
            self.kickable_delta = 0
            self.ball_dist_goal_delta = 0
        self.prev_player_on_ball = self.player_on_ball
        self.player_on_ball = hfo.playerOnBall()

        self.steps += 1

    def get_reward(self):
        res = 0
        res += self.move_to_ball_reward()
        res += self.kick_reward() * 3
        res += self.pass_reward() * 3
        EOT_reward = self.EOT_reward()
        res += EOT_reward
        self.extrinsic_reward = EOT_reward
        self.total_reward == res
        return res

    def move_to_ball_reward(self):
        reward = 0
        if self.player_on_ball.unum < 0 or self.player_on_ball.unum == self.our_unum:
            reward += self.ball_prox_delta
        if self.kickable_delta >= 1 and not self.got_kickable_reward:
            reward += 1.0
            self.got_kickable_reward = True
        return reward

    def kick_reward(self):
        if self.player_on_ball.unum == self.our_unum:
            return -self.ball_dist_goal_delta
        elif self.got_kickable_reward:
            return 0.2 * -self.ball_dist_goal_delta
        return 0

    def pass_reward(self):
        return 0

    def EOT_reward(self):
        if self.status == GOAL:
            if self.player_on_ball.unum == self.our_unum:
                return 5
            else:
                return 1
        else:
            return 0