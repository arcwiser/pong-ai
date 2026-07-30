import math
import random

# game constants
W = 80
H = 24
PADDLE_H = 5
PADDLE_SPEED = 0.6
BALL_SPEED = 0.9
WIN_SCORE = 11


class Pong:
    def __init__(self, ball_speed=0.9, paddle_speed=0.6):
        self.ball_speed = ball_speed
        self.paddle_speed = paddle_speed
        self.reset()

    def reset(self):
        # ball position
        self.ball_x = W / 2.0
        self.ball_y = H / 2.0
        # random angle for serve
        angle = random.uniform(-math.pi / 4, math.pi / 4)
        self.ball_dx = math.cos(angle) * self.ball_speed
        self.ball_dy = math.sin(angle) * self.ball_speed
        # paddles start in the middle
        self.paddle1_y = H / 2.0
        self.paddle2_y = H / 2.0
        self.score1 = 0
        self.score2 = 0
        self.hits = 0  # how many times player 2 hit the ball
        self.steps = 0
        self.done = False

    # get inputs for the neural network (normalized)
    def get_state(self, for_player=2):
        if for_player == 2:
            py = self.paddle2_y
            bdx = -self.ball_dx  # flip so positive = coming toward
        else:
            py = self.paddle1_y
            bdx = self.ball_dx

        return [
            (self.ball_y - py) / H,
            self.ball_dy / self.ball_speed,
            bdx / self.ball_speed,
            self.ball_x / W,
        ]

    # run 1 frame. action1/action2 are -1 (up), 0 (stay), 1 (down)
    def step(self, action1, action2):
        self.steps += 1

        # move paddles
        self.paddle1_y += action1 * self.paddle_speed
        self.paddle2_y += action2 * self.paddle_speed
        # keep paddles in bounds
        self.paddle1_y = max(PADDLE_H / 2, min(H - PADDLE_H / 2, self.paddle1_y))
        self.paddle2_y = max(PADDLE_H / 2, min(H - PADDLE_H / 2, self.paddle2_y))

        # move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # bounce off top/bottom walls
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_dy = abs(self.ball_dy)
        elif self.ball_y >= H:
            self.ball_y = H
            self.ball_dy = -abs(self.ball_dy)

        # check paddle 1 hit (left side)
        if self.ball_x <= 2 and self.ball_dx < 0 and abs(self.ball_y - self.paddle1_y) <= PADDLE_H / 2:
            self.ball_dx = abs(self.ball_dx)
            self.ball_dy += random.uniform(-0.2, 0.2)
            self.ball_x = 2
            # normalize speed only when changed
            speed = math.sqrt(self.ball_dx ** 2 + self.ball_dy ** 2)
            self.ball_dx = self.ball_dx / speed * self.ball_speed
            self.ball_dy = self.ball_dy / speed * self.ball_speed

        # check paddle 2 hit (right side)
        if self.ball_x >= W - 2 and self.ball_dx > 0 and abs(self.ball_y - self.paddle2_y) <= PADDLE_H / 2:
            self.ball_dx = -abs(self.ball_dx)
            self.ball_dy += random.uniform(-0.2, 0.2)
            self.ball_x = W - 2
            self.hits += 1  # track how many returns player 2 does
            # normalize speed only when changed
            speed = math.sqrt(self.ball_dx ** 2 + self.ball_dy ** 2)
            self.ball_dx = self.ball_dx / speed * self.ball_speed
            self.ball_dy = self.ball_dy / speed * self.ball_speed

        # check if someone scored
        if self.ball_x < 0:
            self.score2 += 1
            self._reset_ball(serve_right=False)
            if self.score2 >= WIN_SCORE:
                self.done = True
        elif self.ball_x > W:
            self.score1 += 1
            self._reset_ball(serve_right=True)
            if self.score1 >= WIN_SCORE:
                self.done = True

    # put the ball back in the center after a point
    def _reset_ball(self, serve_right):
        self.ball_x = W / 2.0
        self.ball_y = H / 2.0
        angle = random.uniform(-math.pi / 4, math.pi / 4)
        self.ball_dx = math.cos(angle) * self.ball_speed
        self.ball_dy = math.sin(angle) * self.ball_speed
        if serve_right:
            self.ball_dx = abs(self.ball_dx)
        else:
            self.ball_dx = -abs(self.ball_dx)

    # draw the game in the terminal with colors
    def render(self):
        import os
        os.system("cls" if os.name == "nt" else "clear")
        
        # ANSI Colors
        C_RESET = "\033[0m"
        C_P1 = "\033[94m" # Blue
        C_P2 = "\033[91m" # Red
        C_BALL = "\033[92m" # Green
        C_BORDER = "\033[90m" # Gray
        
        print(f"{C_BORDER}+{'-' * W}+{C_RESET}")
        for y in range(H):
            line = f"{C_BORDER}|{C_RESET}"
            for x in range(W):
                ch = " "
                # left paddle
                if x == 1 and abs(y - self.paddle1_y) <= PADDLE_H / 2:
                    ch = f"{C_P1}#{C_RESET}"
                # right paddle
                elif x == W - 2 and abs(y - self.paddle2_y) <= PADDLE_H / 2:
                    ch = f"{C_P2}#{C_RESET}"
                # ball
                elif abs(x - self.ball_x) < 0.6 and abs(y - self.ball_y) < 0.6:
                    ch = f"{C_BALL}O{C_RESET}"
                # center line
                elif x == W // 2 and y % 2 == 0:
                    ch = f"{C_BORDER}|{C_RESET}"
                line += ch
            line += f"{C_BORDER}|{C_RESET}"
            print(line)
        print(f"{C_BORDER}+{'-' * W}+{C_RESET}")
        print(f"  {C_P1}P1: {self.score1}{C_RESET}   -   {C_P2}P2: {self.score2}{C_RESET}   (hits: {self.hits})")
