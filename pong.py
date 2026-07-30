import math
import random

# game constants
W = 80
H = 24
PADDLE_H = 5
PADDLE_SPEED = 0.6
BALL_SPEED = 0.9
WIN_SCORE = 11

# precalculated constants so we never recompute them
_HALF_PADDLE = PADDLE_H * 0.5
_PADDLE_MIN = _HALF_PADDLE
_PADDLE_MAX = H - _HALF_PADDLE
_HALF_W = W * 0.5
_HALF_H = H * 0.5
_W_MINUS_2 = W - 2
_INV_HALF_PADDLE = 1.0 / _HALF_PADDLE
_PI_4 = math.pi * 0.25


class Pong:
    __slots__ = (
        'ball_speed', 'paddle_speed', 'ball_x', 'ball_y',
        'ball_dx', 'ball_dy', 'paddle1_y', 'paddle2_y',
        'score1', 'score2', 'hits', 'steps', 'done',
    )

    def __init__(self, ball_speed=0.9, paddle_speed=0.6):
        self.ball_speed = ball_speed
        self.paddle_speed = paddle_speed
        self.reset()

    def reset(self):
        # ball position
        self.ball_x = _HALF_W
        self.ball_y = _HALF_H
        # random angle for serve
        angle = random.uniform(-_PI_4, _PI_4)
        self.ball_dx = math.cos(angle) * self.ball_speed
        self.ball_dy = math.sin(angle) * self.ball_speed
        # paddles start in the middle
        self.paddle1_y = _HALF_H
        self.paddle2_y = _HALF_H
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

        inv_h = 1.0 / H
        inv_bs = 1.0 / self.ball_speed
        inv_w = 1.0 / W
        return [
            (self.ball_y - py) * inv_h,
            self.ball_dy * inv_bs,
            bdx * inv_bs,
            self.ball_x * inv_w,
        ]

    # run 1 frame. action1/action2 are -1 (up), 0 (stay), 1 (down)
    def step(self, action1, action2):
        self.steps += 1

        # move paddles - use local vars to avoid repeated attribute lookups
        p1y = self.paddle1_y + action1 * self.paddle_speed
        p2y = self.paddle2_y + action2 * self.paddle_speed

        # clamp paddles
        if p1y < _PADDLE_MIN:
            p1y = _PADDLE_MIN
        elif p1y > _PADDLE_MAX:
            p1y = _PADDLE_MAX
        if p2y < _PADDLE_MIN:
            p2y = _PADDLE_MIN
        elif p2y > _PADDLE_MAX:
            p2y = _PADDLE_MAX

        self.paddle1_y = p1y
        self.paddle2_y = p2y

        # move ball - use locals
        bx = self.ball_x + self.ball_dx
        by = self.ball_y + self.ball_dy
        bdx = self.ball_dx
        bdy = self.ball_dy

        # bounce off top/bottom walls
        if by <= 0:
            by = 0
            if bdy < 0:
                bdy = -bdy
        elif by >= H:
            by = H
            if bdy > 0:
                bdy = -bdy

        # check paddle 1 hit (left side)
        if bx <= 2 and bdx < 0 and abs(by - p1y) <= _HALF_PADDLE:
            bdx = -bdx  # reverse direction
            # strategic bounce based on paddle offset
            offset = (by - p1y) * _INV_HALF_PADDLE
            bdy = offset * 0.8
            bx = 2
            # normalize speed only when angle changes
            speed = math.sqrt(bdx * bdx + bdy * bdy)
            inv_speed = self.ball_speed / speed
            bdx *= inv_speed
            bdy *= inv_speed

        # check paddle 2 hit (right side)
        if bx >= _W_MINUS_2 and bdx > 0 and abs(by - p2y) <= _HALF_PADDLE:
            bdx = -bdx
            offset = (by - p2y) * _INV_HALF_PADDLE
            bdy = offset * 0.8
            bx = _W_MINUS_2
            self.hits += 1
            speed = math.sqrt(bdx * bdx + bdy * bdy)
            inv_speed = self.ball_speed / speed
            bdx *= inv_speed
            bdy *= inv_speed

        # write back locals
        self.ball_x = bx
        self.ball_y = by
        self.ball_dx = bdx
        self.ball_dy = bdy

        # check if someone scored
        if bx < 0:
            self.score2 += 1
            self._reset_ball(False)
            if self.score2 >= WIN_SCORE:
                self.done = True
        elif bx > W:
            self.score1 += 1
            self._reset_ball(True)
            if self.score1 >= WIN_SCORE:
                self.done = True

    # put the ball back in the center after a point
    def _reset_ball(self, serve_right):
        self.ball_x = _HALF_W
        self.ball_y = _HALF_H
        angle = random.uniform(-_PI_4, _PI_4)
        self.ball_dx = math.cos(angle) * self.ball_speed
        self.ball_dy = math.sin(angle) * self.ball_speed
        if serve_right:
            self.ball_dx = abs(self.ball_dx)
        else:
            self.ball_dx = -abs(self.ball_dx)

    # draw the game in the terminal with colors
    def render(self):
        # build the entire frame as one string, then print once
        # this is WAY faster than calling print() for every single line
        C_RESET = "\033[0m"
        C_P1 = "\033[94m"  # Blue
        C_P2 = "\033[91m"  # Red
        C_BALL = "\033[92m"  # Green
        C_BORDER = "\033[90m"  # Gray

        # precompute ball and paddle positions as ints for comparison
        ball_ix = int(self.ball_x + 0.5)
        ball_iy = int(self.ball_y + 0.5)
        p1_min = int(self.paddle1_y - _HALF_PADDLE)
        p1_max = int(self.paddle1_y + _HALF_PADDLE)
        p2_min = int(self.paddle2_y - _HALF_PADDLE)
        p2_max = int(self.paddle2_y + _HALF_PADDLE)
        center_x = W // 2

        lines = [f"\033[H{C_BORDER}+{'-' * W}+{C_RESET}"]
        for y in range(H):
            row = [f"{C_BORDER}|{C_RESET}"]
            for x in range(W):
                # left paddle
                if x == 1 and p1_min <= y <= p1_max:
                    row.append(f"{C_P1}#{C_RESET}")
                # right paddle
                elif x == _W_MINUS_2 and p2_min <= y <= p2_max:
                    row.append(f"{C_P2}#{C_RESET}")
                # ball
                elif x == ball_ix and y == ball_iy:
                    row.append(f"{C_BALL}O{C_RESET}")
                # center line
                elif x == center_x and y % 2 == 0:
                    row.append(f"{C_BORDER}|{C_RESET}")
                else:
                    row.append(" ")
            row.append(f"{C_BORDER}|{C_RESET}")
            lines.append("".join(row))
        lines.append(f"{C_BORDER}+{'-' * W}+{C_RESET}")
        lines.append(f"  {C_P1}P1: {self.score1}{C_RESET}   -   {C_P2}P2: {self.score2}{C_RESET}   (hits: {self.hits})")

        # single write to stdout is way faster than dozens of print() calls
        import sys
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
