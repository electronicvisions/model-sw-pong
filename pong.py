import numpy as np
from random import randint, choice
from time import time, sleep
from threading import Thread

LEFT_WIN = -1
RIGHT_WIN = +1
NO_WIN = 0

MOVE_UP = +1
MOVE_DOWN = -1
DONT_MOVE = 0


class BallOrPuck:
    """Class that represents either a ball or a puck.

        Args:
            game (GameOfPong): Pong game.
            x_pos (float): Initial x position in unit length.
            y_pos (float): Initial y position in unit length.
            velocity (float): Change in position per iteration.
            direction (list, float): Heading.

    """

    def __init__(self, game,
                 x_pos=0.5,
                 y_pos=0.5,
                 velocity=1 / 5.,
                 direction=0):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.velocity = velocity
        self.direction = direction
        self.game = game
        self.update_cell()

    def get_cell(self):
        return self.cell

    def update_cell(self):
        """Update cell based on position.
        """
        x_cell = int(np.floor(
            (self.x_pos / self.game.x_length) * self.game.x_grid))
        y_cell = int(np.floor(
            (self.y_pos / self.game.y_length) * self.game.y_grid))
        self.cell = [x_cell, y_cell]


class Ball(BallOrPuck):
    """Class representing the ball.
    
        Args:
            radius (float): Radius of ball in unit length.

        For other args, see :class:`BallOrPuck`.
    """

    def __init__(self, game,
                 x_pos=0.5,
                 y_pos=0.5,
                 velocity=0.025,
                 direction=[-1 / 2., 1 / 2.],
                 radius=0.02):
        BallOrPuck.__init__(self, game, x_pos, y_pos, velocity, direction)
        self.ball_radius = radius  # unit length
        self.update_cell()


class Puck(BallOrPuck):
    """Class representing the puck.

        Args:
            direction (float, int): +1 for up, -1 for down, 0 for no movement.

        For other args, see :class:`BallOrPuck`.
    """

    def __init__(self, game,
                 length=0.07,
                 y_pos=0.5,
                 velocity=0.05,
                 direction=0.):
        BallOrPuck.__init__(self, game, game.x_length, y_pos, velocity,
                            direction)
        self.paddle_length = length  # unit length
        self.update_cell()


class GameOfPong(object):
    """Class representing a game of Pong. Playing field: 1 by 1 discretized into cells.

        Args:
            x_grid (int): Number of cells to discretize x-axis into.
            y_grid (int): Number of cells to discretize y-axis into.
            debug (bool): Print debugging messages.
            norm (string): "L1" or "L2", defines norm to be used for velocity vector.
    """

    def __init__(self, x_grid=32, y_grid=32, debug=False, norm="L1"):
        self.x_length = 1  # length in x-direction in unit length
        self.y_length = 1  # length in y-direction in unit length
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.right_paddle = Puck(self
                                 )  # AI paddle, center position along y-axis
        if norm == "L2":
            initial_angle = choice([0., 180.]) + randint(
                -45, 45)  # random initial direction toward either end
            initial_vx = np.cos(initial_angle / 360. * 2 * np.pi)
            initial_vy = np.sin(initial_angle / 360. * 2 * np.pi)
        elif norm == "L1":
            initial_vx = 0.5 + 0.5 * np.random.random()
            initial_vy = 1. - initial_vx
            initial_vx *= choice([-1., 1.])
            initial_vy *= choice([-1., 1.])
        self.ball = Ball(self, direction=[initial_vx, initial_vy])
        self.result = False
        self.debug = debug

    def update_ball_direction(self):
        """In case of a collision, update the direction of ball velocity. Also determine if the ball is in either player's net.

        Returns:
            Either NO_WIN or RIGHT_WIN.
        """
        if self.ball.y_pos + self.ball.ball_radius >= self.y_length:  # upper edge
            self.ball.direction[1] = -self.ball.direction[1]
            return NO_WIN
        if self.ball.y_pos - self.ball.ball_radius <= 0:  # lower edge
            self.ball.direction[1] = -self.ball.direction[1]
            return NO_WIN
        # left paddle/wall
        if self.ball.x_pos - self.ball.ball_radius <= 0:
            self.ball.direction[0] = -self.ball.direction[0]
            return NO_WIN
        # right paddle
        if self.ball.x_pos + self.ball.ball_radius >= self.x_length:
            # check if win for right player
            if self.right_paddle.y_pos - self.right_paddle.paddle_length / 2 <= self.ball.y_pos <= self.right_paddle.y_pos + self.right_paddle.paddle_length / 2:
                self.ball.direction[0] = -self.ball.direction[0]
                return NO_WIN
            else:
                return RIGHT_WIN
        return NO_WIN

    def propagate_ball_and_paddles(self):
        """Update ball and paddle coordinates based on direction and velocity. Also update their cells.
        """
        self.right_paddle.y_pos += self.right_paddle.direction * self.right_paddle.velocity
        if self.right_paddle.y_pos < 0:
            self.right_paddle.y_pos = 0
        if self.right_paddle.y_pos > self.y_length:
            self.right_paddle.y_pos = self.y_length
        self.ball.y_pos += self.ball.velocity * self.ball.direction[1]
        self.ball.x_pos += self.ball.velocity * self.ball.direction[0]
        self.ball.update_cell()
        self.right_paddle.update_cell()

    def move_right_paddle_up(self):
        self.right_paddle.direction = MOVE_UP

    def move_right_paddle_down(self):
        self.right_paddle.direction = MOVE_DOWN

    def dont_move_right_paddle(self):
        self.right_paddle.direction = DONT_MOVE

    def get_right_paddle_position(self):
        return self.right_paddle.y_pos

    def get_right_paddle_length(self):
        return self.right_paddle.paddle_length

    def get_right_paddle_cell(self):
        return self.right_paddle.get_cell()

    def get_ball_cell(self):
        return self.ball.get_cell()

    def step(self):
        """Perform one game step.
        """
        ball_status = self.update_ball_direction()
        self.propagate_ball_and_paddles()
        if ball_status != NO_WIN:
            self.result = ball_status
        return ball_status
