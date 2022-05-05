"""
http://incompleteideas.net/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class MountainCarEnvWithStops(gym.Env):
    """
    ### Description

    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. There are two versions
    of the mountain car domain in gym: one with discrete actions and one with continuous.
    This version is the one with discrete actions.

    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)

    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {University of Cambridge},
        year = {1990}
    }
    ```

    ### Observation Space

    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Num | Observation                                                 | Min                | Max    | Unit |
    |-----|-------------------------------------------------------------|--------------------|--------|------|
    | 0   | position of the car along the x-axis                        | -Inf               | Inf    | position (m) |
    | 1   | velocity of the car                                         | -Inf               | Inf  | position (m) |

    ### Action Space

    There are 3 discrete deterministic actions:

    | Num | Observation                                                 | Value   | Unit |
    |-----|-------------------------------------------------------------|---------|------|
    | 0   | Accelerate to the left                                      | Inf    | position (m) |
    | 1   | Don't accelerate                                            | Inf  | position (m) |
    | 2   | Accelerate to the right                                     | Inf    | position (m) |

    ### Transition Dynamics:

    Given an action, the mountain car follows the following transition dynamics:

    *velocity<sub>t+1</sub> = velocity<sub>t</sub> + (action - 1) * force - cos(3 * position<sub>t</sub>) * gravity*

    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

    where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall. The position is clipped to the range `[-1.2, 0.6]` and velocity is clipped to the range `[-0.07, 0.07]`.


    ### Reward:

    The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is penalised with a reward of -1 for each timestep it isn't at the goal and is not penalised (reward = 0) for when it reaches the goal.

    ### Starting State

    The position of the car is assigned a uniform random value in *[-0.6 , -0.4]*. The starting velocity of the car is always assigned to 0.

    ### Episode Termination

    The episode terminates if either of the following happens:
    1. The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
    2. The length of the episode is 200.


    ### Arguments

    ```
    gym.make('MountainCar-v0')
    ```

    ### Version History

    * v0: Initial versions release (1.0.0)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
        stops=[
            # (state_center, state_width, reward)
            ([ 0.525, 0.035], [0.15, np.infty], 10),
            ([-0.5,   0.   ], [0.2,  0.02    ], 100),
        ],
        observable_RM=True, # include the Reward Machine state in the state
        discrete_action=True, # use discrete action space
    ):
        self.observable_RM = bool(observable_RM)
        self.discrete_action = discrete_action
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        # self.max_position = 0.55 # TODO remove
        self.max_speed = 0.07
        self.stops = [
            # in the form: ([pos_center, vel_center], [pos_width, vel_width], reward)
            (np.asarray(goal_center), np.asarray(goal_width), goal_reward)
            for (goal_center, goal_width, goal_reward) in stops
        ]
        self.power = 0.0015

        self.force = 0.001
        self.gravity = 0.0025

        self.RM_low = 0
        self.RM_high = len(self.stops)
        self.low  = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position,  self.max_speed], dtype=np.float32)
        if self.observable_RM:
            self.low = np.append(self.low, self.RM_low)
            self.high = np.append(self.high, self.RM_high)
        self.screen = None
        self.clock = None
        self.isopen = True

        if self.discrete_action:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(
                low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
            )
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
    
    @property
    def labels(self):
        """return the truth value for each proposition x_i: "The agent is at the i-th goal"."""
        return np.array([
            np.all(np.abs(self.MDP_state-goal_center) <= goal_width/2)
            for (goal_center, goal_width, goal_reward) in self.stops
        ], dtype=bool)
    
    @property
    def state(self):
        state = self.MDP_state
        if self.observable_RM: state = np.append(state, self.RM_state)
        return state

    @property
    def kinetic(self):
        return 0.5 * (self.MDP_state[1] * np.sqrt((3 * np.cos(3 * self.MDP_state[0]))**2+1)) ** 2

    @property
    def potential(self):
        return (self.gravity * (np.sin(3 * self.MDP_state[0])+1))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.MDP_state = np.array([np.random.uniform(low=-.6, high=-.4), 0])
        self.RM_state = 0
        self.thrust = 0
        if return_info: return np.array(self.state, dtype=np.float32), {}
        else:           return np.array(self.state, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.MDP_state
        if self.discrete_action:
            self.thrust = (action - 1) * self.force
            velocity += self.thrust + math.cos(3 * position) * (-self.gravity)
        else:
            force = min(max(action[0], self.min_action), self.max_action)
            velocity += force * self.power - 0.0025 * math.cos(3 * position)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position <= self.min_position and velocity < 0:
            velocity = 0
        if position >= self.max_position-0.05 and velocity > 0:
            velocity = 0
        self.MDP_state = np.array([position, velocity], dtype=np.float32)


        reward = -.1
        goal_center, goal_width, goal_reward = self.stops[self.RM_state]
        if np.all(np.abs(self.MDP_state-goal_center) <= goal_width/2):
            self.RM_state += 1
            reward += goal_reward
        done = (self.RM_state == self.RM_high)

        return self.state, reward, done, {}

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self, mode="human", fps=None):
        import pygame
        from pygame import gfxdraw
        if fps is None: fps = self.metadata["render_fps"]

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.MDP_state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for thrust,c in [
            (-self.thrust, (carwidth / 4, 0)),
            (+self.thrust, (-carwidth / 4, 0)),
        ]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            color = (128, 128, 128)
            # if thrust > 0: color = (128+127*thrust/self.force, 128, 128)
            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), color
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), color
            )

        for goal_i, (goal_center, goal_width, goal_reward) in enumerate(self.stops):
            flagx = int((goal_center[0] - self.min_position) * scale)
            flagy1 = int(self._height(goal_center[0]) * scale)
            flagy2 = flagy1 + 50
            gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

            if   goal_i <  self.RM_state: color = (  0, 255,   0) # past goals
            elif goal_i == self.RM_state: color = (255,   0,   0) # current goal
            else:                         color = (127, 127, 127) # future goals

            gfxdraw.aapolygon(
                self.surf,
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
                color,
            )
            gfxdraw.filled_polygon(
                self.surf,
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
                color,
            )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(fps)
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
