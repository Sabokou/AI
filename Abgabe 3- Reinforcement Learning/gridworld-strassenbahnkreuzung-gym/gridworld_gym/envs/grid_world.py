import random
import time
from abc import ABC
from collections import deque
from typing import Tuple, Union, Optional

import gym
import numpy as np
from gym import spaces

import sys

from gridworld_gym.envs.helper import Signal, Switch, Stop
from gridworld_gym.envs.tram import Tram


class GridWorldEnv(gym.Env, ABC):
    """The schematic of Mannheim's central metro system. It is simplified into a gridworld and slightly altered."""

    def __init__(self, config=False):
        """
        Initializes all grid components with automatically generated components. Also sets the world steps to 0 as
        initial value.
        """
        super(GridWorldEnv, self).__init__()
        # Grid components
        self.grid = None
        self.tram_grid = None
        self.world_step = 0
        self._init_grid()

        # Gym specific variables
        self.max_episode_steps = 400
        self.state = dict()
        self.action_space = spaces.Tuple([spaces.Discrete(4), spaces.Discrete(5)])
        self.observation_space = spaces.Dict({"signal_cluster_links": spaces.Box(low=-np.inf, high=np.inf,
                                                                                 shape=(3,), dtype=np.float32),
                                              "signal_cluster_rechts": spaces.Box(low=-np.inf, high=np.inf,
                                                                                  shape=(4,), dtype=np.float32)})

    def step(self, action):
        """
        Overwritten gym internal function used to progress a step.
        """
        # add new trains to the world
        self._add_lines()
        # update the world according to actions 
        self._update_signal(action)
        # simulate a step in the world and calculate reward
        reward = self._update_world()
        # convert grid world to observation space
        obs_state = self._convert_to_observation_space()
        # simulate at max 800 steps before returning status done = True
        if self.world_step > 800:
            done = True
        else:
            done = False
        # as additional information generate the average delay of all trains
        average_delay = self._return_average_delay()
        return obs_state, reward, done, average_delay

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # reinitialize the grids to reset them
        self._init_grid()
        self.world_step = 0
        # need to convert grid to observation space to confirm to expected function behaviour
        obs_state = self._convert_to_observation_space()
        return obs_state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        output = ""
        # iterate over all rows
        for y, row in enumerate(self.grid):
            # iterate over all elements in the row
            for x, col in enumerate(row):
                # if there is no tram on the grid coordinate look into the normal grid
                if self.tram_grid[y][x] == 0:
                    if col == 0:
                        output += "  "
                    elif col in ["-", "|", "/"]:
                        output += 2 * col
                    elif col == "\\":
                        output += col * 2
                    elif type(col) == Signal:
                        output += "S" + str(col.status)
                    elif type(col) == Switch:
                        output += "W" + col.status_switched
                    elif type(col) == Stop:
                        output += "SP"  # col.name[:2]
                # if there is a tram on this grid coordinate print the tram instead
                else:
                    output += str(self.tram_grid[y][x].line_number) + self.tram_grid[y][x].direction
            output += "\n"
        return output

    def add_tram_to_grid(self, x: int, y: int, tram: Tram):
        """
        Adds a tram based on initial coordinates into the world by adding it to the tram_grid. It has no logic to check
        if a tram already occupies the coordinate.
        :param x: integer between 0 and the length of a grid row
        :param y: integer between 0 and the height of the grid
        :param tram: tram object that is to be placed
        :return: current grid step
        """
        self.tram_grid[y][x] = tram
        return self.world_step

    def _return_average_delay(self):
        delay = 0
        amount_trams = 0
        for row in self.tram_grid:
            for col in row:
                if type(col) == Tram:
                    delay += col.delay
                    amount_trams += 1

        delay = (delay / amount_trams) if amount_trams else 0

        return {"average_delay": delay}

    def _convert_to_observation_space(self):
        """
        Converts the grid space to the observation space. The observation is given by the delay a train has at the given
        signal position. The default value if no train is present at the signal is -100 representing that there is an
        imaginary train with a time buffer of 100 minutes at the signal.
        """
        # Get delays at left signal cluster
        # Southern signal
        if type(self.tram_grid[7][5]) == Tram:
            delay1 = self.tram_grid[7][5].delay
        else:
            delay1 = -100
        # Western signal
        if type(self.tram_grid[5][2]) == Tram:
            delay2 = self.tram_grid[5][2].delay
        else:
            delay2 = -100
        # Eastern signal
        if type(self.tram_grid[4][6]) == Tram:
            delay3 = self.tram_grid[4][6].delay
        else:
            delay3 = -100
        # any box space is representable by a np array conforming to the defined dimensions and min / max values
        self.state["signal_cluster_links"] = np.array([delay1, delay2, delay3], dtype=np.float32)

        # Get delays at right signal cluster 
        # Northern Signal
        if type(self.tram_grid[2][14]) == Tram:
            delay1 = self.tram_grid[2][14].delay
        else:
            delay1 = -100
        # Southern signal
        if type(self.tram_grid[7][15]) == Tram:
            delay2 = self.tram_grid[7][15].delay
        else:
            delay2 = -100
        # Western signal
        if type(self.tram_grid[5][12]) == Tram:
            delay3 = self.tram_grid[5][12].delay
        else:
            delay3 = -100
        # Eastern signal
        if type(self.tram_grid[4][17]) == Tram:
            delay4 = self.tram_grid[4][17].delay
        else:
            delay4 = -100
        self.state["signal_cluster_rechts"] = np.array([delay1, delay2, delay3, delay4], dtype=np.float32)

        return self.state

    def _init_grid(self):
        """
        Sets all grids to their respective default.

        :param: grid
        The grid is defined as a 2-dimensional list of element.
        An empty spot where no tracks exist is represented as 0. Vertical tracks - allowing vertical travel are
        represented by the string |.
        Horizontal tracks are given by '-'. Diagonal travel is represented by either \\ or /.
        Signals, stops and switches are integrated as objects of their respective data types.

        :param: tram_grid
        The tram grid is used to track the individual tram objects on their path through the tracks. It has to be off
        the same dimensionality as the default grid.

        :return: None
        """
        self.grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "|", "|", 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "|", "|", 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Signal(), Stop(), 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Switch("|", "/"), "|", 0, 0, 0, 0],
                     ["-", "-", "-", Stop(), "-", Switch("-", "/"), Signal(), "-", "-", "-", "-", "-", "-", "-", "-",
                      "-", Switch("-", "\\"), Signal(), "-", "-"],
                     ["-", "-", Signal(), Switch("-", "\\"), "-", "-", "-", Stop(), "-", "-", "-", "-", Signal(),
                      Switch("-", "\\"), Switch("-", "/"), Switch("-", "\\"), "-", Stop(), "-", "-"],
                     [0, 0, 0, 0, "|", Switch("|", "/"), 0, 0, 0, 0, 0, 0, 0, 0, "|", Switch("|", "/"), 0, 0, 0, 0],
                     [0, 0, 0, 0, Stop(), Signal(), 0, 0, 0, 0, 0, 0, 0, 0, Stop(), Signal(), 0, 0, 0, 0],
                     [0, 0, 0, 0, "|", "|", 0, 0, 0, 0, 0, 0, 0, 0, "|", "|", 0, 0, 0, 0],
                     [0, 0, 0, 0, "|", "|", 0, 0, 0, 0, 0, 0, 0, 0, "|", "|", 0, 0, 0, 0]]
        self.tram_grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    def _update_tram(self, *args):
        """
        Recursive function to update all subsequent trams. It checks if there is a tram object on the proposed new
        coordinate, and it already moved on this grid step. If it hasn't moved the function update_tram is called for
        this tram. The exit conditions are either there being no tram on the track segment where the original tram
        object wants to move to or the tram on the tile "ahead" already moved this step.
        :param args: new x & y value of the tram, the calculated reward in move function and the tram object
        :return: float reward
        """

        if type(args[0]) == tuple:
            # list unpacking of args
            new_x, new_y, new_direction, reward, tile = args[0]
            count = args[1]
        else:
            new_x, new_y, new_direction, reward, tile = args[0][0].read_track()
            count = args[0][1]
            # workaround for infinite recursion
            if count >= 15:
                return 0

        # check if tram goes out-of-bounds and deletes it
        if new_x >= len(self.grid[0]) or new_x < 0 or new_y < 0 or new_y >= len(self.grid):
            # remove tram from tram_grid
            tile.grid.tram_grid[tile.y][tile.x] = 0
            # delete tram object
            del tile
            return reward
        else:
            # calculates the new position of the tram
            expected_new_tile = self.tram_grid[new_y][new_x]

        # Change world step if condition 2
        # check which condition applies to the new position
        # Condition 1: tram on new position that hasn't moved on this step
        if type(expected_new_tile) == Tram:
            if expected_new_tile.world_step != self.world_step:
                if expected_new_tile != tile:
                    # recursive call for the tram on new position
                    reward = self._update_tram([expected_new_tile, count + 1])
                    # move the original out of recursion tram
                    expected_new_tile.move(new_x, new_y, new_direction)
                else:
                    reward = 0
                return reward
            # Condition 2: tram on new position that has already moved this step
            elif expected_new_tile.world_step == self.world_step:
                tile.world_step = self.world_step
                expected_new_tile.move(new_x, new_y, new_direction)
                return reward - 1
        # Condition 3: Nothing on the new position
        else:
            tile.move(new_x, new_y, new_direction)
            return reward + 0

    def _update_world(self):
        self.world_step += 1
        reward = 0
        for row in self.tram_grid:
            for tile in row:
                if type(tile) == Tram and tile.world_step != self.world_step:
                    reward += self._update_tram(tile.read_track(), 0)
        reward = max(reward, -100)
        return reward

    def _update_signal(self, action_list):
        """
        [0,0,2,3,4,3,2]
        :param action_list: list of 7 integers
        :return:
        """
        # set signals based on discrete action
        signal_positions = [[self.grid[7][5], self.grid[4][6], self.grid[5][2]],
                            [self.grid[2][14], self.grid[7][15], self.grid[4][17], self.grid[5][2]]]
        for cnt, element in enumerate(action_list):
            for signal in signal_positions[cnt]:
                try:
                    signal.turn_red()
                except AttributeError:
                    print(signal_positions[cnt])
            if element != 0:
                signal_positions[cnt][element - 1].turn_green()

    def _create_line(self, line_number: int, reverse: bool):
        """
        Creates a tram line based on the line number and its tracking direction.
        :param line_number:
        :param reverse:
        """
        delay = random.randint(-3, 3)
        # Line 1: left to right
        if line_number == 1:
            # entry left side
            switches = deque(["-", "-"])
            if reverse is False:
                line = Tram(start_x=0, start_y=5, direction=">", grid=self, switches=switches, delay=delay, line=1)
            # entry right side
            else:
                line = Tram(start_x=19, start_y=4, direction="<", grid=self, switches=switches, delay=delay, line=1)

        # Line 2: left to bottom right
        elif line_number == 2:
            # entry left side
            if reverse is False:
                switches = deque(["-", "\\"])
                line = Tram(start_x=0, start_y=5, direction=">", grid=self, switches=switches, delay=delay, line=2)
            # entry bottom right side
            else:
                switches = deque(["|", "\\", "-"])
                line = Tram(start_x=15, start_y=9, direction="^", grid=self, switches=switches, delay=delay, line=2)
        # Line 3: Top right from bottom left
        elif line_number == 3:
            # entry top right side
            if reverse is False:
                switches = deque(["/", "/"])
                line = Tram(start_x=14, start_y=0, direction="v", grid=self, switches=switches, delay=delay, line=3)
            # entry bottom left side
            else:
                switches = deque(["/", "-", "/"])
                line = Tram(start_x=5, start_y=9, direction="^", grid=self, switches=switches, delay=delay, line=3)

        else:
            raise NotImplemented
        return line

    def _add_lines(self):
        if self.world_step % 5 == 0:
            pass

        elif self.world_step % 5 == 1:
            self._create_line(1, False)
            self._create_line(2, True)

        elif self.world_step % 5 == 2:
            self._create_line(1, True)

        elif self.world_step % 5 == 3:
            self._create_line(3, True)
            self._create_line(1, False)
        elif self.world_step % 5 == 4:
            self._create_line(3, False)
            self._create_line(2, False)
