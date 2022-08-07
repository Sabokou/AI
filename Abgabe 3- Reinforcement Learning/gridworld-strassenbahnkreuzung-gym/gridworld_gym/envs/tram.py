import random
from gridworld_gym.envs.helper import Switch, Signal, Stop


class Tram:
    def __init__(self, start_x: int, start_y: int, direction: str, line: int, grid, switches,
                 delay: int = 0):
        """
        Initializes the instance with all necessary information.
        """
        self.x = start_x
        self.y = start_y
        self.direction = direction
        self.grid = grid
        self.delay = delay
        self.world_step = self.grid.add_tram_to_grid(self.x, self.y, self)
        self.switches = switches
        self.line_number = line

    def read_track(self):
        """
        Reads the grid symbol at the current coordinates to determine alongside the current direction of movement where
        the tram should be in the next time step.
        """
        grid_symbol = self.grid.grid[self.y][self.x]
        # if the symbol is a switch object, try switching it to the intended state - due to being stuck could result in
        # multiple pops leading to index errors
        if type(grid_symbol) == Switch:
            try:
                grid_symbol.change_status(self.switches.popleft())
            except IndexError:
                pass
            grid_symbol = grid_symbol.status
        # if the symbol is a Signal object get its current status to later determine if it should move or stop
        elif type(grid_symbol) == Signal:
            grid_symbol = grid_symbol.status
        # if the symbol is a Stop object create a reward
        elif type(grid_symbol) == Stop:
            grid_symbol = "10"

        # left / right horizontal
        if grid_symbol == "-":
            if self.direction == "<":
                new_x = self.x - 1
                new_y = self.y
            elif self.direction == ">":
                new_x = self.x + 1
                new_y = self.y
            elif self.direction == "^":
                new_x = self.x
                new_y = self.y - 1
            elif self.direction == "v":
                new_x = self.x
                new_y = self.y + 1
            direction = self.direction

        # up / down vertical
        elif grid_symbol == "|":
            if self.direction == "^":
                new_x = self.x
                new_y = self.y - 1
            elif self.direction == "v":
                new_x = self.x
                new_y = self.y + 1
            elif self.direction == ">":
                new_x = self.x + 1
                new_y = self.y
            elif self.direction == "<":
                new_x = self.x - 1
                new_y = self.y
            direction = self.direction

        # station / reward tile
        elif grid_symbol == "10":
            if self.direction == "^":
                new_x = self.x
                new_y = self.y - 1
            elif self.direction == "v":
                new_x = self.x
                new_y = self.y + 1
            elif self.direction == "<":
                new_x = self.x - 1
                new_y = self.y
            elif self.direction == ">":
                new_x = self.x + 1
                new_y = self.y
            # create a random delay from waiting at the station
            self.delay += random.randint(0, 2)
            direction = self.direction

        # curve left
        elif grid_symbol == "/":
            if self.direction == "<":
                new_x = self.x - 1
                new_y = self.y + 1
                direction = "v"
            elif self.direction == ">":
                new_x = self.x + 1
                new_y = self.y - 1
                direction = "^"
            elif self.direction == "^":
                new_x = self.x + 1
                new_y = self.y - 1
                direction = ">"
            elif self.direction == "v":
                new_x = self.x - 1
                new_y = self.y + 1
                direction = "<"

        # curve right
        elif grid_symbol == "\\":
            if self.direction == "<":
                new_x = self.x - 1
                new_y = self.y - 1
                direction = "^"
            elif self.direction == ">":
                new_x = self.x + 1
                new_y = self.y + 1
                direction = "v"
            elif self.direction == "^":
                new_x = self.x - 1
                new_y = self.y - 1
                direction = "<"
            elif self.direction == "v":
                new_x = self.x + 1
                new_y = self.y + 1
                direction = ">"

        elif grid_symbol == 1:  # Signal == rot
            new_x = self.x
            new_y = self.y
            self.delay += 1 # increase delay by a minute since train had to wait
            direction = self.direction
        elif grid_symbol == 0:  # Signal == grün
            if self.direction == "^":
                new_x = self.x
                new_y = self.y - 1
            elif self.direction == "v":
                new_x = self.x
                new_y = self.y + 1
            elif self.direction == "<":
                new_x = self.x - 1
                new_y = self.y
            elif self.direction == ">":
                new_x = self.x + 1
                new_y = self.y
            direction = self.direction

        if grid_symbol == "10":
            # reward function to punish the train arriving at a station with delay but maximum reward remains 100 if the
            # train has negative delay i.e. arriving too early
            reward = min(round(100 - (abs(1 / 3 * self.delay ** 3) + abs(5 / 8 * self.delay)), 1), 100)
        else:
            reward = 0

        return new_x, new_y, direction, reward, self

    def move(self, new_x, new_y, new_direction):
        """
        Move the tram to the new coordinates by setting internal values and increases world_step.
        """
        self.grid.tram_grid[self.y][self.x] = 0
        self.grid.tram_grid[new_y][new_x] = self

        self.x = new_x
        self.y = new_y
        self.direction = new_direction

        self.world_step += 1
