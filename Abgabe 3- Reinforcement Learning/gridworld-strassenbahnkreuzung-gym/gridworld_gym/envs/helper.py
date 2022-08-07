class Signal:
    """A signal allows the tram to be stopped on the current segment for ordering and preventing crashes."""

    def __init__(self):
        """
        Defines the default state of a signal as being red
        """
        self.status = 0

    def turn_red(self):
        """
        Switches the status of the signal to 1 (red light)
        """
        self.status = 1

    def turn_green(self):
        """
        Switches the status of the signal to 0 (green light)
        """
        self.status = 0


class Switch:
    """A switch that replaces a track segment and allows the tram to move in different directions."""
    def __init__(self, alternative: str, default: str):
        """
        Initializes a switch that has two states. The default state is given by what the track would be if no switching
        would be allowed i.e. straight. The alternative state is always a directional state i.e. diagonally up and right.
        :param alternative: switch's alternative state
        :param default: switch's default state (usually "-" or "|")
        """
        self.status = default
        self.status_switched = alternative
        self.default = default

    def change_status(self, status_updated):
        """
        Switches Status between the default state and the activated state
        :param status_updated: new state for switch
        """
        self.status = status_updated


class Stop:
    """A stop is defined as a station on the network."""

    def __init__(self, name: str= None):
        """
        Initializes the stop with a given name. The name can be used in downstream task to check if a tram should be
        there.
        :param name:
        """
        self.name = name