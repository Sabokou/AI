# Argument parser for CLI
import argparse
# Data manipulation
import statistics
import numpy as np
from numpy.random import default_rng
import pandas as pd
# Charting library
import plotly.express as px
# Imports for CLI formatting
from rich.console import Console
from rich.table import Table
from rich.live import Live

# Create random number generator as per numpy documentation
RNG = default_rng()

# Add argument parser to enable running this script without having to modify source code
parser = argparse.ArgumentParser(description="Training algorithm for a simple genetic algorithm.")
# -- All arguments for calling this program --#
# Mandatory parameters
parser.add_argument("population", type=int, help="defines the population size used for evolution", default=5)
parser.add_argument("features", type=int, help="defines the amount of features in each individual", default=10)
# Optional parameters to specify behavior
parser.add_argument("-S", "--selection_strategy", help="the strategy used to determine mating pairs", default="fitness",
                    choices=['fitness', 'best'])
parser.add_argument("-C", "--crossver_strategy", help="the strategy used to determine mating process",
                    default="one-point-crossover", choices=["one-point-crossover", "template", "shuffle"])
parser.add_argument("-M", "--mutation_strategy", help="the strategy used to mutate individuals",
                    default="replace", choices=["replace", "swap"])
parser.add_argument("-c", "--chance", type=float, help="mutation chance", default=0.01)
parser.add_argument("-m", "--max_mutations", type=int, help="maximum of mutations in one offspring", default="1")
parser.add_argument('--chart', '-v', help="if used implies will generate a chart showing max score at each generation",
                    action='store_true', default=False)
# Read arguments from CLI
args = parser.parse_args()


class Individual:
    """
    Custom class to simulate the problem of maximizing the value of a genome by manipulating the genes.
    """
    def __init__(self, size: int):
        self.features = None
        self.size = size

    def feature_initialisation(self):
        """
        Fills the genome with randomly generated integers from the interval [0, 1]
        """
        self.features = RNG.integers(0, 2, size=self.size)

    def calculate_fitness(self):
        """
        The fitness in this scenario is defined as the sum of all gene values.

        Returns
        -------
        float: sum of all genes
        """
        return np.sum(self.features)


class GeneticAlgorithm:
    """"
    General purpose class to be used for genetic algorithm training. Not all methods are compatible with all use cases.
    """
    def __init__(self, population: int = 5, feature_size: int = 5, training_object=None):
        self.population_count = population
        self.feature_size = feature_size
        self.population = list()
        self.fitness_dict = None
        self.individual = training_object
        self.generation_counter = 1

    def initialisation(self):
        for _ in range(self.population_count):
            agent = self.individual(self.feature_size)
            agent.feature_initialisation()
            self.population.append(agent)

    def calculate_fitness(self):
        """

        Returns
        -------
        float: Minimum fitness of generation
        float: Maximum fitness of generation
        float: Average fitness of all individuals in the current generation
        """
        self.fitness_dict = dict()
        for count, agent in enumerate(self.population):
            self.fitness_dict[count] = agent.calculate_fitness()
        return min(self.fitness_dict.values()), max(self.fitness_dict.values()), statistics.fmean(
            self.fitness_dict.values())

    def selection(self, method: str = "fitness", reduction_factor: float = 0):
        assert self.fitness_dict is not None
        mating_pairs = list()

        # create n new kids to maintain population
        new_population_size = len(self.population) // (reduction_factor + 1)

        # The selection criterion is based on fitness
        if method == "fitness":
            # get all fitness scores to calculate odds of being included in mating procedure
            fitness_values = list(self.fitness_dict.values())

            # normalize fitness values
            fitness_values = [float(i) / sum(fitness_values) for i in fitness_values]

            for _ in range(new_population_size):
                pair = RNG.choice(list(self.fitness_dict.keys()), 2, replace=False, p=fitness_values)
                mating_pairs.append(pair)

        elif method == "best":
            candidates = [item[0] for item in
                          sorted(list(self.fitness_dict.items()), key=lambda x: x[1], reverse=True)[:2]]

            for _ in range(new_population_size):
                pair = RNG.choice(candidates, 2, replace=False)
                mating_pairs.append(pair)

        return mating_pairs

    def crossover(self, mating_pairs, strategy: str = "one-point-crossover"):
        # Failure checking in case inputs couldn't be matched to procedure
        assert mating_pairs is not None
        assert strategy in ["one-point-crossover", "template", "shuffle"]

        new_population = list()
        if strategy == "one-point-crossover":
            # determine up to which index the first partner should be used and when the second index
            crossover_index = RNG.integers(self.feature_size + 1)
            for pair in mating_pairs:
                chromosome = list()
                for index, gene in enumerate(self.population[pair[0]].features):
                    if index <= crossover_index:
                        chromosome.append(gene)
                    else:
                        chromosome.append(self.population[pair[1]].features[index])
                new_agent = Individual(self.feature_size)
                new_agent.features = chromosome
                new_population.append(new_agent)

        elif strategy == "template":
            crossover_template = RNG.integers(0, 2, size=self.feature_size)
            for pair in mating_pairs:
                chromosome = list()
                for index, element in enumerate(crossover_template):
                    chromosome.append(self.population[pair[element]].features[index])
                new_agent = Individual(self.feature_size)
                new_agent.features = chromosome
                new_population.append(new_agent)

        elif strategy == "shuffle":
            for pair in mating_pairs:
                baseline_chromosome = list()
                for i in range(self.feature_size):
                    baseline_chromosome.append([i, self.population[pair[0]].features[i]])
                    baseline_chromosome.append([i, self.population[pair[1]].features[i]])

                RNG.shuffle(baseline_chromosome)
                baseline_chromosome = sorted(baseline_chromosome[:self.feature_size],
                                             key=lambda x: x[0], reverse=True)
                chromosome = [gene[1] for gene in baseline_chromosome]

                new_agent = Individual(self.feature_size)
                new_agent.features = chromosome
                new_population.append(new_agent)

        self.population = new_population

    def mutation(self, chance: float = 0.1, max_mutation_count: int = 1, method: str = "replace"):
        if method == "replace":
            for agent in self.population:
                mutation_index = RNG.integers(0, self.feature_size, size=max_mutation_count)
                for index in mutation_index:
                    if chance >= RNG.random():
                        if agent.features[index] == 0:
                            agent.features[index] = 1
                        else:
                            agent.features[index] = 0
        elif method == "swap":
            for index, agent in enumerate(self.population):
                if chance >= RNG.random():
                    mutation_index = RNG.integers(0, self.feature_size, size=(max_mutation_count, 2))
                    for mutation in mutation_index:
                        agent.features[mutation[0]], agent.features[mutation[1]] = agent.features[mutation[1]], \
                                                                                   agent.features[mutation[0]]
        # Generation ends with mutation, therefore the counter is increased
        self.generation_counter += 1

    def create_gen_visualiser(self, max_score, avg_score):
        # Init Table object from rich library as stylising object for output and add information as title and caption
        table = Table(title=f"Generation {self.generation_counter}",
                      caption=f"Current Average Score: {round(avg_score, 2)} | Maximum Score reached: {max_score}")

        # Create column structure for table view
        table.add_column("Agent ID", no_wrap=True)
        table.add_column("Chromosome", no_wrap=True)
        table.add_column("Fitness")

        # Additional check to ensure fitness_dict is initialized is during initial call for visualization not given
        if self.fitness_dict is not None:
            for agent_number, fitness in self.fitness_dict.items():
                chromosome = ""
                for gene in self.population[agent_number].features:
                    if gene == 0:
                        color = "[red]"
                    else:
                        color = "[green]"
                    chromosome += color + str(gene) + " "

                if fitness == self.feature_size:
                    fitness = "[bright_green]" + str(fitness)
                elif self.feature_size - 1 >= fitness >= self.feature_size - self.feature_size // 4:
                    fitness = "[pale_green1]" + str(fitness)
                elif self.feature_size - self.feature_size // 4 > fitness >= self.feature_size // 2:
                    fitness = "[bright_yellow]" + str(fitness)
                elif self.feature_size // 2 > fitness >= self.feature_size // 4:
                    fitness = "[dark_orange3]" + str(fitness)
                else:
                    fitness = "[bright_red]" + str(fitness)

                table.add_row(str(agent_number + 1), chromosome, fitness)

        return table


if __name__ == "__main__":
    algo = GeneticAlgorithm(population=args.population, feature_size=args.features, training_object=Individual)
    algo.initialisation()

    list_performances = list()

    maximum = 0
    with Live(algo.create_gen_visualiser(maximum, 0), refresh_per_second=4) as live:
        while maximum < args.features:
            # Evaluate current generation
            minimum, maximum, average = algo.calculate_fitness()
            list_performances.append([algo.generation_counter, minimum, maximum, average])

            live.update(algo.create_gen_visualiser(maximum, average))

            # Creates new generation
            pairs = algo.selection(args.selection_strategy)
            algo.crossover(pairs, args.crossver_strategy)

            algo.mutation(chance=args.chance, max_mutation_count=args.max_mutations, method=args.mutation_strategy)

    if args.chart:
        df = pd.DataFrame(list_performances, columns=["generation_counter", "minimum", "maximum", "average"])
        fig = px.line(df, x="generation_counter", y="maximum")
        fig.show()
