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
    """
    General purpose class to be used for genetic algorithm training. Not all methods are compatible with all use cases.
    Especially crossover might not work if all possible genes must be included in the new genome same for the replace
    mutation. The visualization function is also written to explicitly conform to test individual.
    """

    def __init__(self, population: int = 5, feature_size: int = 5, training_object=None):
        """
        Initializing the class with default or None values as per PEP8.
        """
        self.population_count = population
        self.feature_size = feature_size
        self.population = list()
        self.fitness_dict = None
        self.individual = training_object
        self.generation_counter = 1

    def initialisation(self):
        """
        Create the first generation based on the passed training object and add it to the list of individuals of the
        current generation.
        """
        for _ in range(self.population_count):
            # Assuming that the init function of a training object accepts at least one parameter designating how
            # many features it has
            agent = self.individual(self.feature_size)
            # calls a separate feature initialisation to generate the generation's chromosome
            agent.feature_initialisation()
            # add the initialised individual to the population
            self.population.append(agent)

    def calculate_fitness(self):
        """
        Uses the integrated function of the individuals to generate the fitness for all individuals in a given population.

        Returns
        -------
        float: Minimum fitness of generation
        float: Maximum fitness of generation
        float: Average fitness of all individuals in the current generation
        """
        # saves the fitness in a dictionary matching the index of an individual with their performance
        self.fitness_dict = dict()

        for count, agent in enumerate(self.population):
            self.fitness_dict[count] = agent.calculate_fitness()

        return min(self.fitness_dict.values()), max(self.fitness_dict.values()), statistics.fmean(
            self.fitness_dict.values())  # fast integrated mean function

    def selection(self, method: str = "fitness", reduction_factor: float = 0) -> list[list]:
        """
        Selection is the process of determining the pairs of mates. Multiple methods are implemented and the
        reduction of agents from one generation to the next is also possible. Although there is no explicit stop
        criterion implemented for if the population count drops to >2.
        """
        # if the function is called prior to determining the pairs raise an error
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

            # based on the determined new population size generate pairs
            for _ in range(new_population_size):
                # randomly sample 2 individuals from the generation without allowing self-reproduction based on fitness
                pair = RNG.choice(list(self.fitness_dict.keys()), 2, replace=False, p=fitness_values)
                mating_pairs.append(pair)

        # Selection criterion based solely on the 2 best individual by fitness ranking
        elif method == "best":
            # Fetch the top 2 candidates and their population index by "sorting" the dictionary and limiting the list
            # index
            candidates = [item[0] for item in
                          sorted(list(self.fitness_dict.items()), key=lambda x: x[1], reverse=True)[:2]]

            for _ in range(new_population_size):
                # The order of the individuals in their mating order is determined randomly
                pair = RNG.choice(candidates, 2, replace=False)
                mating_pairs.append(pair)

        return mating_pairs

    def crossover(self, mating_pairs, strategy: str = "one-point-crossover"):
        """
        The crossover function determines the offspring of mating pairs.
        one-point-crossover: A single point (index) is determined to split from which individual the genes are taken
                             Up to the point genes from the first individual is taken afterwards from the 2nd.
        ----
        Strategies:
        template: Create a random template from which of the two individual the gene of a given position should be taken
                  from.
        shuffle: Create a list of all genes and their index. From the list the first n elements are taken and sorted by
                 their original index thereby randomly sampling from both individuals but preserving the gene order.
        """
        # Failure checking in case inputs couldn't be matched to function necessities
        assert mating_pairs is not None
        assert strategy in ["one-point-crossover", "template", "shuffle"]

        new_population = list()
        if strategy == "one-point-crossover":
            # determine up to which index the first partner should be used and when the second index
            crossover_index = RNG.integers(self.feature_size + 1)
            for pair in mating_pairs:
                chromosome = list()
                # iterate over the first individual
                for index, gene in enumerate(self.population[pair[0]].features):
                    # if the index of the gene is smaller than the crossver index take it
                    if index <= crossover_index:
                        chromosome.append(gene)
                    # otherwise append the gene of the same index from the second individual of the pair
                    else:
                        chromosome.append(self.population[pair[1]].features[index])
                # old individual will be discarded - create new individual
                new_individual = Individual(self.feature_size)
                # change its features to be the generated chromosome
                new_individual.features = chromosome
                # add the individual to the new population list
                new_population.append(new_individual)

        elif strategy == "template":
            # generate a set of n integers either 0 or 1 where n is given by the feature size (genome length)
            crossover_template = RNG.integers(0, 2, size=self.feature_size)
            # the crossover template is used for all pairs
            for pair in mating_pairs:
                chromosome = list()
                # for each entry in the crossover template and its index
                for index, element in enumerate(crossover_template):
                    # take the gene at the index of the individual from the mating pair of the population
                    chromosome.append(self.population[pair[element]].features[index])

                # create new individual, change its features and append it to the new population list
                new_individual = Individual(self.feature_size)
                new_individual.features = chromosome
                new_population.append(new_individual)

        elif strategy == "shuffle":
            for pair in mating_pairs:
                baseline_chromosome = list()
                # for all feature indices
                for i in range(self.feature_size):
                    # generate a list with the index and the feature at the index for both individual and append to the
                    # baseline chromosome from which the new individual will be generated
                    baseline_chromosome.append([i, self.population[pair[0]].features[i]])
                    baseline_chromosome.append([i, self.population[pair[1]].features[i]])

                # inplace shuffling of the sublists
                RNG.shuffle(baseline_chromosome)
                # take the first n lists and sort them by their index from lowest to highest
                baseline_chromosome = sorted(baseline_chromosome[:self.feature_size],
                                             key=lambda x: x[0], reverse=True)
                # generate a new list by taking the feature from the sublist and discarding the index
                chromosome = [gene[1] for gene in baseline_chromosome]

                # create new individual, change its features and append it to the new population list
                new_individual = Individual(self.feature_size)
                new_individual.features = chromosome
                new_population.append(new_individual)

        # overwrite old population with new population
        self.population = new_population

    def mutation(self, chance: float = 0.1, max_mutation_count: int = 1, method: str = "replace"):
        """
        The mutation function changes genes in possibly all individuals.
        ----
        Methods
        replace: replaces the gene at an index with the alternative (0->1 | 1->0)
        swap: randomly swap the position of genes
        """
        if method == "replace":
            # all agents have "the chance" to receive a mutation
            for individual in self.population:
                # determine based on max mutation count random integers between 0 and the feature size
                mutation_index = RNG.integers(0, self.feature_size, size=max_mutation_count)
                # iterate over mutation index list
                for index in mutation_index:
                    # if a random float between 0 and 1 is below the chance threshold the mutation happens
                    if chance >= RNG.random():
                        # replace 0 with 1 and 0 with 1 at the index
                        if individual.features[index] == 0:
                            individual.features[index] = 1
                        else:
                            individual.features[index] = 0

        elif method == "swap":
            # all agents have "the chance" to receive a mutation
            for individual in enumerate(self.population):
                # generate a list with n entries with lists of 2 random integers between 0 and the max index
                mutation_index = RNG.integers(0, self.feature_size, size=(max_mutation_count, 2))
                # for each determined mutation
                for mutation in mutation_index:
                    # if a random float between 0 and 1 is below the chance threshold the mutation happens
                    if chance >= RNG.random():
                        individual.features[mutation[0]], individual.features[mutation[1]] = individual.features[
                                                                                                 mutation[1]], \
                                                                                             individual.features[
                                                                                                 mutation[0]]
        # generation ends with mutation, therefore the counter is increased
        self.generation_counter += 1

    def create_gen_visualiser(self, max_score: float, avg_score: float) -> Table:
        """
        Create a colored table with informationen from the training including a look at all genomes and their fitness.
        Returns
        -------
        Table: string-like object from rich
        """
        # init Table object from rich library as stylising object for output and add information as title and caption
        table = Table(title=f"Generation {self.generation_counter}",
                      caption=f"Current Average Score: {round(avg_score, 2)} | Maximum Score reached: {max_score}")

        # create column structure for table view
        table.add_column("Agent ID", no_wrap=True)
        table.add_column("Chromosome", no_wrap=True)
        table.add_column("Fitness")

        # additional check to ensure fitness_dict is initialized is during initial call for visualization not given
        if self.fitness_dict is not None:
            for agent_number, fitness in self.fitness_dict.items():
                chromosome = ""
                # iterate over each gene to add them to output string with correct coloration
                for gene in self.population[agent_number].features:
                    # if the gene is 0 - meaning it is unwanted - it's colored in red
                    if gene == 0:
                        color = "[red]"
                    # else (gene = 1) it will be colored in green
                    else:
                        color = "[green]"
                    chromosome += color + str(gene) + " "

                # depending on fitness band color the value in the fitness column accordingly
                # maximum fitness achieved
                if fitness == self.feature_size:
                    fitness = "[bright_green]" + str(fitness)
                # fitness between 100% & 75% of maximum
                elif self.feature_size - 1 >= fitness >= self.feature_size - self.feature_size // 4:
                    fitness = "[pale_green1]" + str(fitness)
                # fitness between 75% & 50% of maximum
                elif self.feature_size - self.feature_size // 4 > fitness >= self.feature_size // 2:
                    fitness = "[bright_yellow]" + str(fitness)
                # fitness between 50% & 25% of maximum
                elif self.feature_size // 2 > fitness >= self.feature_size // 4:
                    fitness = "[dark_orange3]" + str(fitness)
                # fitness below 25% of maximum
                else:
                    fitness = "[bright_red]" + str(fitness)

                # add all information as strings with colors to the table
                # agent number is +1 to have the index start with 1
                table.add_row(str(agent_number + 1), chromosome, fitness)

        return table


if __name__ == "__main__":
    # initialize an instance of the genetic algorithm
    algo = GeneticAlgorithm(population=args.population, feature_size=args.features, training_object=Individual)
    # create the first generation
    algo.initialisation()

    # create an empty list to later create a log
    list_performances = list()
    # variable initialization outside of while loop that stores the current generation's maximum fitness
    # it doesn't matter if it were the overall maximum fitness instead of the current fitness
    maximum = 0
    # create a widget to update generating only one console output instead of printing generating multiple tables
    with Live(algo.create_gen_visualiser(maximum, 0), refresh_per_second=4) as live:
        # genetic evolution will continue to happen until the theoretical maximum is reached
        while maximum < args.features:
            # evaluate current generation
            minimum, maximum, average = algo.calculate_fitness()
            # log performances
            list_performances.append([algo.generation_counter, minimum, maximum, average])
            # update the widget with the current snapshot
            live.update(algo.create_gen_visualiser(maximum, average))

            # creates new generation
            # create mating pairs with the passed selection strategy
            pairs = algo.selection(args.selection_strategy)
            # use generated pairs to create offspring
            algo.crossover(pairs, args.crossver_strategy)

            # mutate the offspring
            algo.mutation(chance=args.chance, max_mutation_count=args.max_mutations, method=args.mutation_strategy)

    # if user wants a chart
    if args.chart:
        # convert data into a pandas dataframe to make graphing easier
        df = pd.DataFrame(list_performances, columns=["generation_counter", "minimum", "maximum", "average"])
        # create a line chart with plotly and all tracked values to enable interactive analysis
        fig = px.line(df, x="generation_counter", y=["minimum", "maximum", "average"])
        # shows the chart in the user's browser
        fig.show()
