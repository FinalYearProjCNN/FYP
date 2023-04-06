
import pickle
import random
from os import device_encoding

import torch
from sklearn.ensemble import GradientBoostingClassifier


class GeneticAlgorithm:

    # Define the problem
    GENERATIONS = 25
    POPULATION_SIZE = 100
    NUM_EPOCHS_RANGE = [50, 1000]
    LEARNING_RATE_RANGE = [0.000001, 0.1]
    RATE_OF_DECAY_LENGTH = 10
    MUTATION_PROBABILITY = 0.1

    def __init__(self, rate_of_decay, num_epochs, learning_rate):
        self.learning_rate = learning_rate
        self.rate_of_decay = rate_of_decay
        self.num_epochs = num_epochs
        self.population = []

    # Define the fitness function

    def fitness(self, rate_of_decay, num_epochs, learning_rate):
        with open('generators.pickle', 'rb') as f:
            train_generator, test_generator, validation_generator = pickle.load(
                f)
        # Train a model with the given parameters and evaluate its accuracy
        model = self.train_model(rate_of_decay, num_epochs, learning_rate)
        return self.evaluate_model(model, test_generator)

    def train_model(self, rate_of_decay, num_epochs, learning_rate):
        with open('generators.pickle', 'rb') as f:
            train_generator, test_generator, validation_generator = pickle.load(
                f)
        # Instantiate the classifier with the given parameters
        clf = GradientBoostingClassifier(n_estimators=num_epochs, learning_rate=self.learning_rate, subsample=1.0,
                                         max_depth=3, random_state=42)

        # Train the classifier
        for _ in range(num_epochs):
            clf.fit(train_generator, validation_generator)
            learning_rate *= rate_of_decay
            clf.set_params(learning_rate=learning_rate)

        # Return the trained classifier
        return clf

    def evaluate_model(self, model, test_generator):
        results = model.evaluate(test_generator)
        return {'accuracy': results[1], 'loss': results[0]}

    # Define the genotype
    def random_rate_of_decay(self):
        RATE_OF_DECAY_LENGTH = 10
        return [random.randint(0, 1) for _ in range(RATE_OF_DECAY_LENGTH)]

    def random_num_epochs(self):
        NUM_EPOCHS_RANGE = [50, 1000]
        return random.randint(NUM_EPOCHS_RANGE[0], NUM_EPOCHS_RANGE[1])

    def random_learning_rate(self):
        LEARNING_RATE_RANGE = [0.000001, 0.1]
        return random.uniform(LEARNING_RATE_RANGE[0], LEARNING_RATE_RANGE[1])

    def random_genotype(self):
        return (self.random_rate_of_decay(), self.random_num_epochs(), self.random_learning_rate())

    def initialize_population(self):
        self.population = [self.random_genotype()
                           for _ in range(self.POPULATION_SIZE)]

    def crossover(self, parent1, parent2):
        RATE_OF_DECAY_LENGTH = 10
        rate_of_decay = [0] * RATE_OF_DECAY_LENGTH
        if float('-inf') in [parent1[0], parent2[0], rate_of_decay]:
            rate_of_decay = self.random_rate_of_decay()
        else:
            for i in range(RATE_OF_DECAY_LENGTH):
                rate_of_decay[i] = parent1[0][i] if random.random(
                ) < 0.5 else parent2[0][i]
        num_epochs = (parent1[1] + parent2[1]) // 2
        learning_rate = (parent1[2] + parent2[2]) / 2
        return (rate_of_decay, num_epochs, learning_rate)

    def mutate(self, genotype, mutation_rate):
        # Check if genotype is None
        if genotype is None:
            return None
        # Check if genotype is a tuple
        if not isinstance(genotype, tuple) or len(genotype) != 3:
            raise ValueError("Genotype should be a tuple of 3 elements")
        rate_of_decay, num_epochs, learning_rate = genotype
        # Mutate rate of decay
        if random.random() < mutation_rate:
            RATE_OF_DECAY_LENGTH = 10
            for i in range(RATE_OF_DECAY_LENGTH):
                rate_of_decay[i] = 1 - rate_of_decay[i]
                # Mutate num epochs
                if random.random() < mutation_rate:
                    num_epochs = int(num_epochs * random.uniform(0.9, 1.1))
                # Mutate learning rate
                if random.random() < mutation_rate:
                    learning_rate = learning_rate * random.uniform(0.9, 1.1)
                    # Return mutated genotype
                    return (rate_of_decay, num_epochs, learning_rate)

    def tournament_selection(self, population, fitness_values, tournament_size=5):
        selected_parents = []
        for _ in range(len(population)):
            participants = random.sample(
                range(len(population)), tournament_size)
            best_index = participants[0]
            for j in range(1, tournament_size):
                if fitness_values[participants[j]] is not None and fitness_values[best_index] is not None and \
                        fitness_values[participants[j]] >= fitness_values[best_index]:
                    best_index = participants[j]
            # Replace None values with a very low fitness value (-inf)
            if fitness_values[best_index] is None:
                selected_parents.append(float('-inf'))
            else:
                selected_parents.append(population[best_index])
        return selected_parents

    # Define the main loop
    def main(self):
        GENERATIONS = 25
        POPULATION_SIZE = 100
        MUTATION_PROBABILITY = 0.1
        for i in range(GENERATIONS):
            # Select survivors
            selected_survivors = random.sample(
                self.population, k=POPULATION_SIZE)

            # Generate new population through crossover and mutation
            new_population = []
            fitness_values = []
            while len(new_population) < POPULATION_SIZE:
                # Select two parents randomly
                parent1, parent2 = random.sample(selected_survivors, 2)

                # Perform crossover to create offspring
                offspring = self.crossover(parent1, parent2)

                # Mutate offspring
                mutated_offspring = self.mutate(
                    offspring, MUTATION_PROBABILITY)

                # Add mutated offspring to new population
                new_population.append(mutated_offspring)

        # Replace old population with new population
        population = new_population

        # Evaluate fitness of new population
        fitness_values = [self.fitness(rate_of_decay, num_epochs, learning_rate)
                          for rate_of_decay, num_epochs, learning_rate in population]

        # Get the best genotype and its fitness
        best_genotype, best_fitness = None, -float('inf')
        for genotype, fitness_value in zip(population, fitness_values):
            if fitness_value is not None and fitness_value > best_fitness:
                best_genotype = genotype
                best_fitness = fitness_value

            # Print the best genotype and its fitness
            print(
                f"Generation {i + 1}: Best genotype = {best_genotype}, Best fitness = {best_fitness}")

            if best_genotype is not None:
                rate_of_decay, num_epochs, learning_rate = best_genotype
                print("Optimal rate of decay:", rate_of_decay)
                print("Optimal learning rate:", learning_rate)
                print("Optimal number of epochs:", num_epochs)
