from fitness.evaluation import evaluate_fitness
from operators.crossover import crossover
from operators.mutation import mutation
from operators.replacement import replacement, steady_state
from operators.selection import selection
from operators.adaptive import update_crossover_and_mutation
from algorithm.parameters import params
from stats.stats import get_stats

def step(individuals):
    """
    Runs a single generation of the evolutionary algorithm process:
        Selection
        Variation
        Evaluation
        Replacement
    
    :param individuals: The current generation, upon which a single
    evolutionary generation will be imposed.
    :return: The next generation of the population.
    """

    # Select parents from the original population.
    parents = selection(individuals)
    
#    if params['ADAPTATIVE_CROSSOVER_AND_MUTATION']:
#        parents = update_crossover_and_mutation(parents)
    
    # Crossover parents and add to the new population.
    cross_pop = crossover(parents)

    # Mutate the new population.
    new_pop = mutation(cross_pop)

    # Evaluate the fitness of the new population.
    new_pop = evaluate_fitness(new_pop)
    
#    if params['ADAPTATIVE_CROSSOVER_AND_MUTATION']:
#        new_pop = update_crossover_and_mutation(new_pop)

#    for i in range(100):
#        print(individuals[i].crossover_probability, individuals[i].fitness)

    # Replace the old population with the new population.
    individuals = replacement(new_pop, individuals)

    # Generate statistics for run so far
    get_stats(individuals)
    
    if params['ADAPTATIVE_CROSSOVER_AND_MUTATION']:
        individuals = update_crossover_and_mutation(individuals)
    
    return individuals


def steady_state_step(individuals):
    """
    Runs a single generation of the evolutionary algorithm process,
    using steady state replacement.

    :param individuals: The current generation, upon which a single
    evolutionary generation will be imposed.
    :return: The next generation of the population.
    """
    
    individuals = steady_state(individuals)
    
    return individuals 
