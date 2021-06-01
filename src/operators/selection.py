from random import sample, shuffle, randint

from algorithm.parameters import params
from utilities.algorithm.NSGA2 import compute_pareto_metrics, \
    crowded_comparison_operator

from representation.individual import Individual
from typing import List

def selection(population):
    """
    Perform selection on a population in order to select a population of
    individuals for variation.

    :param population: input population
    :return: selected population
    """

    return params['SELECTION'](population)

def another_lexicase(population) -> List[Individual]:
    """
    Given an entire population, choose the individuals that do the best on
    randomly chosen test cases. Allows for selection of 'specialist' individuals
    that do very well on some test cases even if they do poorly on others.

    :param population: A population from which to select individuals.
    :return: A population of the selected individuals from lexicase selection -- allows
             repeated individuals
    """
    # Initialise list of lexicase selections
    winners = []

    # Max or min
    maximise_fitness = params['FITNESS_FUNCTION'].maximise

    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        available = population
    else:
        available = [i for i in population if not i.invalid]

    # Basic ensure individuals have been tested on same number of test cases, and that there is at least one test case
    assert (len(available[0].test_case_results) == len(available[1].test_case_results))
    assert (len(available[0].test_case_results) > 0)

    while len(winners) < params['GENERATION_SIZE']:
        # Random ordering of test cases
        random_test_case_list = list(range(len(available[0].test_case_results)))
        shuffle(random_test_case_list)

        # Only choose from a sample not from the entire available population

        if params['LEXICASE_TOURNAMENT']:
            candidates = sample(available, params['TOURNAMENT_SIZE'])
        else:
            candidates = available
        candidate_size = len(candidates)
        while candidate_size > 0:
            # Calculate best score for chosen test case from candidates
            scores = []
            for ind in candidates:
                scores.append(ind.test_case_results[random_test_case_list[0]])
            if maximise_fitness:
                best_score = max(scores)
            else:
                best_score = min(scores)

            # Only retain individuals who have the best score for the test case
            remaining = []
            candidate_size = 0
            for ind in candidates:
                if ind.test_case_results[random_test_case_list[0]] == best_score:
                    remaining.append(ind)
                    candidate_size += 1
            candidates = remaining

            # If only one individual remains, choose that individual
            if len(candidates) == 1:
                winners.append(candidates[0])
                break
                
            # If this was the last test case, randomly choose an individual from remaining candidates
            elif len(random_test_case_list) == 1:
                # Penalize longer solutions
                min_nodes = params["MAX_TREE_NODES"] + 1
                best_ind = None
                for ind in candidates:
                    if ind.nodes < min_nodes:
                        best_ind = ind
                        min_nodes = ind.nodes
                winners.append(best_ind)

                # Choose randomly among solutions
                # winners.append(sample(candidates, 1)[0])
                break

            # Go to next test case and loop
            else:
                random_test_case_list.pop(0)

    # Return the population of lexicase selections.
    return winners

def lexicase(population):
    
    # Initialise list of lexicase winners.
    winners = []
    
    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        candidates = population
    else:
        candidates = [i for i in population if not i.invalid]
   
    l_samples = len(candidates[0].predict_result)
    
    cases = list(range(0,l_samples))
    
    while len(winners) < params['GENERATION_SIZE']:
        shuffle(cases)
        while len(candidates) > 1 and len(cases) > 0:
            candidates_update = [i for i in candidates if i.predict_result[cases[0]] == True]
            #if there is no individual which predicted the case correctly, then we return the list of candidates to the previous stage
            if len(candidates_update) == 0:
                candidates_update = [i for i in candidates if i.predict_result[cases[0]] == False]
            del cases[0]
            candidates = candidates_update

        if len(candidates) == 1:
            winners.append(candidates[0])
        elif len(cases) == 0:# and len(candidates) >= 1:
            r = randint(0,len(candidates)-1)
            winners.append(candidates[r])
        
        if params['INVALID_SELECTION']:
            candidates = population
        else:
            candidates = [i for i in population if not i.invalid]
       
        cases = list(range(0,l_samples))
            
    return winners


def tournament(population):
    """
    Given an entire population, draw <tournament_size> competitors randomly and
    return the best. Only valid individuals can be selected for tournaments.

    :param population: A population from which to select individuals.
    :return: A population of the winners from tournaments.
    """
    
    #print(population)

    # Initialise list of tournament winners.
    winners = []

    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        available = population
    else:
        available = [i for i in population if not i.invalid]

    while len(winners) < params['GENERATION_SIZE']:
        # Randomly choose TOURNAMENT_SIZE competitors from the given
        # population. Allows for re-sampling of individuals.
        competitors = sample(available, params['TOURNAMENT_SIZE'])
        #random.sample(sequence, k)
        #Returns: k length new list of elements chosen from the sequence.
#        print()
#        print(competitors[0].test, competitors[1].fitness)
#        print(max(competitors).fitness)

        # Return the single best competitor.
        winners.append(max(competitors))

    # Return the population of tournament winners.
    #print(winners)
    return winners


def truncation(population):
    """
    Given an entire population, return the best <proportion> of them.

    :param population: A population from which to select individuals.
    :return: The best <proportion> of the given population.
    """

    # Sort the original population.
    population.sort(reverse=True)

    # Find the cutoff point for truncation.
    cutoff = int(len(population) * float(params['SELECTION_PROPORTION']))

    # Return the best <proportion> of the given population.
    return population[:cutoff]


def nsga2_selection(population):
    """Apply NSGA-II selection operator on the *population*. Usually, the
    size of *population* will be larger than *k* because any individual
    present in *population* will appear in the returned list at most once.
    Having the size of *population* equals to *k* will have no effect other
    than sorting the population according to their front rank. The
    list returned contains references to the input *population*. For more
    details on the NSGA-II operator see [Deb2002]_.
    
    :param population: A population from which to select individuals.
    :returns: A list of selected individuals.
    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """

    selection_size = params['GENERATION_SIZE']
    tournament_size = params['TOURNAMENT_SIZE']

    # Initialise list of tournament winners.
    winners = []

    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        available = population
    else:
        available = [i for i in population if not i.invalid]

    # Compute pareto front metrics.
    pareto = compute_pareto_metrics(available)

    while len(winners) < selection_size:
        # Return the single best competitor.
        winners.append(pareto_tournament(available, pareto, tournament_size))

    return winners


def pareto_tournament(population, pareto, tournament_size):
    """
    The Pareto tournament selection uses both the pareto front of the
    individual and the crowding distance.

    :param population: A population from which to select individuals.
    :param pareto: The pareto front information.
    :param tournament_size: The size of the tournament.
    :return: The selected individuals.
    """
    
    # Initialise no best solution.
    best = None
    
    # Randomly sample *tournament_size* participants.
    participants = sample(population, tournament_size)
    
    for participant in participants:
        if best is None or crowded_comparison_operator(participant, best,
                                                       pareto):
            best = participant
    
    return best


# Set attributes for all operators to define multi-objective operators.
nsga2_selection.multi_objective = True
