from random import sample, shuffle, randint

from algorithm.parameters import params
from utilities.algorithm.NSGA2 import compute_pareto_metrics, \
    crowded_comparison_operator

from representation.individual import Individual
from typing import List

from utilities.algorithm.initialise_run import set_selection_imports

import time
from stats.stats import stats
import numpy as np


def selection(population):
    """
    Perform selection on a population in order to select a population of
    individuals for variation.

    :param population: input population
    :return: selected population
    """

    if params['CHANGE'] == True and params['SELECTION_CHANGE'] == True: #stop doing lexicase
        params['lexicase'] = False
        params['CHANGE'] = False
        params['SELECTION'] = "operators.selection.tournament"
        set_selection_imports()
        
    start = time.time()    
    selection = params['SELECTION'](population)
    end = time.time()
    stats['time_selection'] = end-start
    
    return selection

def lexicase(population):
    print("Doing lexicase")   
    # Initialise list of lexicase winners.
    winners = []
    
    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        candidates = population
    else:
        candidates = [i for i in population if not i.invalid]
        
    l_samples = np.shape(candidates[0].predict_result)[0]
    if params['COUNT_SAMPLES_USED_LEXICASE']:
        stats["samples_used"] = [0]*l_samples
        stats["samples_attempted"] = [0]*l_samples
        stats["samples_unsuccessful1"] = [0]*l_samples
        stats["samples_unsuccessful2"] = [0]*l_samples
        
        
    if params['SAMPLING'] == 'interleaved_rand' and stats['gen'] % 2 != 0: #odd
#        l_samples = np.shape(population[0].predict_result)[0] #[0] because the result is (_, )
        list_indexes = list(range(l_samples))
        shuffle(list_indexes)
        n = randint(1,l_samples) #number of samples used
        print("doing interleaved_rand sampling with n_samples = ", n)
        
        cases = list(range(0,n))
        
        while len(winners) < params['GENERATION_SIZE']:
            shuffle(cases)
            
            for i in range(len(candidates)):
                candidates[i].partial_predict_result = candidates[i].predict_result[list_indexes[0:n]] 
        
            while len(candidates) > 1 and len(cases) > 0:
                candidates_update = [i for i in candidates if i.partial_predict_result[cases[0]] == True]
                if params['COUNT_SAMPLES_USED_LEXICASE']:
                    stats["samples_attempted"][list_indexes[cases[0]]] += 1
                    if (len(candidates_update) < len(candidates)) and (len(candidates_update) > 0):
                        stats["samples_used"][list_indexes[cases[0]]] += 1
                    if (len(candidates_update) == len(candidates)):
                        stats["samples_unsuccessful1"][list_indexes[cases[0]]] += 1
                    if len(candidates_update) == 0:
                        stats["samples_unsuccessful2"][list_indexes[cases[0]]] += 1
                #if there is no individual which predicted the case correctly, then we return the list of candidates to the previous stage
                if len(candidates_update) == 0:
                    candidates_update = [i for i in candidates if i.partial_predict_result[cases[0]] == False]
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
           
            cases = list(range(0,n))
            
    elif params['SAMPLING'] == 'interleaved_delta' and stats['gen'] % 2 != 0: #odd
#        l_samples = np.shape(population[0].predict_result)[0] #[0] because the result is (_, )
        n = round(params['SAMPLING_COEFFICIENT']*l_samples) #number of samples used
        list_indexes = list(range(l_samples))
        shuffle(list_indexes)
        print("doing interleaved_delta sampling with n_samples = ", n)
        
        cases = list(range(0,n))
        
        while len(winners) < params['GENERATION_SIZE']:
            shuffle(cases)
            for i in range(len(candidates)):
                candidates[i].partial_predict_result = candidates[i].predict_result[list_indexes[0:n]] 
        
            while len(candidates) > 1 and len(cases) > 0:
                candidates_update = [i for i in candidates if i.partial_predict_result[cases[0]] == True]
                if params['COUNT_SAMPLES_USED_LEXICASE']:
                    stats["samples_attempted"][list_indexes[cases[0]]] += 1
                    if (len(candidates_update) < len(candidates)) and (len(candidates_update) > 0):
                        stats["samples_used"][list_indexes[cases[0]]] += 1
                    if (len(candidates_update) == len(candidates)):
                        stats["samples_unsuccessful1"][list_indexes[cases[0]]] += 1
                    if len(candidates_update) == 0:
                        stats["samples_unsuccessful2"][list_indexes[cases[0]]] += 1
                #if there is no individual which predicted the case correctly, then we return the list of candidates to the previous stage
                if len(candidates_update) == 0:
                    candidates_update = [i for i in candidates if i.partial_predict_result[cases[0]] == False]
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
           
            cases = list(range(0,n))        

    else:
#        l_samples = np.shape(candidates[0].predict_result)[0] #[0] because the result is (_, )
    
        cases = list(range(0,l_samples))
        
        if params['LEXICASE_EACH_BIT']:
            #predict_result has scores, not True or False
            
            while len(winners) < params['GENERATION_SIZE']:
                shuffle(cases)
                while len(candidates) > 1 and len(cases) > 0:
                    list_scores = []
                    for i in candidates:
                        list_scores.append(i.predict_result[cases[0]])
                    max_score = max(list_scores)
                    if max_score == 0: #no one candidate was able to predict any bit correctly
                        if params['COUNT_SAMPLES_USED_LEXICASE']:
                            stats["samples_attempted"][cases[0]] += 1
                            stats["samples_unsuccessful2"][cases[0]] += 1
                    else:
                        candidates_update = [i for i in candidates if i.predict_result[cases[0]] == max_score]
                        if params['COUNT_SAMPLES_USED_LEXICASE']:
                            stats["samples_attempted"][cases[0]] += 1
                            if len(candidates_update) < len(candidates):
                                stats["samples_used"][cases[0]] += 1
                            if len(candidates_update) == len(candidates):
                                stats["samples_unsuccessful1"][cases[0]] += 1
                        candidates = candidates_update
                    
                    del cases[0]
        
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

            
            
            
        else:
            #predict_result has True or False
            while len(winners) < params['GENERATION_SIZE']:
                #print("one time")
                #print(len(candidates))
                shuffle(cases)
                while len(candidates) > 1 and len(cases) > 0:
                    #print(len(cases))
     #               print(len(candidates[0].predict_result))
                    candidates_update = [i for i in candidates if i.predict_result[cases[0]] == True]
                    #print(len(candidates_update))
                    if params['COUNT_SAMPLES_USED_LEXICASE']:
                        stats["samples_attempted"][cases[0]] += 1
                        if (len(candidates_update) < len(candidates)) and (len(candidates_update) > 0):
                            stats["samples_used"][cases[0]] += 1
                        if (len(candidates_update) == len(candidates)):
                            stats["samples_unsuccessful1"][cases[0]] += 1
                        if len(candidates_update) == 0:
                            stats["samples_unsuccessful2"][cases[0]] += 1
     #               if params['COUNT_SAMPLES_USED_LEXICASE']:
     #                   stats["samples_attempted"][cases[0]] += 1
     #                   if (len(candidates_update) < len(candidates)) and (len(candidates_update) > 0):
     #                       stats["samples_used"][cases[0]] += 1
                    #if there is no individual which predicted the case correctly, then we return the list of candidates to the previous stage
                    if len(candidates_update) == 0:
                        #candidates_update = [i for i in candidates if i.predict_result[cases[0]] == False]
                        pass
                    else:
                        candidates = candidates_update    
                    del cases[0]                    
        
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
    
    print("doing tournament")

    # Initialise list of tournament winners.
    winners = []

    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        available = population
    else:
        available = [i for i in population if not i.invalid]
        
    candidates = []
        
    if params['SAMPLING'] == 'interleaved_rand' and stats['gen'] % 2 != 0: #odd
        l_samples = population[0].n_samples
        list_indexes = list(range(l_samples))
        shuffle(list_indexes)
        n = randint(params['TOURNAMENT_SIZE'],l_samples) #number of samples used
        print("doing interleaved_rand sampling with r = ", n)
        for i in range(n):
            candidates.append(population[list_indexes[i]])
        print(len(candidates))
        while len(winners) < params['GENERATION_SIZE']:
            # Randomly choose TOURNAMENT_SIZE competitors from the given
            # sampling candidates. Allows for re-sampling of individuals.
            competitors = sample(candidates, params['TOURNAMENT_SIZE'])
    
            # Return the single best competitor.
            winners.append(max(competitors))
        
    elif params['SAMPLING'] == 'interleaved_delta' and stats['gen'] % 2 != 0: #odd
        l_samples = population[0].n_samples
        n = round(params['SAMPLING_COEFFICIENT']*l_samples) #number of samples used
        list_indexes = list(range(l_samples))
        shuffle(list_indexes)
        print("doing interleaved_delta sampling with r = ", n)
        for i in range(n):
            candidates.append(population[list_indexes[i]])
        print(len(candidates))
        while len(winners) < params['GENERATION_SIZE']:
            # Randomly choose TOURNAMENT_SIZE competitors from the given
            # sampling candidates. Allows for re-sampling of individuals.
            competitors = sample(candidates, params['TOURNAMENT_SIZE'])
    
            # Return the single best competitor.
            winners.append(max(competitors))
        
    else:
        while len(winners) < params['GENERATION_SIZE']:
            # Randomly choose TOURNAMENT_SIZE competitors from the given
            # population. Allows for re-sampling of individuals.
            competitors = sample(available, params['TOURNAMENT_SIZE'])
    
            # Return the single best competitor.
            winners.append(max(competitors))

    # Return the population of tournament winners.
    return winners

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
