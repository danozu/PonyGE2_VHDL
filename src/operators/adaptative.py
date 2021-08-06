#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 15:41:33 2021

@author: allan
"""

from algorithm.parameters import params
from stats.stats import stats

import numpy as np
from scipy.spatial import distance

def update_crossover_and_mutation(individuals):
    
    #Calculate average fitness
#    total_fitness = 0
#    n = 0
    
#    for name, ind in enumerate(individuals):
#        if ind.eval_ind:
#            total_fitness += ind.fitness
#            n += 1
    
#    fitness_avg = total_fitness/n
#    print(fitness_avg)
    
    genome_matrix = np.zeros([params['POPULATION_SIZE'], stats["max_genome_length"]], dtype=float)
    for i in range(params['POPULATION_SIZE']):
        genome_matrix[i, 0:len(individuals[i].genome)] = individuals[i].genome
    genome_matrix = genome_matrix/params['CODON_SIZE']
    euclidean_matrix = np.zeros([params['POPULATION_SIZE'], params['POPULATION_SIZE']], dtype=float)
    for i in range(params['POPULATION_SIZE']):
        for j in range(params['POPULATION_SIZE']):
            euclidean_matrix[i,j] = distance.euclidean(genome_matrix[i], genome_matrix[j])

    euclidean_distance_ind = np.sum(euclidean_matrix, axis=0) #d_i
    euclidean_distance_pop = np.sum(euclidean_distance_ind) #D_euclidean
    stats['D_Euclidean'] = euclidean_distance_pop
    print("updated D_Euclidean")
    

        

    #Update crossover and mutation probabilities for each individual
    for name, ind in enumerate(individuals):
        if not ind.invalid: #update just the valid individuals; the invalid ones have already a probability of 1
#            ind.crossover_probability = 0.8
#            ind.mutation_probability = 0.01
            #first results - crossover_probability is updated using 1-fit and mutation_probability not
#            ind.crossover_probability = (1 - ind.fitness)*stats['ave_fitness'] + 1 - (1 - ind.fitness)
#            alfa = euclidean_distance_ind[name]/params['POPULATION_SIZE']
#            beta = (alfa * euclidean_distance_pop + 1) * ind.fitness
#            ro = 1 - (1 - ind.fitness + stats['ave_fitness']) * ind.fitness / beta
#            ind.mutation_probability  = ((1 - ind.fitness) * ro * (stats['ave_fitness'])**2) / 10
            #second results - both are updated using 1-fit
#            ind.crossover_probability = (1 - ind.fitness)*stats['ave_fitness'] + 1 - (1 - ind.fitness)
#            alfa = euclidean_distance_ind[name]/params['POPULATION_SIZE']
#            beta = (alfa * euclidean_distance_pop + 1) * (1 - ind.fitness)
#            ro = 1 - (ind.fitness + stats['ave_fitness']) * (1 - ind.fitness) / beta
#            ind.mutation_probability  = (ind.fitness * ro * (stats['ave_fitness'])**2) / 10
            #third results - both are updated using fit
#            ind.crossover_probability = ind.fitness * stats['ave_fitness'] + 1 - ind.fitness
#            alfa = euclidean_distance_ind[name]/params['POPULATION_SIZE']
#            beta = (alfa * euclidean_distance_pop + 1) * ind.fitness
#            ro = 1 - (1 - ind.fitness + stats['ave_fitness']) * ind.fitness / beta
#            ind.mutation_probability  = ((1 - ind.fitness) * ro * (stats['ave_fitness'])**2) / 10
            #fourth results - both are updated using 1-fit and 1-favg
            #if np.isnan(ind.fitness):
            #    ind.crossover_probability = 1
            #    ind.mutation_probability = 1
            #else:                 
            ind.crossover_probability = (1 - ind.fitness) * (1 - stats['ave_fitness']) + ind.fitness
            alpha = euclidean_distance_ind[name]/params['POPULATION_SIZE']
            beta = (alpha * euclidean_distance_pop + 1) * (1 - ind.fitness)
            if beta < 1e-3:
                rho = 1
            else:
                rho = 1 - (ind.fitness + (1 - stats['ave_fitness'])) * (1 - ind.fitness) / beta
            ind.mutation_probability  = ((ind.fitness) * rho * (1 - stats['ave_fitness'])**2) #/ 10
                       
            #usando o mesmo de cima, só que com a fitness original (e não 1 - fit)
            #ind.crossover_probability = ind.fitness * stats['ave_fitness'] + (1 - ind.fitness)
            #alpha = euclidean_distance_ind[name]/params['POPULATION_SIZE']
            #beta = (alpha * euclidean_distance_pop * ind.fitness) + ind.fitness
            #if beta < 1e-3:
            #    rho = 1
            #else:
            #    rho = 1 - ((1-ind.fitness) + stats['ave_fitness']) * (ind.fitness) / beta
            #ind.mutation_probability  = ((1-ind.fitness) * rho * (stats['ave_fitness'])**2) #/ 10
    
    return individuals