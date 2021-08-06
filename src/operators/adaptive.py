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

    #Update crossover and mutation probabilities for each individual
    for name, ind in enumerate(individuals):
        if not ind.invalid: #update just the valid individuals; the invalid ones have already a probability of 1
            ind.crossover_probability = (1 - ind.fitness) * (1 - stats['ave_fitness']) + ind.fitness
            alpha = euclidean_distance_ind[name]/params['POPULATION_SIZE']
            beta = (alpha * euclidean_distance_pop + 1) * (1 - ind.fitness)
            if beta < 1e-3:
                rho = 1
            else:
                rho = 1 - (ind.fitness + (1 - stats['ave_fitness'])) * (1 - ind.fitness) / beta
            ind.mutation_probability  = ((ind.fitness) * rho * (1 - stats['ave_fitness'])**2) #/ 10
                       
    return individuals
