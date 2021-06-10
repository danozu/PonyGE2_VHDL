import numpy as np
np.seterr(all="raise")

from algorithm.parameters import params, set_params
from utilities.fitness.get_data import get_data
from utilities.fitness.math_functions import *
from utilities.fitness.optimize_constants import optimize_constants

from fitness.base_ff_classes.base_ff import base_ff

import subprocess
import numpy as np
import random

import os
import shutil

from stats.stats import stats

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True, cwd=r'../../VHDL/individuals')
    proc_stdout = process.communicate()[0].strip()
    return proc_stdout.decode('utf-8')

def eval_vhdl(ind):
    r = random.randint(0,10**10)
    vhdl = open(r'../../VHDL/individuals/ind' + str(r) + '.vhdl','w+')
    
    # Replace the target string
    ind = ind.replace('ind', 'ind' + str(r))
    ind = ind.replace('""', '')

    vhdl.write(ind)
    vhdl.close()
    
    if params['PROBLEM'] == 'ssd':
    
        with open(r'../../VHDL/ssd/tb.vhdl', 'r') as file :
            filedata = file.read()
    
    elif params['PROBLEM'] == 'multiplexer':
        with open(r'../../VHDL/multiplexer/tb.vhdl', 'r') as file :
            filedata = file.read()
    
    # Replace the target string
    filedata = filedata.replace('ind', 'ind' + str(r))
    filedata = filedata.replace('tb', 'tb' + str(r))
    
    # Write the file out again
    with open(r'../../VHDL/individuals/tb' + str(r) + '.vhdl', 'w') as file:
        file.write(filedata)
      
    if params['SIMULATOR'] == 'ghdl':
        result = subprocess_cmd("ghdl -a --std=08 --work=" + str(r) + " ind" + str(r) + ".vhdl tb" + str(r) + ".vhdl ; ghdl -e --std=08 --work=" + str(r) + " tb" + str(r) + " ; ghdl -r --std=08 --work=" + str(r) + " tb" + str(r))
    elif params['SIMULATOR'] == 'nvc':
        result = subprocess_cmd("nvc --std=08 --work=" + str(r) + " -a ind" + str(r) + ".vhdl tb" + str(r) + ".vhdl -e tb" + str(r) + "-r")

    os.remove(r'../../VHDL/individuals/tb' + str(r) + '.vhdl')
    os.remove(r'../../VHDL/individuals/ind' + str(r) + '.vhdl')
    if params['SIMULATOR'] == 'ghdl':
        os.remove(r'../../VHDL/individuals//' + str(r) + '-obj08.cf')
    elif params['SIMULATOR'] == 'nvc':
        shutil.rmtree(r'../../VHDL/individuals//' + str(r))
    
    result_lines = result.replace("\r", "")

    #split the results (each line is splitted in three pieces and we want the last one)
    results_splitted = result_lines.split("'")
    
    yhat = results_splitted[1:len(results_splitted):2]

    if params['PROBLEM'] == 'ssd':
        for i in range(len(yhat)):
            yhat[i] = int(yhat[i],16)
    
    elif params['PROBLEM'] == 'multiplexer':
        for i in range(len(yhat)):
            yhat[i] = int(yhat[i])
    return yhat

class supervised_learning(base_ff):
    """
    Fitness function for supervised learning, ie regression and
    classification problems. Given a set of training or test data,
    returns the error between y (true labels) and yhat (estimated
    labels).

    We can pass in the error metric and the dataset via the params
    dictionary. Of error metrics, eg RMSE is suitable for regression,
    while F1-score, hinge-loss and others are suitable for
    classification.

    This is an abstract class which exists just to be subclassed:
    should not be instantiated.
    """

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Get training and test data
        self.training_in, self.training_exp, self.test_in, self.test_exp = \
            get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])

        # Find number of variables.
        self.n_vars = np.shape(self.training_in)[0]

        # Regression/classification-style problems use training and test data.
        if params['DATASET_TEST']:
            self.training_test = True

    def evaluate(self, ind, **kwargs):
        """
        Note that math functions used in the solutions are imported from either
        utilities.fitness.math_functions or called from numpy.

        :param ind: An individual to be evaluated.
        :param kwargs: An optional parameter for problems with training/test
        data. Specifies the distribution (i.e. training or test) upon which
        evaluation is to be performed.
        :return: The fitness of the evaluated individual.
        """

        dist = kwargs.get('dist', 'training')

        if dist == "training":
            # Set training datasets.
            x = self.training_in
            #print(x.dtype)
            y = self.training_exp

        elif dist == "test":
            # Set test datasets.
            x = self.test_in
            y = self.test_exp

        else:
            raise ValueError("Unknown dist: " + dist)

        if params['OPTIMIZE_CONSTANTS']:
            # if we are training, then optimize the constants by
            # gradient descent and save the resulting phenotype
            # string as ind.phenotype_with_c0123 (eg x[0] +
            # c[0] * x[1]**c[1]) and values for constants as
            # ind.opt_consts (eg (0.5, 0.7). Later, when testing,
            # use the saved string and constants to evaluate.
            if dist == "training":
                return optimize_constants(x, y, ind)

            else:
                # this string has been created during training
                phen = ind.phenotype_consec_consts
                c = ind.opt_consts
                # phen will refer to x (ie test_in), and possibly to c
                yhat = eval(phen)
                assert np.isrealobj(yhat)

                # let's always call the error function with the
                # true values first, the estimate second
                return params['ERROR_METRIC'](y, yhat)

        else:
            if params['PROBLEM_TYPE'] == 'vhdl':
                yhat = eval_vhdl(ind.phenotype)
            else:
                yhat = eval(ind.phenotype)
                if np.shape(y) != np.shape(yhat):
                    l = len(y)
                    temp = yhat
                    yhat = np.zeros([l], dtype=float)
                    yhat[:] = temp
            #    y = (y > 0)
            #    yhat = (yhat > 0)
            
            if params['SAMPLING'] == 'interleaved':
                if stats['gen'] % 2 == 0: #even
                    pass
                else: #odd
                    r = random.randint(0,len(y)-1)
                    y = y[r]
                    yhat = yhat[r]
            elif params['SAMPLING'] == 'interleaved_p':
                if stats['gen'] % 2 == 0: #even
                    pass
                else: #odd
                    list_indexes = list(range(len(y)))
                    random.shuffle(list_indexes)
                    r = random.random() #per cent between 0 and 100%
                    l = max(1,int(len(y)*r)) #number of samples used
                    y = y[list_indexes[0:l]]
                    yhat = [yhat[i] for i in list_indexes[0:l]]
                    
            
            assert np.isrealobj(yhat)
            
            if params['PENALTY_COEFFICIENT']:
                error = params['ERROR_METRIC'](y, yhat)
                if error == 0: #just apply penalties to individuals with perfect score
                    error = ind.nodes/params['PENALTY_COEFFICIENT']
            else:
                error = params['ERROR_METRIC'](y, yhat)
            
            if params['lexicase']:
#                y = (y > 0)
#                yhat = (yhat > 0)
                self.predict_result = np.equal(y,yhat)
                return error, self.predict_result
            else:
                # let's always call the error function with the true
                # values first, the estimate second
                return error
