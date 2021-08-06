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
from stats.propagation_delay import count_gates_critical_path

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True, cwd=r'../../VHDL/individuals')
    proc_stdout = process.communicate()[0].strip()
    return proc_stdout.decode('utf-8')

def compare_bit(y,yhat, base='hexadecimal'):
    """"
    Inputs are both in hexadecimal and must have the same length.
    y: correct values
    yhat: predicted values
    
    Output has the same length as y or yhat and means the number of bits predicted correcly of each sample
    
    Inputs and output are lists.
    """
    if len(y) != len(yhat):
        print("error")
        return
    
    l = len(y)
    y_matrix = np.zeros([l,4], dtype=int)
    yhat_matrix = np.zeros([l,4], dtype=int)
    
#    print(y)
#    print(yhat)
    
    if base == 'hexadecimal':
        scale = 16 #hexadecimal 
        num_of_bits = 4
        for i in range(l):
            bin_value_y = bin(int(y[i], scale))[2:].zfill(num_of_bits)
            bin_value_yhat = bin(int(yhat[i], scale))[2:].zfill(num_of_bits)
            for j in range(4):
                y_matrix[i,j] = int(bin_value_y[j])
                yhat_matrix[i,j] = int(bin_value_yhat[j])
    elif base == 'decimal':
        for i in range(l):
            bin_value_y = '{0:04b}'.format(int(y[i]))
            bin_value_yhat = '{0:04b}'.format(int(yhat[i]))
            for j in range(4):
                y_matrix[i,j] = int(bin_value_y[j])
                yhat_matrix[i,j] = int(bin_value_yhat[j])
    
    compare_matrix = np.equal(y_matrix,yhat_matrix)    
    
    compare = np.sum(compare_matrix,axis=1).tolist()
    
#    print(compare)
#    print()
    return compare

def eval_vhdl_pop(individuals, y):
    r = random.randint(0,10**10)
    vhdl = open(r'../../VHDL/individuals/ind' + str(r) + '.vhdl','w+')
    tb = open(r'../../VHDL/individuals/tb' + str(r) + '.vhdl','w+')
    
    if params['PROBLEM'] == '2-bit_multiplier':
        vhdl.write("""library ieee; 
use ieee.std_logic_textio.all;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ind""" +  str(r) +  """ is 
port(a: in  STD_LOGIC_VECTOR (1 downto 0); 
     b: in  STD_LOGIC_VECTOR (1 downto 0); 
     o: out STD_LOGIC_VECTOR (3 downto 0)); 
end ind""" +  str(r) +  """;

""")
        tb.write("""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb""" +  str(r) +  """ is
end tb""" +  str(r) +  """;

architecture dataflow of tb""" +  str(r) +  """ is
    signal aa: STD_LOGIC_VECTOR (1 downto 0); 
    signal bb: STD_LOGIC_VECTOR (1 downto 0); 
""")
        n = 0
        if type(individuals) == list:
            for name, ind in enumerate(individuals):
                if ind.eval_ind:
                    phenotype = ind.phenotype
                    phenotype = phenotype.replace('dataflow', 'dataflow' + str(r) + str(n))
                    phenotype = phenotype.replace('ind', 'ind' + str(r))
                    vhdl.write(phenotype + "\n\n")
                    tb.write("    signal o" + str(r) + str(n) + ": STD_LOGIC_VECTOR (3 downto 0);\n")
                    n += 1
        else: #there is just one ind in individuals
            phenotype = individuals.phenotype
            phenotype = phenotype.replace('dataflow', 'dataflow' + str(r) + str(n))
            phenotype = phenotype.replace('ind', 'ind' + str(r))
            vhdl.write(phenotype + "\n\n")
            tb.write("    signal o" + str(r)  + str(n) + ": STD_LOGIC_VECTOR (3 downto 0);\n")
            n += 1
            
        tb.write("\nbegin\n")
        for i in range(n):
            tb.write("ind" + str(r) + str(i) + ": entity work.ind" + str(r) + "(dataflow" + str(r) + str(i) + ") port map (a => aa, b => bb, o => o" + str(r) + str(i) + ");\n")
        tb.write("""process 
variable count: std_logic_vector(3 downto 0);
begin 

for idx in 0 to 15 loop
count := std_logic_vector(to_unsigned(idx,4));
	
aa(1) <= count(0);
aa(0) <= count(1);
bb(1) <= count(2);
bb(0) <= count(3);

wait for 1 ns; 
report "'" """)

        for i in range(n):
            tb.write("& to_hstring(o" + str(r) + str(i) + ") & \"'\" ")
        tb.write(";\n\n end loop; wait; end process; end dataflow;")
    
    elif params['PROBLEM'] == 'multiplexer':
        vhdl.write("""library ieee; 
use ieee.std_logic_textio.all;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ind""" +  str(r) +  """ is 
port(d0: in std_ulogic; 
		 d1: in std_ulogic; 
		 d2: in std_ulogic; 
		 d3: in std_ulogic; 
		 d4: in std_ulogic; 
		 d5: in std_ulogic; 
		 d6: in std_ulogic; 
		 d7: in std_ulogic; 
		 a: in  STD_LOGIC_VECTOR (2 downto 0);
		 o: out std_ulogic); 
end ind""" +  str(r) +  """;

""")
        tb.write("""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb""" +  str(r) +  """ is
end tb""" +  str(r) +  """;

architecture behave of tb""" +  str(r) +  """ is
    signal aa: STD_LOGIC_VECTOR (2 downto 0); 
    signal d0, d1, d2, d3, d4, d5, d6, d7: std_ulogic; 
""")
        n = 0
        if type(individuals) == list:
            for name, ind in enumerate(individuals):
                if ind.eval_ind:
                    phenotype = ind.phenotype
                    phenotype = phenotype.replace('behave', 'behave' + str(r) + str(n))
                    phenotype = phenotype.replace('ind', 'ind' + str(r))
                    vhdl.write(phenotype + "\n\n")
                    tb.write("    signal o" + str(r) + str(n) + ": std_ulogic;\n")
                    n += 1
        else: #there is just one ind in individuals
            phenotype = individuals.phenotype
            phenotype = phenotype.replace('behave', 'behave' + str(r) + str(n))
            phenotype = phenotype.replace('ind', 'ind' + str(r))
            vhdl.write(phenotype + "\n\n")
            tb.write("    signal o" + str(r)  + str(n) + ": std_ulogic;\n")
            n += 1
            
        tb.write("\nbegin\n")
        for i in range(n):
            tb.write("ind" + str(r) + str(i) + ": entity work.ind" + str(r) + "(behave" + str(r) + str(i) + ") port map (d0 => d0, d1 => d1, d2 => d2, d3 => d3, d4 => d4, d5 => d5, d6 => d6, d7 => d7, a => aa, o => o" + str(r) + str(i) + ");\n")
        tb.write("""process 
variable count: std_logic_vector(10 downto 0);
begin 

for idx in 0 to 2047 loop
count := std_logic_vector(to_unsigned(idx,11));
	
d0 <= count(10);
d1 <= count(9);
d2 <= count(8);
d3 <= count(7);
d4 <= count(6);
d5 <= count(5);
d6 <= count(4);
d7 <= count(3);
aa(2) <= count(2);
aa(1) <= count(1);
aa(0) <= count(0);

wait for 1 ns; 
report "'" """)

        for i in range(n):
            tb.write("& std_ulogic'image(o" + str(r) + str(i) + ") & \"'\" ")
        tb.write(";\n\n end loop; wait; end process; end behave;")
        
   
    vhdl.close()
    tb.close()
    
    if params['SIMULATOR'] == 'ghdl':
        result = subprocess_cmd("ghdl -a --std=08 --work=" + str(r) + " ind" + str(r) + ".vhdl tb" + str(r) + ".vhdl ; ghdl -e --std=08 --work=" + str(r) + " tb" + str(r) + " ; ghdl -r --std=08  --work=" + str(r) + " tb"  + str(r))
    
    result_lines = result.replace("\r", "") #each line report the results from all individuals using one sample
    results_splitted = result_lines.split("'")
    results_splitted_final = []
    
    if params['PROBLEM'] == '2-bit_multiplier':
        for i in range(16):
            del results_splitted[i*n] #list with all results
    elif params['PROBLEM'] == 'multiplexer':
  #      for i in range(2048):
  #          del results_splitted[i*n] #list with all results
#        while '' in results_splitted:
#            results_splitted.remove('')
        for i in range(len(results_splitted)):
            if results_splitted[i] == '0' or results_splitted[i] == '1':
                results_splitted_final.append(results_splitted[i])
    
    if type(individuals) == list:
        idx = 0
        for name, ind in enumerate(individuals):
            if ind.eval_ind:
                
                if params['PROBLEM'] == '2-bit_multiplier':
                    yhat = results_splitted[idx:len(results_splitted)-1:n] #list with the results from ind
                elif params['PROBLEM'] == 'multiplexer':
                    yhat = results_splitted_final[idx:len(results_splitted_final):n] #list with the results from ind

                if params['PROBLEM'] == '2-bit_multiplier':
                    for j in range(len(yhat)):
                        yhat[j] = int(yhat[j],16)
                elif params['PROBLEM'] == 'multiplexer':
                    for i in range(len(yhat)):
                        yhat[i] = int(yhat[i])

                assert np.isrealobj(yhat)
                
                if params['PENALTY_COEFFICIENT']:
                    error = params['ERROR_METRIC'](y, yhat)
                    if error == 0:
                        if params['PENALTY_TYPE'] == 'gates':
                            l = len(params['GATES_TO_COUNT'])
                            n_gates = 0
                            for j in range(l):
                                n_gates += ind.phenotype.count(params['GATES_TO_COUNT'][j])
                            error = n_gates/params['PENALTY_COEFFICIENT']
                        elif params['PENALTY_TYPE'] == 'nodes':
                            error = ind.nodes/params['PENALTY_COEFFICIENT']
                        elif params['PENALTY_TYPE'] == 'gates_critical_path':
                            n_gates_critical_path = count_gates_critical_path(ind.phenotype,params['GATES_TO_COUNT'])
                            error = n_gates_critical_path/params['PENALTY_COEFFICIENT']
                else:
                    error = params['ERROR_METRIC'](y, yhat)
                
                if params['lexicase']:
                    if params['LEXICASE_EACH_BIT']:
                        ind.predict_result = compare_bit(y,yhat,'decimal')
                    else:
                        ind.predict_result = np.equal(y,yhat)
                        
                ind.fitness = error
                ind.n_samples = len(y)
                
                idx += 1 #update just if eval_ind = True, since this is indexing results_splitted

    else: #if len(population) = 1 or in the end of the evolution when the best individual is used in the test set
        
        if params['PROBLEM'] == '2-bit_multiplier':
            yhat = results_splitted[0:len(results_splitted)-1]
            for j in range(len(yhat)):
                yhat[j] = int(yhat[j],16)
        elif params['PROBLEM'] == 'multiplexer':
            yhat = results_splitted_final[0:len(results_splitted_final)] #list with the results from ind
            for i in range(len(yhat)):
                yhat[i] = int(yhat[i])
        assert np.isrealobj(yhat)
            
        if params['PENALTY_COEFFICIENT']:
            error = params['ERROR_METRIC'](y, yhat)
            if error == 0:
                if params['PENALTY_TYPE'] == 'gates':
                    l = len(params['GATES_TO_COUNT'])
                    n_gates = 0
                    for j in range(l):
                        n_gates += individuals.phenotype.count(params['GATES_TO_COUNT'][j])
                    error = n_gates/params['PENALTY_COEFFICIENT']
                elif params['PENALTY_TYPE'] == 'nodes':
                    error = individuals.nodes/params['PENALTY_COEFFICIENT']
                elif params['PENALTY_TYPE'] == 'gates_critical_path':
                            n_gates_critical_path = count_gates_critical_path(individuals.phenotype,params['GATES_TO_COUNT'])
                            error = n_gates_critical_path/params['PENALTY_COEFFICIENT']
        else:
            error = params['ERROR_METRIC'](y, yhat)
            
        
        if params['lexicase']:
            if params['LEXICASE_EACH_BIT']:
                ind.predict_result = compare_bit(y,yhat,'decimal')
            else:
                ind.predict_result = np.equal(y,yhat)
        individuals.fitness = error
        individuals.n_samples = len(y)
            
    os.remove(r'../../VHDL/individuals/tb' + str(r) + '.vhdl')
    os.remove(r'../../VHDL/individuals/ind' + str(r) + '.vhdl')
    if params['SIMULATOR'] == 'ghdl':
        os.remove(r'../../VHDL/individuals//' + str(r) + '-obj08.cf')
    
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
    
    elif params['PROBLEM'] == '2-bit_multiplier':
        with open(r'../../VHDL/2-bit multiplier/tb.vhdl', 'r') as file :
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
    
    elif params['PROBLEM'] == '2-bit multiplier':
        for i in range(len(yhat)):
            yhat[i] = int(yhat[i],16)
    
    return yhat

class supervised_learning_vhdl(base_ff):
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

        if params['ONE_FILE_PER_GENERATION'] == True:
            eval_vhdl_pop(ind, y) #all individuals
        else:    
            yhat = eval_vhdl(ind.phenotype)
        
        if params['ONE_FILE_PER_GENERATION'] == True:
            pass
        else:
            assert np.isrealobj(yhat)
            
            if params['PENALTY_COEFFICIENT']:
                error = params['ERROR_METRIC'](y, yhat)
                if error == 0:
                    if params['PENALTY_TYPE'] == 'gates':
                        l = len(params['GATES_TO_COUNT'])
                        n_gates = 0
                        for i in range(l):
                            n_gates += ind.phenotype.count(params['GATES_TO_COUNT'][i])
                        error = n_gates/params['PENALTY_COEFFICIENT']
                    elif params['PENALTY_TYPE'] == 'nodes':
                        error = ind.nodes/params['PENALTY_COEFFICIENT']
                    elif params['PENALTY_TYPE'] == 'gates_critical_path':
                        n_gates_critical_path = count_gates_critical_path(ind.phenotype,params['GATES_TO_COUNT'])
                        error = n_gates_critical_path/params['PENALTY_COEFFICIENT']
            else:
                error = params['ERROR_METRIC'](y, yhat)
            
            self.n_samples = len(y)
            if params['lexicase']:
                if params['LEXICASE_EACH_BIT']:
                    pass #TODO complete afterwards
                else:
                    self.predict_result = np.equal(y,yhat)
                    return error, self.predict_result
            else:
                # let's always call the error function with the true
                # values first, the estimate second
                return error#, self.n_samples