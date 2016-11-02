#!/usr/bin/env python
"""Python script for the COCO experimentation module `cocoex`.

Usage from a system shell::

    python example_experiment.py bbob

runs a full but short experiment on the bbob suite. The optimization
algorithm used is determined by the `SOLVER` attribute in this file.

    python example_experiment.py bbob 20

runs the same experiment but with a budget of 20 * dimension
f-evaluations.

    python example_experiment.py bbob-biobj 1e3 1 20

runs the first of 20 batches with maximal budget of
1000 * dimension f-evaluations on the bbob-biobj suite.
All batches must be run to generate a complete data set.

Usage from a python shell::

    >>> import example_experiment as ee
    >>> ee.suite_name = "bbob-biobj"
    >>> ee.main(5, 100, 100)  # doctest: +ELLIPSIS
    Benchmarking solver...

runs the last of 100 batches with budget 5 * dimension.

Calling `example_experiment` without parameters prints this
help and the available suite names.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
del absolute_import, division, print_function, unicode_literals
try: range = xrange
except NameError: pass
import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
import cocoex
from cocoex import Suite, Observer, log_level
verbose = 1

import math
try: import cma  # cma.fmin is a solver option, "pip install cma" installs cma
except: pass
try: from scipy.optimize import fmin_slsqp  # "pip install scipy" installs scipy
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass

import random

def default_observers():
    """return a map from suite names to default observer names"""
    # this is a function only to make the doc available and
    # because @property doesn't work on module level
    return {'bbob':'bbob',
            'bbob-largescale':'bbob',  # todo: needs to be confirmed
            'bbob-constraint':'bbob',  # todo: needs to be confirmed
            'bbob-biobj': 'bbob-biobj'}

def print_flush(*args):
    """print without newline and flush"""
    print(*args, end="")
    sys.stdout.flush()


def ascetime(sec):
    """return elapsed time as str.

    Example: return `"0h33:21"` if `sec == 33*60 + 21`. 
    """
    h = sec / 60**2
    m = 60 * (h - h // 1)
    s = 60 * (m - m // 1)
    return "%dh%02d:%02d" % (h, m, s)


class ShortInfo(object):
    """print minimal info during benchmarking.

    After initialization, to be called right before the solver is called with
    the respective problem. Prints nothing if only the instance id changed.

    Example output:

        Jan20 18h27:56, d=2, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

        Jan20 18h27:56, d=3, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

        Jan20 18h27:57, d=5, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

    """
    def __init__(self):
        self.f_current = None  # function id (not problem id)
        self.d_current = 0  # dimension
        self.t0_dimension = time.time()
        self.evals_dimension = 0
        self.evals_by_dimension = {}
        self.runs_function = 0
    def print(self, problem, end="", **kwargs):
        print(self(problem), end=end, **kwargs)
        sys.stdout.flush()
    def add_evals(self, evals, runs):
        self.evals_dimension += evals
        self.runs_function += runs
    def dimension_done(self):
        self.evals_by_dimension[self.d_current] = (time.time() - self.t0_dimension) / self.evals_dimension
        s = '\n    done in %.1e seconds/evaluation' % (self.evals_by_dimension[self.d_current])
        # print(self.evals_dimension)
        self.evals_dimension = 0
        self.t0_dimension = time.time()
        return s
    def function_done(self):
        s = "(%d)" % self.runs_function + (2 - int(np.log10(self.runs_function))) * ' '
        self.runs_function = 0
        return s
    def __call__(self, problem):
        """uses `problem.id` and `problem.dimension` to decide what to print.
        """
        f = "f" + problem.id.lower().split('_f')[1].split('_')[0]
        res = ""
        if self.f_current and f != self.f_current:
            res += self.function_done() + ' '
        if problem.dimension != self.d_current:
            res += '%s%s, d=%d, running: ' % (self.dimension_done() + "\n\n" if self.d_current else '',
                        ShortInfo.short_time_stap(), problem.dimension)
            self.d_current = problem.dimension
        if f != self.f_current:
            res += '%s' % f
            self.f_current = f
        # print_flush(res)
        return res
    def print_timings(self):
        print("  dimension seconds/evaluations")
        print("  -----------------------------")
        for dim in sorted(self.evals_by_dimension):
            print("    %3d      %.1e " %
                  (dim, self.evals_by_dimension[dim]))
        print("  -----------------------------")
    @staticmethod
    def short_time_stap():
        l = time.asctime().split()
        d = l[0]
        d = l[1] + l[2]
        h, m, s = l[3].split(':')
        return d + ' ' + h + 'h' + m + ':' + s

# ===============================================
# prepare (the most basic example solver)
# ===============================================
def random_search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between `lbounds` and `ubounds`."""
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    population = lbounds + (ubounds - lbounds) * np.random.rand(alpha, len(lbounds))
    F = np.array([fun(x) for x in population])
    budget -= len(population)
    maxGenerationNumber = 0
    
    while True:        
        if fun.number_of_objectives == 2:
            if (maxGenerationNumber==max_geretation or budget<=0):
                pareto = non_dominated_selection(population,indicator)
                break
            else:
                t1 = time.time()
                FN,fitness,indicator,max_indicator = fitness_assignment(F)
                t2 = time.time()
                #print("time step2:",t2-t1)
                               
                t1 = time.time()
                population,F,FN,fitness,indicator = environmental_selection(population,F,FN,fitness,indicator,max_indicator);
                t2 = time.time()
                #print("time step3:",t2-t1) 
                t1 = time.time()
                
                parent_population = binary_tournament_selection(population, F, fitness)
                t2 = time.time()
                #print("time step5:",t2-t1) 
                t1 = time.time()

                mutationBabyPopulation = variation(parent_population);
                
                F2 = [fun(x) for x in mutationBabyPopulation];
                budget -= len(mutationBabyPopulation);
                F = np.concatenate((F,F2),axis=0);
                population = np.concatenate((population, mutationBabyPopulation),axis=0);
                t2 = time.time()
                #print("time step6:",t2-t1) 
                #    return (population, F,FN, fitness, indicator);
            maxGenerationNumber += 1
    return pareto

def fitness_assignment(F):
    minf1,minf2 = np.amin(F, axis=0)
    maxf1,maxf2 = np.amax(F, axis=0)
    FN = np.array(F,dtype=float)
    fitness = np.zeros(len(FN))


    FN[:,0] = (F[:,0]-minf1)/(maxf1-minf1)
    FN[:,1] = (F[:,1]-minf2)/(maxf2-minf2)
    
    
    indicator= np.zeros((len(FN),len(FN)));

    max_indicator = 0;  
    indicator = [(indicator_value(FN[x],FN[y],referencePointZ)) for x, y in np.ndindex(len(indicator),len(indicator))]
    indicator =  np.reshape(indicator, (math.sqrt(len(indicator)), math.sqrt(len(indicator))));
    max_indicator = np.amax(np.absolute(indicator))
    
    
    fitness = np.array([-np.exp(-indicator[x,y]/((max_indicator * k_factor))) for x,y in np.ndindex(len(indicator),len(indicator))])    
    fitness =  fitness.reshape(len(indicator),len(indicator));
    fitness = fitness.sum(axis=0)
    

    '''
    for f1 in range(len(FN)):
        for f2 in range(len(FN)):
            if (f1 != f2):
                fitness[f1] -= np.exp(-indicator[f2,f1] / (max_indicator * k_factor));
    '''

    
    return FN,fitness,indicator,max_indicator
    
def indicator_value(x1, x2, referencePointZ):
    ix2 = abs(referencePointZ[0]-x2[0])*abs(referencePointZ[1]-x2[1]);
    ix1 = abs(referencePointZ[0]-x1[0])*abs(referencePointZ[1]-x1[1]);
    ix12 = abs(referencePointZ[0]-min(x1[0],x2[0]))*abs(referencePointZ[1]-min(x1[1],x2[1]))-(max(x1[0],x2[0])-min(x1[0],x2[0]))*(max(x1[1],x2[1])-min(x1[1],x2[1]));
    if(x1[0]<x2[0] and x1[1]<x2[1]):
        return ix2-ix1;
    elif(x1[0]>x2[0] and x1[1]>x2[1]):
        return ix2-ix1;
    else:
        return ix12-ix1;    
        

    
def environmental_selection(population,F,FN,fitness,indicator,max_indicator):
    while (len(F) > alpha):
        
        index_min_fitness = np.argmin(fitness);
        F=np.delete(F,index_min_fitness,0)
        FN=np.delete(FN,index_min_fitness,0);

        
        for i in range(len(fitness)):
            fitness[i] = fitness[i] + np.exp((-indicator[index_min_fitness,i]) / (max_indicator * k_factor));
      
        
        population=np.delete(population,index_min_fitness, 0)
        fitness=np.delete(fitness,index_min_fitness, 0)
        indicator = np.delete(np.delete(indicator,index_min_fitness,1),index_min_fitness,0)
    
    return (population, F,FN, fitness, indicator);
    
def binary_tournament_selection(population, F, fitness):
    maxParentPopulation = int(len(population)/2);
    #print("max parent pop ", maxParentPopulation)
    parentPopulation = np.zeros((maxParentPopulation,len(population[0])));
    parentPopulationCounter = 0;
    
    #print(pArray);
    while parentPopulationCounter < maxParentPopulation:
        parentIndex =  np.random.randint(len(fitness), size= 2);
        #print('parent index: ',parentIndex)
        if(fitness[parentIndex[0]] <= fitness[parentIndex[1]]):
            #print('parent 1: ',pArray[parentIndex[1]])
            parentPopulation[parentPopulationCounter] = population[parentIndex[1]];
        else:
            #print('parent 2: ',pArray[parentIndex[0]])
            parentPopulation[parentPopulationCounter] = population[parentIndex[0]];
        parentPopulationCounter += 1;
        
    #print('parents',parentPopulation)    
    return parentPopulation;

def variation(parentPopulation):
    
    recombinationBabyPopulation = recombination(parentPopulation);
    mutationBabyPopulation = mutation(recombinationBabyPopulation);
    
    return mutationBabyPopulation
    
def recombination(parentPopulation):
    recombinationBabyPopulation = np.zeros((0,len(parentPopulation[0])));

    while len(parentPopulation) >= 2:
        parents = random.sample(range(len(parentPopulation)), 2);
        #print("parents",parents)
        a1 = random.uniform(-0.25,1.25);
        a2 = random.uniform(-0.25,1.25);  
                        
        baby1 = np.zeros(len(parentPopulation[parents[0]]));
        baby2 = np.zeros(len(parentPopulation[parents[0]]));
        
        for k in range(len(parentPopulation[parents[0]])):
            baby1[k]= parentPopulation[ parents[0] ][k]*a1 + parentPopulation[ parents[1] ][k]*(1-a1);
            baby2[k]= parentPopulation[ parents[1] ][k]*a2 + parentPopulation[ parents[0] ][k]*(1-a2);
        recombinationBabyPopulation = np.append(recombinationBabyPopulation, [baby1], axis = 0);                
        recombinationBabyPopulation = np.append(recombinationBabyPopulation, [baby2], axis = 0);
        parentPopulation = np.delete(parentPopulation,parents, 0)
        #print(recombinationBabyPopulation);
        
    return recombinationBabyPopulation
    
def mutation(babyPopulation):
    #mutationBabyPopulation = np.zeros((0,len(babyPopulation[0])));
    possibilityThreshold = 0.01;
    for baby in babyPopulation:
        possibility = np.random.random();
        if (possibility < possibilityThreshold) :
            
            normalisation = random.normalvariate(0, 1);

            normalisationArray = [random.normalvariate(0,1) for i in babyPopulation];
            sigmaValueArray = [pow(10,-2) for i in babyPopulation];
            
            t1 = 1/ math.sqrt(2*len(babyPopulation));
            t2 = 1/ math.sqrt(2*math.sqrt(len(babyPopulation)))
            for i in range(len(sigmaValueArray)):
                sigmaValueArray[i] = sigmaValueArray[i]* math.exp(t1*normalisation + t2*normalisationArray[i]);
                babyPopulation[i] = babyPopulation[i] + sigmaValueArray[i]*normalisationArray[i];
                
    return babyPopulation;
    
    
    

def non_dominated_selection(population, indicator):    
    paretoSetApproximation = np.array(population)
    listDominatedPoint = []

    for i in range(indicator.shape[1]):
        if(indicator.min(0)[i]<0):
            listDominatedPoint.append(i)
    
    paretoSetApproximation=np.delete(paretoSetApproximation, listDominatedPoint,0) 

   
    return paretoSetApproximation

# ===============================================
# loops over a benchmark problem suite
# ===============================================
def batch_loop(solver, suite, observer, budget,
               max_runs, current_batch, number_of_batches):
    """loop over all problems in `suite` calling
    `coco_optimize(solver, problem, budget * problem.dimension, max_runs)`
    for each eligible problem.

    A problem is eligible if
    `problem_index + current_batch - 1` modulo `number_of_batches`
    equals to zero.
    """
    addressed_problems = []
    short_info = ShortInfo()
    for problem_index, problem in enumerate(suite):
        if (problem_index + current_batch - 1) % number_of_batches:
            continue
        observer.observe(problem)
        short_info.print(problem) if verbose else None
        runs = coco_optimize(solver, problem, budget * problem.dimension, max_runs)
        if verbose:
            print_flush("!" if runs > 2 else ":" if runs > 1 else ".")
        short_info.add_evals(problem.evaluations, runs)
        problem.free()
        addressed_problems += [problem.id]
    print(short_info.function_done() + short_info.dimension_done())
    short_info.print_timings()
    print("  %s done (%d of %d problems benchmarked%s)" %
           (suite_name, len(addressed_problems), len(suite),
             ((" in batch %d of %d" % (current_batch, number_of_batches))
               if number_of_batches > 1 else "")), end="")
    if number_of_batches > 1:
        print("\n    MAKE SURE TO RUN ALL BATCHES", end="")
    return addressed_problems

#===============================================
# interface: ADD AN OPTIMIZER BELOW
#===============================================
def coco_optimize(solver, fun, max_evals, max_runs=1e9):
    """`fun` is a callable, to be optimized by `solver`.

    The `solver` is called repeatedly with different initial solutions
    until either the `max_evals` are exhausted or `max_run` solver calls
    have been made or the `solver` has not called `fun` even once
    in the last run.

    Return number of (almost) independent runs.
    """
    range_ = fun.upper_bounds - fun.lower_bounds
    center = fun.lower_bounds + range_ / 2
    if fun.evaluations:
        print('WARNING: %d evaluations were done before the first solver call' %
              fun.evaluations)

    for restarts in range(int(max_runs)):
        remaining_evals = max_evals - fun.evaluations
        x0 = center + (restarts > 0) * 0.8 * range_ * (
                np.random.rand(fun.dimension) - 0.5)
        fun(x0)  # can be incommented, if this is done by the solver

        if solver.__name__ in ("random_search", ):
            solver(fun, fun.lower_bounds, fun.upper_bounds,
                   remaining_evals)
        elif solver.__name__ == 'fmin' and solver.__globals__['__name__'] in ['cma', 'cma.evolution_strategy', 'cma.es']:
            if x0[0] == center[0]:
                sigma0 = 0.02
                restarts_ = 0
            else:
                x0 = "%f + %f * np.random.rand(%d)" % (
                        center[0], 0.8 * range_[0], fun.dimension)
                sigma0 = 0.2
                restarts_ = 6 * (observer_options.find('IPOP') >= 0)

            solver(fun, x0, sigma0 * range_[0], restarts=restarts_,
                   options=dict(scaling=range_/range_[0], maxfevals=remaining_evals,
                                termination_callback=lambda es: fun.final_target_hit,
                                verb_log=0, verb_disp=0, verbose=-9))
        elif solver.__name__ == 'fmin_slsqp':
            solver(fun, x0, iter=1 + remaining_evals / fun.dimension,
                   iprint=-1)
############################ ADD HERE ########################################
        # ### IMPLEMENT HERE THE CALL TO ANOTHER SOLVER/OPTIMIZER ###
        # elif True:
        #     CALL MY SOLVER, interfaces vary
##############################################################################
        else:
            raise ValueError("no entry for solver %s" % str(solver.__name__))

        if fun.evaluations >= max_evals or fun.final_target_hit:
            break
        # quit if fun.evaluations did not increase
        if fun.evaluations <= max_evals - remaining_evals:
            if max_evals - fun.evaluations > fun.dimension + 1:
                print("WARNING: %d evaluations remaining" %
                      remaining_evals)
            if fun.evaluations < max_evals - remaining_evals:
                raise RuntimeError("function evaluations decreased")
            break
    return restarts + 1

# ===============================================
# set up: CHANGE HERE SOLVER AND FURTHER SETTINGS AS DESIRED
# ===============================================
######################### CHANGE HERE ########################################
# CAVEAT: this might be modified from input args
alpha = 100 # population size
max_geretation = 150 # max number of generations
possibilityThreshold = 0.1;

k_factor = 0.05 # fitness scaling factor 
budget = 1000  # maxfevals = budget x dimension ### INCREASE budget WHEN THE DATA CHAIN IS STABLE ###
max_runs = 1e9  # number of (almost) independent trials per problem instance
number_of_batches = 1  # allows to run everything in several batches
current_batch = 1      # 1..number_of_batches
referencePointZ = np.array([2,2]);

##############################################################################
SOLVER = random_search
#SOLVER = my_solver # fmin_slsqp # SOLVER = cma.fmin
suite_name = "bbob-biobj"
# suite_name = "bbob"
suite_instance = "year:2016"
suite_options = "dimensions: 2,3,5,10,20"  # "dimensions: 2,3,5,10,20 "  # if 40 is not desired
observer_name = default_observers()[suite_name]
observer_options = (
    ' result_folder: %s_on_%s_budget%04dxD '
                 % (SOLVER.__name__, suite_name, budget) +
    ' algorithm_name: %s ' % SOLVER.__name__ +
    ' algorithm_info: "A SIMPLE RANDOM SEARCH ALGORITHM" ')  # CHANGE THIS
######################### END CHANGE HERE ####################################

# ===============================================
# run (main)
# ===============================================
def main(budget=budget,
         max_runs=max_runs,
         current_batch=current_batch,
         number_of_batches=number_of_batches):
    """Initialize suite and observer, then benchmark solver by calling
    `batch_loop(SOLVER, suite, observer, budget,...`.
    """
    observer = Observer(observer_name, observer_options)
    suite = Suite(suite_name, suite_instance, suite_options)
    print("Benchmarking solver '%s' with budget=%d*dimension on %s suite, %s"
          % (' '.join(str(SOLVER).split()[:2]), budget,
             suite.name, time.asctime()))
    if number_of_batches > 1:
        print('Batch usecase, make sure you run *all* %d batches.\n' %
              number_of_batches)
    t0 = time.clock()
    batch_loop(SOLVER, suite, observer, budget, max_runs,
               current_batch, number_of_batches)
    print(", %s (%s total elapsed time)." % (time.asctime(), ascetime(time.clock() - t0)))

# ===============================================
if __name__ == '__main__':
    """read input parameters and call `main()`"""
    if len(sys.argv) < 2 or sys.argv[1] in ["--help", "-h"]:
            print(__doc__)
            print("Recognized suite names: " + str(cocoex.known_suite_names))
            exit(0)
    suite_name = sys.argv[1]
    observer_name = default_observers()[suite_name]
    if len(sys.argv) > 2:
        budget = float(sys.argv[2])
        if observer_options.find('budget') > 0:  # reflect budget in folder name
            idx = observer_options.find('budget')
            observer_options = observer_options[:idx+6] + \
                "%04d" % int(budget + 0.5) + observer_options[idx+10:]
    if len(sys.argv) > 3:
        current_batch = int(sys.argv[3])
    if len(sys.argv) > 4:
        number_of_batches = int(sys.argv[4])
    if len(sys.argv) > 5:
        messages = ['Argument "%s" disregarded (only 4 arguments are recognized).' % sys.argv[i]
            for i in range(5, len(sys.argv))]
        messages.append('See "python example_experiment.py -h" for help.')
        raise ValueError('\n'.join(messages))
    main(budget, max_runs, current_batch, number_of_batches)
