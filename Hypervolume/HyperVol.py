#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals
try: range = xrange
except NameError: pass
import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
import cocoex
from cocoex import Suite, Observer, log_level
verbose = 1

import random 

try: import cma  # cma.fmin is a solver option, "pip install cma" installs cma
except: pass
try: from scipy.optimize import fmin_slsqp  # "pip install scipy" installs scipy
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass


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
def Hypervolume_Indicator_Based_Selection_Multiobjective_Search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between `lbounds` and `ubounds`."""
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim= len(lbounds)
    alphaValuePopulationSize = 100; # population size
    initialPopulationP = lbounds + (ubounds - lbounds) * np.random.rand(alphaValuePopulationSize, dim);
    F = [fun(x) for x in initialPopulationP];
    kValue = 0.05; # fitness scaling factor
    maxGenerationNumber = 20; # max number of generation
    referencePointZ = np.array([2,2]);
    max_chunk_size = 1 + 4e4 / dim
  

    while budget > 0:        
        chunk = int(min([budget, max_chunk_size]))
        mValueCounter = 0;
        
        if fun.number_of_objectives == 2:
            print('Start')           
            # population initalization       
            while True:
                #print(budget)
                #print(mValueCounter)
            
                #Step 2 - Fitness assignment:
                #return (pArray, indicatorArray,fitnessArray,cValueMaxIndicator);
                arrayStep2 = fitness_assignment(F,referencePointZ,kValue);
                F = arrayStep2[0];

                #Step 3 - Environmental selection: 
                #return (initialPopulationP, pArray, fitnessArray, indicatorArray);
                arrayStep3 = environmental_selection(initialPopulationP, F,arrayStep2[1],arrayStep2[2],alphaValuePopulationSize,arrayStep2[3],kValue);
                initialPopulationP = arrayStep3[0];
                #print("len population apres step3",len(initialPopulationP))

                F = arrayStep3[1];
                indicatorArray = arrayStep3[3];
                
                #Step 4 - Termination:
                if mValueCounter >= maxGenerationNumber:
                    paretoSetApproximation,indicatorArray = non_dominated_selection(initialPopulationP, indicatorArray);
                    print('End')
                    break;
                
                #Step 5 - Mating selection:
                #return parentPopulation;
                arrayStep5 = binary_tournament_selection(initialPopulationP, F,arrayStep3[2]);
                
                
                #Step 6 - Variation:
                #return variationPopulationP
                mutationBabyPopulation = variation(arrayStep5);
                F2 = [fun(x) for x in mutationBabyPopulation];
                F = np.concatenate((F,F2),axis=0);
                initialPopulationP = np.concatenate((initialPopulationP, mutationBabyPopulation),axis=0);

                
                mValueCounter += 1;
                   
        budget -= chunk
    print("outttt")
        
    return paretoSetApproximation;
    
    
def non_dominated_selection(initialPopulationP, indicatorArray):    
    paretoSetApproximation = np.array(initialPopulationP)
    listDominatedPoint = []

    for i in range(indicatorArray.shape[1]):
        if(indicatorArray.min(0)[i]<0):
            listDominatedPoint.append(i)
    
    paretoSetApproximation=np.delete(paretoSetApproximation, listDominatedPoint,0) 
    indicatorArray=np.delete(indicatorArray,listDominatedPoint,0)
    indicatorArray=np.delete(indicatorArray,listDominatedPoint,1)
   
    return (paretoSetApproximation,indicatorArray)
    
def variation(parentPopulation):
    
    recombinationBabyPopulation = recombination(parentPopulation);
    mutationBabyPopulation = mutation(recombinationBabyPopulation);
    
    
    #print('recombinationBabyPopulation: ', len(recombinationBabyPopulation));
    #print('mutationBabyPopulation', len(mutationBabyPopulation));
    
    return mutationBabyPopulation; 

def mutation(babyPopulation):
    #mutationBabyPopulation = np.zeros((0,len(babyPopulation[0])));
    possibilityThreshold = 0.01;
    for baby in babyPopulation:
        possibility = np.random.random();
        if (possibility < possibilityThreshold) :
            index1 = np.random.randint(len(babyPopulation[0]));
            index2 = np.random.randint(len(babyPopulation[0]));
            if(index1 != index2):
                temporalValueBaby = baby[index1];
                baby[index1] = baby[index2];
                baby[index2] = temporalValueBaby
    
    return babyPopulation;
    
def recombination(parentPopulation):
    localisationValue = np.random.randint(1,len(parentPopulation[0]));
    recombinationBabyPopulation = np.zeros((0,len(parentPopulation[0])));

    
    
    for i in range(len(parentPopulation)):
        for j in range(i+1,len(parentPopulation)):
            a1 = random.uniform(-0.25,1.25);
            a2 = random.uniform(-0.25,1.25);
            if (i != j ) :   
                
                baby1 = np.zeros(len(parentPopulation[i]));
                baby2 = np.zeros(len(parentPopulation[i]));

                for k in range(len(parentPopulation[0])):
                    baby1[k]= parentPopulation[i][k]*a1 + parentPopulation[j][k]*(1-a1);
                    baby2[k]= parentPopulation[j][k]*a2 + parentPopulation[i][k]*(1-a2);
                    recombinationBabyPopulation = np.append(recombinationBabyPopulation, [baby1], axis = 0);                
                    recombinationBabyPopulation = np.append(recombinationBabyPopulation, [baby2], axis = 0);
 
                 
               
            #if(i==1 and j==2):
            #    print("parent 1",parentPopulation[i]);
            #    print("parent 2",parentPopulation[j]);

            #    print("baby 1",baby1);
            #    print("baby 2",baby2);
    return np.array(random.sample(recombinationBabyPopulation,3*len(parentPopulation)));
    #return recombinationBabyPopulation;

def binary_tournament_selection(initialPopulationP, pArray, fitnessArray):
    maxParentPopulation = len(initialPopulationP);
    
    parentPopulation = np.zeros((maxParentPopulation,len(initialPopulationP[0])));
    parentPopulationCounter = 0;
    #print(pArray);
    while parentPopulationCounter < maxParentPopulation:
        parentIndex =  np.random.randint(len(fitnessArray), size= 2);
        #print('parent index: ',parentIndex)
        if(fitnessArray[parentIndex[0]] <= fitnessArray[parentIndex[1]]):
            #print('parent 1: ',pArray[parentIndex[1]])
            parentPopulation[parentPopulationCounter] = initialPopulationP[parentIndex[1]];
        else:
            #print('parent 2: ',pArray[parentIndex[0]])
            parentPopulation[parentPopulationCounter] = initialPopulationP[parentIndex[0]];
        parentPopulationCounter += 1;
        
    #print('parents',parentPopulation)    
    return parentPopulation;
    

    
def fitness_assignment(pArray, referencePointZ, kValue):
    
    pArray = np.array(pArray,dtype=float);
    fitnessArray = np.zeros(len(pArray))
    
    minf1 = min(pArray[:,0]);
    maxf1 = max(pArray[:,0]);
    minf2 = min(pArray[:,1]);
    maxf2 = max(pArray[:,1]);
        
    pArray[:,0]=(pArray[:,0]- minf1)/(maxf1-minf1);
    pArray[:,1]=(pArray[:,1]- minf2)/(maxf2-minf2);
        
        
    indicatorArray = np.zeros((len(pArray),len(pArray)));
    cValueMaxIndicator = 0;
        
    for f1 in range(len(pArray)):
        for f2 in range(len(pArray)):
            indicatorArray[f1,f2]=indicator_value(pArray[f1],pArray[f2],referencePointZ);
        
    cValueMaxIndicator = np.amax(np.absolute(indicatorArray))
        
    for f1 in range(len(pArray)):
        for f2 in range(len(pArray)):
            if (f1 != f2):
                fitnessArray[f1] -= np.exp(-indicatorArray[f2,f1] / (cValueMaxIndicator * kValue));
                
    #print (pArray)
    return (pArray, indicatorArray, fitnessArray, cValueMaxIndicator);
        
def environmental_selection(initialPopulationP, pArray, indicatorArray, fitnessArray, alphaValue, cValueMaxIndicator, kValue): 

    while (len(pArray) > alphaValue):
        
        minFitnessIndex = np.argmin(fitnessArray);
        
        pArray=np.delete(pArray,minFitnessIndex,0)
        
        
        for i in range(len(fitnessArray)):
            for j in range(len(fitnessArray)):
                if i!=j:
                	fitnessArray[i] = fitnessArray[i] + np.exp((-indicatorArray[minFitnessIndex,j]) / (cValueMaxIndicator * kValue));
                
        initialPopulationP=np.delete(initialPopulationP,minFitnessIndex, 0)
        fitnessArray=np.delete(fitnessArray,minFitnessIndex, 0)
        indicatorArray = np.delete(np.delete(indicatorArray,minFitnessIndex,1),minFitnessIndex,0)
    
    return (initialPopulationP, pArray, fitnessArray, indicatorArray);
    
    
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
        runs = coco_optimize(solver, problem, budget, max_runs)
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

        if solver.__name__ in ("Hypervolume_Indicator_Based_Selection_Multiobjective_Search", ):
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
budget = 10  # maxfevals = budget x dimension ### INCREASE budget WHEN THE DATA CHAIN IS STABLE ###
max_runs = 1e9  # number of (almost) independent trials per problem instance
number_of_batches = 1  # allows to run everything in several batches
current_batch = 1      # 1..number_of_batches
##############################################################################
SOLVER = Hypervolume_Indicator_Based_Selection_Multiobjective_Search
#SOLVER = my_solver # fmin_slsqp # SOLVER = cma.fmin
suite_name = "bbob-biobj"
#suite_name = "bbob"
suite_instance = "year:2016"
suite_options = "dimensions: 2"  # "dimensions: 2,3,5,10,20 "  # if 40 is not desired
observer_name = suite_name
observer_options = (
    ' result_folder: %s_on_%s_budget%04dxD '
                 % (SOLVER.__name__, suite_name, budget) +
    ' algorithm_name: %s ' % SOLVER.__name__ +
    ' algorithm_info: "HYPERVOLUME INDICATOR BASED SELECTION SEARCH ALGORITHM" ')  # CHANGE THIS
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
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print(__doc__)
            exit(0)
        budget = float(sys.argv[1])
        if observer_options.find('budget') > 0:  # reflect budget in folder name
            idx = observer_options.find('budget')
            observer_options = observer_options[:idx+6] + \
                "%04d" % int(budget + 0.5) + observer_options[idx+10:]
    if len(sys.argv) > 2:
        current_batch = int(sys.argv[2])
    if len(sys.argv) > 3:
        number_of_batches = int(sys.argv[3])
    if len(sys.argv) > 4:
        messages = ['Argument "%s" disregarded (only 3 arguments are recognized).' % sys.argv[i]
            for i in range(4, len(sys.argv))]
        messages.append('See "python example_experiment.py -h" for help.')
        raise ValueError('\n'.join(messages))
    main(budget, max_runs, current_batch, number_of_batches)

