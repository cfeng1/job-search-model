import numpy as np
import pandas as pd
import itertools, functools
import time
from data_simulation_iteration_version import dataSimulationIteration
from data_simulation_recursion_version import dataSimulationRecursion

def ccp_fun_inefficient(data, T=10):
    ccp = np.zeros((T,T))
    # weight ccp by number of observations
    W = np.zeros((T,T))
    for age in range(0,T):
        for exp in range(0,T):
            num = len(data[(data['age'] == age) & (data['work_experience'] == exp) 
                & (data['choice'] == 2) ])
            den = len(data[(data['age'] == age) & (data['work_experience'] == exp)]) 
            W[age,exp] = den
            if den == 0:
                ccp[age,exp] = np.nan
            else:
                ccp[age,exp] = num/den        
    ccp_vec = ccp[np.tril_indices(T)]
    W = W[np.tril_indices(T)]/np.sum(W)
    ccp_vec[np.isnan(ccp_vec)] = 999   
    return ccp_vec, W

def ccp_fun(data, T=10):    
    def ccp_state_fun(arg):
        age , exp =  arg
        mask_den = (data['age'] == age) & (data['work_experience'] == exp)
        mask_num = (mask_den) & (data['choice'] == 2) 
        # weight ccp by number of observations
        W_state = len(data[mask_den])
        ccp_state = len(data[mask_num])/W_state if W_state>0 else 999
        return ccp_state, W_state
    output = [ccp_state_fun(item) for item in filter(lambda x: x[0]>=x[1], 
        itertools.product(range(T),range(T)))]
    ccp = np.array([item[0] for item in output])
    W = np.array([item[1] for item in output])
    W = W/np.sum(W) 
    return ccp, W

# estimation transition probability/success rate
def p_acpt_fun(data, T=10):
    data['future_work_experience'] = data['work_experience'].shift(-1).values.astype(int)
    mask = data.age == T-1
    data.loc[mask,'future_work_experience'] = 999    
    p_acpt = np.zeros((T-1,))
    for i in range(T-1):
        num = len(data[(data['age'] < T-1) & (data['work_experience'] == i) & 
            (data['future_work_experience'] == i + 1) & (data['choice'] == 2)])
        den = len(data[(data['age'] < T-1) & (data['work_experience'] == i) & 
            (data['choice'] == 2)]) 
        if den > 0:
            p_acpt[i] = num/den
        else:
            p_acpt[i] = np.nan
    return p_acpt

# minimizing distance between predicted CCP and actual CCP
# recursion version
def predictCCPRecursion(success_rates, parameters):
    data = dataSimulationRecursion(parameters, success_rates)
    ccp, W = ccp_fun(data)
    return ccp, W

def estimatorRecursion(parameters, actual_ccp, success_rates, actual_W):
    predicted_ccp, W = predictCCPRecursion(success_rates, parameters)
    distance = np.sum(np.multiply((predicted_ccp-actual_ccp)**2,W))
    return distance

# iteration version
def predictCCP(success_rates, parameters):
    data = dataSimulationIteration(success_rates, parameters)
    ccp, W = ccp_fun(data)
    return ccp, W

def estimatorIteration(actual_ccp,success_rates,parameters):
    predicted_ccp, W = predictCCP(success_rates, parameters)
    distance = np.sum(np.multiply((predicted_ccp-actual_ccp)**2,W))
    return distance

# resampling individuals with replacement
def getResample(data):
    individuals = list(set(data.individual.values))
    groups = data.groupby('individual')
    subsample = pd.concat([groups.get_group(i) for i in resample(
                            individuals, replace=True, n_samples=len(individuals))])
    return subsample

# calculate standard errors
def standardErrors(coefficients):
    avg = np.average(coefficients)
    se = np.sqrt(np.sum((coefficients-avg)**2)/(len(coefficients)-1))
    return se

# parallel bootstrap
def estimationRecursionGrid(args):
    # Here the underlying optimization uses grid search
    # we can implement other optimization methods
    data_recursion, parameter_combos = args
    actual_ccp, actual_W = ccp_fun(data_recursion)
    # estimate success rates
    T = 10
    success_rates = np.zeros((T,)) 
    success_rates[0:T-1] = p_acpt_fun(data_recursion)
    # replace nan to 0
    success_rates = [x if np.isnan(x)==False else 0 for x in success_rates]
    # grid search for minimum distance estimation
    estimatorNewRecursion = functools.partial(estimatorRecursion, actual_ccp=actual_ccp, 
                            success_rates=success_rates, actual_W=actual_W)
    obj = [estimatorNewRecursion(item) for item in parameter_combos]
    # find parameters that gives the minimum distance
    search_grid_sol = list(parameter_combos)[np.argmin(obj)]
    return search_grid_sol

def multiBootstrap(args, nprocs):
    # based on "https://eli.thegreenplace.net/2012/01/16/
    # python-parallelizing-cpu-bound-tasks-with-multiprocessing/"
    def worker(args, out_q):
        outlist = []
        for arg in args:
            coefficients = estimationRecursionGrid(arg)
            outlist += [coefficients]
        out_q.put(outlist)
    out_q = Queue()
    procs = []
    chunksize = len(args)//nprocs
    for i in range(nprocs):
        chunk = args[i*chunksize:(i+1)*chunksize]
        p = Process(target=worker,
                args=(chunk, out_q))
        procs.append(p)
        p.start()
    # Collect all results into a single result list
    resultlist = []
    for i in range(nprocs):
        resultlist += [out_q.get()]
    # Wait for all worker processes to finish
    for p in procs:
        p.join()
    return resultlist

if __name__=="__main__":
    ######## RECURSION VERSION OF ESTIMATION
    data_recursion = pd.read_pickle('data_recursion.pkl')
    # estimate ccp and weights
    actual_ccp, actual_W = ccp_fun(data_recursion)
    # estimate success rates
    T = 10
    success_rates = np.zeros((T,)) 
    success_rates[0:T-1] = p_acpt_fun(data_recursion)
    # replace nan to 0
    success_rates = [x if np.isnan(x)==False else 0 for x in success_rates]
    theta0_vec = np.linspace(-1,0,7)
    theta1_vec = np.linspace(1,4,7)
    discount_vec = np.linspace(0.5,1,7)
    # grid search for minimum distance estimation
    start = time.time()
    estimatorNewRecursion = functools.partial(estimatorRecursion, actual_ccp=actual_ccp, 
                            success_rates=success_rates, actual_W=actual_W)
    parameter_combos = itertools.product(theta0_vec, theta1_vec, discount_vec)
    obj = [estimatorNewRecursion(item) for item in parameter_combos]
    end = time.time()
    # find parameters that gives the minimum distance
    parameter_combos = itertools.product(theta0_vec, theta1_vec, discount_vec)
    search_grid_sol = list(parameter_combos)[np.argmin(obj)]
    print("The solution from the search-grid algorithm is :{}.\n \
        It took a total of {} seconds to compute".format(search_grid_sol, end-start))
    # traditional optimization doesn't work well with stochastic objective function
    from scipy.optimize import minimize as smin
    print(smin(fun=estimatorNewRecursion, x0=(0, 0, 0.1), method="Powell"))
    
    ######## ITERATION VERSION OF ESTIMATION
    data_iteration = pd.read_pickle('data_iteration.pkl')
    actual_ccp, actual_W = ccp_fun(data_iteration)
    # estimate success rates
    T = 10
    success_rates = np.zeros((T,)) 
    success_rates[0:T-1] = p_acpt_fun(data_iteration)
    # replace nan to 0
    success_rates = [x if np.isnan(x)==False else 0 for x in success_rates]
    # grid search for minimum distance estimation
    theta0_vec = np.linspace(-1,0,15)
    theta1_vec = np.linspace(0,2,15)
    discount_vec = np.linspace(0.85,0.95,3)
    start = time.time()
    obj = list(map(functools.partial(estimatorIteration,actual_ccp,success_rates),
                   itertools.product(theta0_vec,theta1_vec,discount_vec)))
    search_grid_sol = list(itertools.product(theta0_vec,theta1_vec,discount_vec))[np.argmin(obj)]
    end = time.time()
    print("The solution from the search-grid algorithm is :{}.\n \
            It took a total of {} seconds to compute".format(search_grid_sol,np.round(end-start),2))
    ### Bayesian optimization
    from hyperopt import fmin, hp, tpe, Trials
    tpe_trials = Trials()
    tpe_algo = tpe.suggest
    space = [hp.normal('theta0', 0, 2), hp.normal('theta1', 0, 2), hp.uniform('discount', 0.1, 1)]
    estimatorNew = functools.partial(estimatorIteration,actual_ccp,success_rates)
    best = fmin(fn = estimatorNew, space = space, algo=tpe.suggest, max_evals = 1000)
    print(best)
    
    ######## ESTIMATE STANDARD ERRORS BY BOOTSTRAPPING
    n_bootstrap = 2**7
    ## with multiprocessing
    from multiprocessing import cpu_count, Process, Queue
    parameter_combos = tuple(list(itertools.product(theta0_vec, theta1_vec, discount_vec)))
    grid_size = len(parameter_combos)
    N_individuals = 1000
    args = [(getResample(data_recursion), parameter_combos) for i in range(n_bootstrap)]
    start = time.time()
    result = multiBootstrap(args, cpu_count())
    print(result)
    print("total time used for bootstrapping %d times of grid size %d is %.2f seconds" 
          %(n_bootstrap, grid_size, time.time()-start))
    # should be 32 items in bootstrap_coeffs
    bootstrap_coeffs = list(itertools.chain(*result))
    num_parameters = 3
    bootstrap_coeffs = [np.array([x[i] for x in bootstrap_coeffs]) for i in range(num_parameters)]
    bootstrap_means = tuple([np.mean(x) for x in bootstrap_coeffs])
    print(bootstrap_means)
    bootstrap_standard_errors = [standardErrors(x) for x in bootstrap_coeffs]
    print(bootstrap_standard_errors)