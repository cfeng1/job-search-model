# job search simulation functions
import numpy as np
import pandas as pd
import functools, itertools
import os, time, random

def expectedMax(*args, mu=0, beta=0.15, size=2000):
    # compute Emax[val1+epsilon1, val2+epsilon2]
    np.random.seed(100)
    total_values = [args[i]+np.random.gumbel(mu, beta, size) \
    for i in range(len(args))]
    return np.average([max(x) for x in zip(*total_values)])

## success rate of job application
success = lambda work_experience, T=10: (work_experience/(T-1))*0.2+0.8

# decorator function: function that takes another function as argument
def memoize(function):
    memo = {}
    def helper(x):
        if x not in memo:            
            memo[x] = function(x)
        return memo[x]
    return helper

def saveData(res, T, N):
    individuals = list(itertools.chain(*[[i]*T for i in range(N)]))
    time_periods = list(itertools.chain(*[list(range(T)) for i in range(N)]))
    work_experiences = list(itertools.chain(*[item[0] for item in res]))
    choices = list(itertools.chain(*[item[1] for item in res]))
    data = pd.DataFrame({'individual': individuals, 'age': time_periods, 
                         'work_experience': work_experiences, 'choice': choices})
    return data

# simulate data given the parameter
def dataSimulationRecursion(theta, discount, successRates, N=1000, T=10):
    # print(successRates)
    # utilityWork = [-np.exp(-theta*x)+0.5 for x in range(0,T)]
    # continuation_values = np.zeros((T,T))
    theta0, theta1 = theta
    utilityWork = [theta0+theta1*x/T for x in range(0,T)]
    @memoize
    def continuationValue(arg_tuple):
        nonlocal discount, successRates
        # nonlocal continuation_values
        t, T, work_experience, current_choice = arg_tuple
        work_experience = int(work_experience)
        if t>=T-1:
            # continuation_values[t][work_experience] = 0
            return 0
        else:
            success_rate = successRates[work_experience]
            state_tuple_home = (t+1, T, work_experience, 1)
            value_home = continuationValue(state_tuple_home)
            state_tuple_work = (t+1, T, work_experience, 2)
            value_work = (utilityWork[work_experience]+
                             continuationValue(state_tuple_work))
            if current_choice==1:
                # now home -> state variable next period stays the same
                continuation_value = discount*expectedMax(value_home, value_work)
            else:
                # now work -> state variable next period may change
                # if job application succeeds
                state_tuple_home_success = (t+1, T, work_experience+1, 1)
                value_home_success = continuationValue(state_tuple_home_success)
                state_tuple_work_success = (t+1, T, work_experience+1, 2)
                value_work_success = (utilityWork[work_experience+1]+
                             continuationValue(state_tuple_work_success))
                # total continuation value
                continuation_value = discount*(
                    success_rate*expectedMax(value_home_success, value_work_success)+
                    (1-success_rate)*expectedMax(value_home, value_work))
            # continuation_values[t][work_experience] = continuation_value
            return continuation_value
        
    def generateChoices(T, successRates, discount, mu=0, beta=1):
        # default mu and beta -> type I extreme value
        work_experience = 0
        work_experiences = [work_experience]
        choices = []
        actual_shock_home = np.random.gumbel(mu, beta, T)
        actual_shock_work = np.random.gumbel(mu, beta, T)
        t = 0
        while t<=T-1:
            success_rate = successRates[work_experience]
            job_search = np.random.binomial(n=1, p=success_rate)
            state_tuple_home = (t, T, work_experience, 1)
            state_tuple_work = (t, T, work_experience, 2)
            value_home = actual_shock_home[t]+continuationValue(state_tuple_home)
            value_work = (actual_shock_work[t]+success_rate*utilityWork[work_experience]+
                          continuationValue(state_tuple_work))
            choices += [1+(value_home<=value_work)]
            if t<T-1:
                work_experience += int(job_search*(value_home<=value_work))
                work_experiences += [work_experience]
            t += 1
        return work_experiences, choices
    res = [generateChoices(T, successRates, discount) for i in range(N)]
    data = saveData(res, T, N)
    # print(continuation_values)
    return data

if __name__=="__main__":
    start = time.time()
    successRates = [success(x) for x in range(10)]
    data = dataSimulationRecursion((-0.3,2), 0.9, successRates)
    data.to_pickle('simulation_search_recursion.pkl')
    print(time.time()-start)
    # print(data.head(10))