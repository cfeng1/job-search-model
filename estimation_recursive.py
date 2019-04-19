import numpy as np
import pandas as pd
import itertools
import time
import matplotlib.pyplot as plt
# from data_simulation_iteration_version import dataSimulationIteration
from data_simulation_recursion_version import dataSimulationRecursion


def ccp_fun(data, T=10):    
    def ccp_state_fun(arg):
        age , exp =  arg
        mask_den = (data['age'] == age) & (data['work_experience'] == exp)
        mask_num = (mask_den) & (data['choice'] == 2) 
        # W_state = np.sum(mask_den)
        W_state = len(data[mask_den])
        # ccp_state = np.sum(mask_num) / W_state if W_state>0 else 999
        ccp_state = len(data[mask_num])/W_state if W_state>0 else 999
        return ccp_state, W_state
    output = [ccp_state_fun(item) for item in filter(lambda x: x[0]>=x[1], 
        itertools.product(range(T),range(T)))]
    ccp = np.array([item[0] for item in output])
    W = np.array([item[1] for item in output])
    W = W / np.sum(W) 
    return ccp, W


# estimation transition probability/success rate
def p_acpt_fun(data, T=10):
    data['future_work_experience'] = data['work_experience'].shift(-1).values.astype(int)
    mask = data.age == T-1
    data.loc[mask,'future_work_experience'] = 999
    
    p_acpt = np.zeros((T-1,))
    for i in range(T-1):
        num = np.sum((data['age'] < T-1) & (data['work_experience'] == i) & 
            (data['future_work_experience'] == i + 1) & (data['choice'] == 2))
        den = np.sum((data['age'] < T-1) & (data['work_experience'] == i) & 
            (data['choice'] == 2)) 
        if den > 0:
            p_acpt[i] = num / den
        else:
            p_acpt[i] = np.nan

    return p_acpt


# minimizing distance between predicted CCP and actual CCP
def predictCCP(success_rates, theta, discount):
    # data = dataSimulationIteration(success_rates, theta, discount)
    data = dataSimulationRecursion(theta, discount, success_rates)
    ccp, W = ccp_fun(data)
    return ccp, W

def estimator(parameters):
    global actual_ccp, success_rates, actual_W
    theta = parameters[0]
    discount = parameters[1]
    predicted_ccp, W = predictCCP(success_rates, theta, discount)
    distance = np.sum(np.multiply((predicted_ccp-actual_ccp)**2,W))
    return distance


if __name__=="__main__":
    success = lambda work_experience, T=10: (work_experience/(T-1))*0.2+0.8
    successRates = [success(x) for x in range(10)]
      
    data = dataSimulationRecursion(theta=(-0.3,2), discount=0.9, successRates=successRates)

    # load data and lag the data to get future work experience
    # data = pd.read_pickle('simulation_search_recursion.pkl')
    data['future_work_experience'] = data['work_experience'].shift(-1).values.astype(int)
    T = 10
    mask = data.age == T-1
    data.loc[mask,'future_work_experience'] = 999

    start_time = time.time()
    actual_ccp, actual_W = ccp_fun(data)
    print("Computation time, less inefficient code: {}: ".format(time.time()-start_time))
    print(actual_ccp)

    success_rates = np.zeros((T,)) 
    success_rates[0:T-1] = p_acpt_fun(data)
    # replace nan to 0
    success_rates = [x if np.isnan(x)==False else 0 for x in success_rates]

    print('success rate is: ')
    print(success_rates)

    # estimation procedure
    theta0_vec = np.linspace(-1,0,7)
    theta1_vec = np.linspace(1,4,7)
    discount_vec = np.linspace(0.5,1,7)
    print("data iteration, simulation iteration")
    start = time.time()
    parameter_combos = itertools.product(itertools.product(theta0_vec,theta1_vec), discount_vec)
    obj = [estimator(item) for item in parameter_combos]
    end = time.time()
    parameter_combos = itertools.product(itertools.product(theta0_vec,theta1_vec), discount_vec)
    search_grid_sol = list(parameter_combos)[np.argmin(obj)]
    print("The solution from the search-grid algorithm is :{}.\n \
        It took a total of {} seconds to compute".format(search_grid_sol, end-start))
    #plt.plot(theta1_vec,obj)


