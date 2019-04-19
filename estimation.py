import numpy as np
import pandas as pd
import itertools
import time
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
from data_simulation_iteration_version import dataSimulationIteration
from data_simulation_recursion_version import dataSimulationRecursion

def ccp_fun_inefficient(data, T=10):
    ccp = np.zeros((T,T))
    W = np.zeros((T,T))
    for age in range(0,T):
        for exp in range(0,T):
            num = len(data[(data['age'] == age) & (data['work_experience'] == exp) & (data['choice'] == 2) ])
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
        W_state = np.sum(mask_den)
        ccp_state = np.sum(mask_num) / W_state if W_state>0 else 999
        return ccp_state,W_state
       
    output = [ccp_state_fun(item) for item in filter(lambda x: x[0]>=x[1], itertools.product(range(0,T),range(0,T)))]
    ccp = np.array([item[0] for item in output])
    W = np.array([item[1] for item in output])
    W = W / np.sum(W) 
    return ccp , W


# estimation transition probability/success rate
def p_acpt_fun(data, T=10):
    data['future_work_experience'] = data['work_experience'].shift(-1).values.astype(int)
    mask = data.age == T-1
    data.loc[mask,'future_work_experience'] = 999
    
    p_acpt = np.zeros((T-1,))
    for i in range(0,T-1):
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
    data = dataSimulationIteration(success_rates, theta, discount)
    # data = dataSimulationRecursion(theta, discount, success_rates)
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
    np.product(successRates)
    
    # start = time.time()    
    data = dataSimulationIteration(successRates, theta=2, discount=.9)
    
    #print(np.round(continuation_value,2))
    # data.to_pickle('data_simulation_search_iteration.pkl')
    # end = time.time()
    # print("It takes a total of {} seconds to simulate a \
    #     dataset with 1000 individuals living 10 periods".format(end-start))
    # print("\n")
    # print(data.head())

    # load data and lag the data to get future work experience
    # data = pd.read_pickle('simulation_search_iteration.pkl')
    # data = pd.read_pickle('simulation_search_recursion.pkl')
    data['future_work_experience'] = data['work_experience'].shift(-1).values.astype(int)
    T = 10
    mask = data.age == T-1
    data.loc[mask,'future_work_experience'] = 999

    start = time.time()
    actual_ccp,actual_W = ccp_fun_inefficient(data)
    print("Computation time, very inefficient code: {}: ".format(time.time()-start))

    start_time = time.time()
    actual_ccp, actual_W = ccp_fun(data)
    print("Computation time, less inefficient code: {}: ".format(time.time()-start_time))

    success_rates = np.zeros((T,)) 
    success_rates[0:T-1] = p_acpt_fun(data)
    # replace nan to 0
    success_rates = [x if np.isnan(x)==False else 0 for x in success_rates]

    plt.figure()
    plt.plot(range(0,T-1),success_rates[0:T-1],'r',label='Empirical')
    plt.plot(range(0,T-1),[0.8 + 0.2/(T-1) * item for item in range(0,T-1)],'b',label = 'True')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\lambda(x)$')
    plt.legend()
    plt.show()

    # estimation procedure
    theta_vec = np.linspace(1.01,4.01,100)
    # discount_vec = np.linspace(0.5,1,20)
    print("data iteration, simulation iteration")
    start = time.time()
    # obj = [estimator(item) for item in itertools.product(theta_vec,discount_vec)]
    obj = [estimator(item) for item in [(i, 0.9) for i in theta_vec]]
    end = time.time()
    search_grid_sol = [(i, 0.9) for i in theta_vec][np.argmin(obj)]
    print("The solution from the search-grid algorithm is :{}.\n It took a total of {} seconds to compute".format(search_grid_sol,end-start))
    plt.plot(theta_vec,obj)



