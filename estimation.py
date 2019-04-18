import numpy as np
import pandas as pd
import functools, itertools
import os, time, random
import matplotlib.pyplot as plt
import scipy.optimize as sop

# load data and lag the data to get future work experience
data = pd.read_pickle('simulation_search_data.pkl')
data['future_work_experience'] = data['work_experience'].shift(-1).values.astype(int)
T = 10
mask = data.age == T-1
data.loc[mask,'future_work_experience'] = 999

def ccp_fun_inefficient(data):

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

start = time.time()
data = pd.read_pickle('simulation_search_data.pkl')
actual_ccp,actual_W = ccp_fun_inefficient(data)
print("Computation time, very inefficient code: {}: ".format(time.time()-start))

def ccp_fun(data):    
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
    
 
start_time = time.time()
data = pd.read_pickle('simulation_search_data.pkl')
actual_ccp,actual_W = ccp_fun(data)
print("Computation time, less inefficient code: {}: ".format(time.time()-start_time))

# estimation transition probability/success rate
def p_acpt_fun(data):

    data['future_work_experience'] = data['work_experience'].shift(-1).values.astype(int)
    mask = data.age == T-1
    data.loc[mask,'future_work_experience'] = 999
    
    p_acpt = np.zeros((T-1,))
    for i in range(0,T-1):
        num = np.sum((data['age'] < T-1) & (data['work_experience'] == i) & (data['future_work_experience'] == i + 1) & (data['choice'] == 2))
        den = np.sum((data['age'] < T-1) & (data['work_experience'] == i) & (data['choice'] == 2)) 
        if den > 0:
            p_acpt[i] = num / den
        else:
            p_acpt[i] = np.nan

    return p_acpt

data = pd.read_pickle('simulation_search_data.pkl')
success_rates = p_acpt_fun(data)

plt.figure()
plt.plot(range(0,9),success_rates,'r',label='Empirical')
plt.plot(range(0,9),[0.5 + 0.5/(T-1) * item for item in range(0,9)],'b',label = 'True')
plt.xlabel(r'$x$')
plt.ylabel(r'$\lambda(x)$')
plt.legend()
plt.show()

# minimizing distance between predicted CCP and actual CCP

def predictCCP(success_rates, theta, discount):
    data = dataSimulation(successRates, theta, discount)
    ccp, W = ccp_fun(data)
    return ccp, W

def estimator(parameters):
    global actual_ccp, success_rate
    theta = parameters[0]
    discount = parameters[1]
    predicted_ccp, W = predictCCP(success_rates, theta, discount)
    distance = np.sum(np.multiply((predicted_ccp-actual_ccp)**2,W))
    return distance

theta_vec = np.linspace(1.1,2.9,20)
discount_vec = np.linspace(0.5,1,20)
obj = [estimator(item) for item in itertools.product(theta_vec,discount_vec)]




