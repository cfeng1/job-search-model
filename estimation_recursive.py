import numpy as np
import pandas as pd
import itertools, functools
import time
from data_simulation_recursion_version import dataSimulationRecursion
# from estimation import ccp_fun, predictCCP, p_acpt_fun

def ccpEstimation(data, T=10):    
    def ccpState(arg):
        age , exp =  arg
        mask_den = (data['age'] == age) & (data['work_experience'] == exp)
        mask_num = (mask_den) & (data['choice'] == 2) 
        W_state = np.sum(mask_den)
        ccp_state = np.sum(mask_num) / W_state if W_state>0 else 999
        return ccp_state,W_state
       
    output = [ccpState(item) for item in filter(lambda x: x[0]>=x[1], 
        itertools.product(range(0,T),range(0,T)))]
    ccp = np.array([item[0] for item in output])
    W = np.array([item[1] for item in output])
    W = W / np.sum(W) 
    return ccp , W

# estimation transition probability/success rate
def transitionProbability(data, T=10):
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

### recursion version
def predictCCPRecursion(success_rates, theta, discount):
    data = dataSimulationRecursion(theta, discount, success_rates)
    ccp, W = ccpEstimation(data) 
    return ccp, W 

def estimatorRecursion(parameters, actual_ccp, success_rates):
    theta = parameters[0]
    discount = parameters[1]
    predicted_ccp, W = predictCCPRecursion(success_rates, theta, discount)
    distance = np.sum(np.multiply((predicted_ccp-actual_ccp)**2,W))
    return distance

T = 10
# data = pd.read_pickle('data_simulation_search_iteration.pkl')
data = pd.read_pickle('simulation_search_recursion.pkl')
actual_ccp, actual_W = ccpEstimation(data)

success_rates = list(transitionProbability(data))
success_rates[-1] = 0
estimatorRecur = functools.partial(estimatorRecursion, 
    actual_ccp=actual_ccp, success_rates=success_rates)

theta_vec = np.linspace(1.1,2.9,20)
discount_vec = np.linspace(0.5,1.1,20)

start = time.time()
obj_recursive = [estimatorRecur(item) for item in itertools.product(theta_vec,discount_vec)]
end = time.time()
search_grid_sol_recursive = list(itertools.product(
    theta_vec,discount_vec))[np.argmin(obj_recursive)]
print("The solution from the search-grid algorithm is :{}.\n \
    It took a total of {} seconds to compute".format(search_grid_sol_recursive, end-start))
