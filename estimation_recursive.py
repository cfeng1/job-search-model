import numpy as np
import pandas as pd
import itertools, functools
import time
from data_simulation_recursion_version import dataSimulationRecursion
from estimation import ccp_fun, predictCCP, p_acpt_fun

### recursion version
def predictCCPRecursion(success_rates, theta, discount):
    data = dataSimulationRecursion(theta, discount, success_rates)
    ccp, W = ccp_fun(data) 
    return ccp, W 

def estimatorRecursion(parameters, actual_ccp, success_rates):
    theta = parameters[0]
    discount = parameters[1]
    predicted_ccp, W = predictCCP(success_rates, theta, discount)
    distance = np.sum(np.multiply((predicted_ccp-actual_ccp)**2,W))
    return distance

T = 10
data = pd.read_pickle('data_simulation_search_iteration.pkl')
actual_ccp, actual_W = ccp_fun(data)

success_rates = np.zeros((T,)) 
success_rates[0:T-1] = p_acpt_fun(data)

estimatorRecur = functools.partial(estimatorRecursion, 
    actual_ccp=actual_ccp, success_rates=success_rates)

theta_vec = np.linspace(1.1,2.9,20)
discount_vec = np.linspace(0.5,1,20)

start = time.time()
obj_recursive = [estimatorRecur(item) for item in itertools.product(theta_vec,discount_vec)]
end = time.time()
search_grid_sol_recursive = list(itertools.product(
    theta_vec,discount_vec))[np.argmin(obj_recursive)]
print("The solution from the search-grid algorithm is :{}.\n \
    It took a total of {} seconds to compute".format(search_grid_sol_recursive, end-start))
