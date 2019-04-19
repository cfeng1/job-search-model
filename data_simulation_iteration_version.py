import numpy as np
import pandas as pd
import time


def dataSimulationIteration(success_rates, theta, discount, N=1000, T=10):
    N_sim = 2000
    utilityWork = [-.3+theta*x/T for x in range(0,T)]
    utilityHome = [0]*T
    # no need to allocate space to store
    continuation_value = np.zeros((T+1,T+1))
    epsilon_work = np.random.gumbel(0,.15,size = N_sim)
    epsilon_home = np.random.gumbel(0,.15 ,size = N_sim)  

    for age in range(T-1, -1, -1): 
        for exp in range(age, -1, -1):              
            success_rate = success_rates[exp]

            value_hw = np.zeros((N_sim,2))
            value_hw[:,0] = (utilityHome[exp] + epsilon_home + 
                             discount*continuation_value[age+1,exp])
            value_hw[:,1] = epsilon_work + success_rate*(
                utilityWork[exp] + discount*continuation_value[age+1,exp+1]) + (
                1-success_rate)*(utilityHome[exp] + discount*continuation_value[age+1,exp])

            continuation_value[age,exp] = np.mean(np.max(value_hw,axis=1))
    
    def individualSimulation(i):  
        nonlocal T, success_rates, continuation_value

        epsilon_work_i = np.random.gumbel(0,1,size = T)
        epsilon_home_i = np.random.gumbel(0,1,size = T)
        success_shock_sim = np.random.random(size=T)   
        
        exp_i = np.zeros((T,),dtype = int)
        choice_i = np.zeros((T,),dtype = int)
        
        for age in range(T):            
            success_rate = success_rates[exp_i[age]]
            value_home = (utilityHome[exp_i[age]] + epsilon_home_i[age] + 
                          discount*continuation_value[age+1,exp_i[age]])
            value_work = (epsilon_work_i[age] + success_rate*(utilityWork[exp_i[age]] + 
                          discount*continuation_value[age+1,exp_i[age]+1]) + 
                          (1-success_rate)*(utilityHome[exp_i[age]] + 
                        discount*continuation_value[age+1,exp_i[age]]))
            choice_i[age] = 1 + int(value_home <= value_work)
            
            if (age < T-1):
                    exp_i[age+1] = exp_i[age] + (choice_i[age] == 2) *(
                        success_shock_sim[age] <= success_rate) 
            
        matrix_sim_i = np.zeros((T,4),dtype = int)
        matrix_sim_i[:,0] = i*np.ones((T,))
        matrix_sim_i[:,1] = choice_i
        matrix_sim_i[:,2] = exp_i
        matrix_sim_i[:,3] = range(0,T)                    
        return matrix_sim_i
    
    matrix_sim = np.zeros((N*T,4))           
    for i in range(0,N):
        
        matrix_sim[i*T:(i+1)*T,:] = individualSimulation(i)
        
    df_sim = pd.DataFrame(matrix_sim, 
        columns=["individual", "choice", "work_experience", "age"],dtype = int)
            
    return df_sim

if __name__=="__main__":
    success = lambda work_experience, T=10: (work_experience/(T-1))*0.2+0.8
    start = time.time()
    success_rates = [success(x) for x in range(10)]
    data = dataSimulationIteration(success_rates, 2, 0.9)
    data.to_pickle('simulation_search_iteration.pkl')
    print(time.time()-start)
    print(data.head())
