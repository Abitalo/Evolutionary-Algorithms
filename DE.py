import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from matplotlib.animation import FuncAnimation

# Updated in 20 Aug. 2019, supports higher dimensions of feature space.

# This file implemented Differential Evolution Algorithm(DE) to search the global sub-optimal value for function obj_fn(x), 
# which you can replace it with other function if you like.
# For practice purpose, might be wrong and could be updated in any time.


# the objectve function made up by personal choice.
def obj_fn(x):   
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
        
    ret = ((x[:,0]-12)**2) + ((x[:,1]-8)**2) # triviously, global minimum is (12,8)
#     ret = np.sin(x-3)*np.sin(0.5*x-3)-np.log(x**2+1) # from the plot, global maximum near 0.2

    # negative symbol is needed in minimum search scenario.
    return -ret.squeeze() 


def mutate(P, f=0.9):
    M = np.empty_like(P)
    

    # randomly pick 2 index of individuals from subsets other than current idx i, for i in range(N)
    indice = [np.random.choice(np.concatenate([list(range(i)),list(range(i+1,N))]), 2, replace=False) for i in range(N)]
    indice = np.asarray(indice, dtype=np.int)

    # c = f*(a-b)
    M = f * (P[indice[:,0]] - P[indice[:,1]])

    return M


def cross(P, M, cr=0.5):
    # crossover subject to cross_rate cr.
    new_P = np.where(np.random.rand(N,D)<cr,M,P)
    
    return new_P


def select(P):
    # 修正边界
    P = P[np.all(abs(P)<50,axis=-1)]
    fitness = obj_fn(P)   

    #top K largest fitness
    new_P = P[np.argpartition(-fitness,N)][:N]

    return new_P   

if __name__ == '__main__':
    N = 100     # population size
    D = 2       # dimensions of feature space 
    F = 0.9     # mutation rate
    CR = 0.5    # cross rate
    max_G = 100 # max generation
    P = np.random.rand(N,D)*100-50 # uniform sampled from (-50,50)
    
    # for gif generating purpose.
    P_seq = []
    
    try:
        with tqdm(range(max_G)) as t:

            # iterate N generation
            for i in t:
                # record current population state.
                P_seq.append(P)
                
                M = mutate(P,F)
                new_P = cross(P,M,CR)
                P = select(np.concatenate([P,new_P]))

    except KeyboardInterrupt:
        t.close()
        raise
    finally:
        t.close()
        
    # record the final population state.
    P_seq.append(P)
    
    print(f'the sub-optimal found in {max_G} generations is: {P.mean(axis=0)}')
