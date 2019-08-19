import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation


# This file implemented Differential Evolution Algorithm(DE) to search the maximum value for function obj_fn(x), 
# which you can replace it with other function if you like.
# For practice purpose, might be wrong and could be updated in any time.


# the objectve function made up by personal choice.
def obj_fn(x):
    if not np.all(x):
        return None
    
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
        
    ret = np.sin(x-3)*np.sin(0.5*x-3)-np.log(x**2+1)
    return ret



def mutate(P, f=0.9):
    # empty placeholder for mutated population, with same size to original(or previous) population.
    M = np.empty_like(P)

    # iterate individual c in previous population, mutate by difference from  2 randomly individuals picked from the rest population.
    for i, c in enumerate(P):
        
        # randomly pick 2 individuals from the subset in which the current individual was excluded.
        a,b = np.random.choice(np.concatenate([P[:i],P[i+1:]]),2)

        # mutation.
        M[i] = c + F*(a-b)

    return M


def cross(P, M, cr=0.5):
    # crossover for 1-dimensional individuals given that scenario.
    new_P = np.asarray([M[i] if np.random.rand()<cr else P[i] for i in range(len(P))])
    
    return new_P


def select(P):
    # remove the illegal individuals.
    P = P[abs(P)<50]
    fitness = obj_fn(P)
    # keept those individuals with top K fitness.
    new_P = P[np.argpartition(-fitness,N)][:N]

    return new_P   

if __name__ == '__main__':
    # plot the objective function.
    X = list(np.linspace(-50,50,1001))
    Y = []
    for x in np.linspace(-50,50,1001):
        Y.append(obj_fn(x))
    plt.plot(X,Y)

    # basic hyper parameter
    N = 100 # population size
    F = 0.9 # scale factor
    CR = 0.5 # cross rate
    max_G = 20 # max generation

    # initial population randomly sampled from (-50,50).
    P = np.random.rand(N)*100-50
    
    # for gif generating purpose.
    P_seq = [[]] 
    
    try:
        with tqdm(range(max_G)) as t:

            # iterate N generation
            for i in t:
                # record current population state.
                P_seq.append(P)
                
                M = mutate(P)
                new_P = cross(P,M)  
                P = select(np.concatenate([P,new_P], axis=0))

    except KeyboardInterrupt:
        t.close()
        raise
    finally:
        t.close
        
    # record the final population state.
    P_seq.append(P)
    
    
    # generate animation gif.
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    line = ax.plot(X,Y)
    sc = ax.scatter(None,None,c='red')
    def update(t):
        ax.set_xlabel(f'iteration {t}')
        sc.set_offsets(np.c_[P_seq[t],obj_fn(P_seq[t])])
    
    anim = FuncAnimation(fig, update, frames=np.arange(0, 21), interval=200)
    anim.save('de_convergence.gif', dpi=80, writer='imagemagick')
