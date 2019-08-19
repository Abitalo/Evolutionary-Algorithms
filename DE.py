import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
from tqdm import tqdm


# This file implemented Differential Evolution Algorithm(DE) to search the maximum value for function obj_fn(x), 
# which you can replace it with other function if you like.
# For practice purpose, might be wrong and could be updated in any time.


# the objectve function made up by personal choice.
def obj_fn(x):
    ret = np.sin(x-3)*np.sin(0.5*x-3)-np.log(x**2+1)
    return ret


# convert the maplotlib figure to RGB ndarray.
def get_img(X,Y,P):
    fig = plt.figure()
    plt.plot(X,Y)
    plt.scatter(P,obj_fn(P), color='red', marker='*')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return data


def mutate(P, f=0.9):
    # empty placeholder for mutated population, with same size to original(or previous) population.
    M = np.empty_like(P)

    # iterate individual c in previous population, mutate by difference from  2 randomly individuals picked from the rest population.
    for i, c in enumerate(P):

        # get the subset of population with current individuals removed.
        indice = [x for x in range(N) if x is not i]

        # randomly pick 2 individuals from the subset.
        a,b = np.random.choice(P[indice],2)

        # mutation.
        M[i] = c + F*(a-b)

    return M


def cross(P, M, cr=0.5):
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

    img_list = [] # only for image saving purpose.
    try:
        with tqdm(range(max_G)) as t:

            # iterate N generation
            for i in t:
                # plot current location of individuals over ojective function figure.
                img_list.append(get_img(X,Y,P))
                M = mutate(P)
                P = cross(P,M)           
                P = select(P)

    except KeyboardInterrupt:
        t.close()
        raise
    finally:
        t.close

    # generate animation gif.
    duration = 1
    fps = 20

    # callback function required by generate animation.
    def make_frame_mpl(t):
        t=int(t*duration*fps)
        return img_list[t] # RGB ndarray for image.

    animation =mpy.VideoClip(make_frame_mpl, duration=duration)
    animation.write_gif("de_iteration.gif", fps)
