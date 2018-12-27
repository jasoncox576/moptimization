import matplotlib.pyplot as plt
import random
import os.path
from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy as np
import math

from sympy import symbols, diff



"""
Optimization techniques to try:

    Stochastic Gradient Descent

    Mini-batching

    SVRG

    Momentum

    Alternating Minimization Methods

    Branch-and-bound methods








"""





def banana(x, y):

    """
    Rosenbrock banana function
    classic benchmark for testing 
    optimization algorithms
    """

    #x, y should be a single pair.


    a = 1
    b = 100

    return ((a-x)**2) + (b*(y-(x**2)))**2



def nesterov_momentum(x_list, y_list, z_list, batches, epochs, learning_rate, alpha=0.0):

    """
    Nesterov momentum:

    Starts off at a certain random point.

    Makes a random initial jump

    Takes the gradient at the point it jumped to and corrects

    Take a big jump in the direction of previous gradient

    """

    


    

def sgd(x_list, y_list, z_list, batches, epochs, learning_rate, momentum=False, alpha=0.0):

    """ This is just standard sgd implementation
        
        If momentum set to True, will execute the momentum variant of sgd
        the alpha parameter must be set as well to something other than zero.
    
    """
    
    #initialize a member to store value of current update
    # only used for momentum
    prev_update_x = 0.0
    prev_update_y = 0.0




    # pick a random x, y
    
    print(np.shape(x_list))

    # initialize a random x and y and attempt to optimize
    descending_x = x_list[random.randint(0, len(x_list)-1)] 
    descending_y = y_list[random.randint(0, len(y_list)-1)]


    #descending_x = 0
    #descending_y = 0

    x_stops = []
    y_stops = []

    x_stops.append(descending_x)
    y_stops.append(descending_y)

    print("STARTING LOCATIONS: X: " + str(descending_x) + " Y: " + str(descending_y))

    for epoch in range(10): 
        for iterations in range(500):
            rand_x = random.randint(0, len(x_list)-1)
            rand_y = random.randint(0, len(y_list)-1)

            #Compute partial with respect to x
            # first, fix the 'y' and grab that array:

            x_slice = z_list[rand_y]
            partial_dx = np.gradient(x_slice) 
            #now that we have a 'gradient' (sort of) for each element in this slice we 
            # can pick out the one corresponding to our fixed x or y coord
            x_grad = partial_dx[rand_x] 

        
            y_slice = np.transpose(z_list)[rand_x]
            partial_dy = np.gradient(y_slice)
            y_grad = partial_dy[rand_y]
            

            T_x = learning_rate * x_grad
            T_y = learning_rate * y_grad

            descending_x -= T_x 
            print("T_x" + str(T_x))
            print("Descending x: " + str(descending_x))
            descending_y -= T_y


            if momentum:
                descending_x += (alpha) * (prev_update_x)
                print("Prev_update_x: " + str(prev_update_x))
                descending_y += (alpha) * (prev_update_y)

                prev_update_x = (alpha) * (prev_update_x) - T_x
                prev_update_y = (alpha) * (prev_update_y) - T_y

            #print(descending_x, descending_y)
                
            x_stops.append(descending_x)
            y_stops.append(descending_y)

            #print("X-grad: ", str(x_grad), "Y-grad: ", str(y_grad), "rand_x: ", str(x_list[rand_x]), "rand_y", str(y_list[rand_y]))
        
        #You NEED the below line
        learning_rate =  learning_rate / 10
        
    
    
    plt.plot(x_stops)
    plt.plot(y_stops)
    

    return (descending_x, descending_y)



def plot_3d(X, Y, Z, minimum_coords):
    

    x_min, y_min = minimum_coords

    fig = plt.figure()
    ax = Axes3D(fig)


    print("=========================================================")


    ax.plot_surface(Y, X, Z, rstride=1, cstride=1, cmap="terrain")

    text = "Global Minimum: " + str(x_min) + " : " + str(y_min) + " : " + str(banana(x_min, y_min))

    ax.text(x_min, y_min, banana(x_min, y_min), text, size=20) 
    plt.show()



def __main__():

    """
    if os.path.isfile('banana.obj'):
        fhandle = open('banana.obj', "rb")
        pickle.load( 
    """

    x_list = [x for x in np.arange(-2, 2.01, 0.0005)]
    
    y_list = [x for x in np.arange(-2, 2.01, 0.0005)]


    #Note: when plotting function it may be best to just use
    #a subset of the total samples so that it may render.

    Y, X = np.meshgrid(y_list, x_list)

    z_list = np.empty([len(x_list), len(y_list)])
    for y_index in range(len(y_list)):
        for x_index in range(len(x_list)):
            z = banana(x_list[x_index], y_list[y_index])
            z_list[y_index][x_index] = z
    
     
    print("Finshed computing Rosenbrock Banana")

    minimum_coords_1 = sgd(x_list, y_list, z_list, 10, 1000, 0.0001)
    minimum_coords_2 = sgd(x_list, y_list, z_list, 100, 100, 0.0001, True, 0.5)
     
    

    
    offset = 200
    
    
    
    plot_x_list = [x_list[x] for x in np.arange(0, len(x_list), offset)]
    plot_y_list = [y_list[y] for y in np.arange(0, len(y_list), offset)]

    #plot_x_list = np.rint(plot_x_list)
    #plot_y_list = np.rint(plot_y_list)



    plot_z_list = np.empty([len(plot_y_list), len(plot_x_list)])


    #Fill up plot-z list
    for y in range(len(plot_y_list)):
        for x in range(len(plot_x_list)):
            plot_z_list[y][x] = z_list[y*offset][x*offset]
         
    plot_z_list = np.rint(plot_z_list)

    
    Y, X = np.meshgrid(plot_y_list, plot_x_list) 
    print("Meshed grid...") 
    
    print("Regular sgd: " + str(banana(minimum_coords_1[0], minimum_coords_1[1])))
    print("sgd with momentum: " + str(banana(minimum_coords_2[0], minimum_coords_2[1])))

    plot_3d(X, Y, plot_z_list, minimum_coords_1)
    plot_3d(X, Y, plot_z_list, minimum_coords_2)







__main__()
