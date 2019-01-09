import os.path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle
import numpy as np
import math

from functions import *

#from sympy import symbols, diff


# Random seed for starting location
# of each descent technique
RAND_SEED = random.randint(1, 1000)

#set testing landscape
FUNC = sphere_func
DOMAIN_MAX = 4
DOMAIN_DX = 0.001

"""
Optimization techniques to try:

    Stochastic Gradient Descent

    Mini-batching

    SVRG

    Momentum

    Alternating Minimization Methods

    Branch-and-bound methods








"""





def nesterov_momentum(x_list, y_list, z_list, batches, epochs, learning_rate, alpha=0.0):

    """
    Nesterov momentum:

    Starts off at a certain random point.

    Makes a random initial jump

    Takes the gradient at the point it jumped to and corrects

    Take a big jump in the direction of previous gradient

    """



def normalize(x, minimum, maximum):
    return (x - minimum) / (maximum - minimum)



def newton(x_list, y_list, z_list, batches, learning_rate):

    """
    Note: This is the bona-fide newton's method that
    computes the actual hessian matrix.
    I may later implement a quasi-algorithm and compare speed.
	However the speed difference is probably irrelevant here
	since we have a two variable function.

    TODO:
        For some reason, if you have too many iterations,
        the y-coordinate seems to get way too high.
        This is probably/possibly an issue with this implementation..

    """


    random.seed(RAND_SEED)
    descending_x = x_list[random.randint(0, len(x_list)-1)] 
    random.seed(RAND_SEED)
    descending_y = y_list[random.randint(0, len(y_list)-1)]
	
    x_stops = [descending_x]
    y_stops = [descending_y] 

    print("Newton STARTING LOCATIONS: X: " + str(descending_x) + " Y: " + str(descending_y))

    #vec_gradient = np.vectorize(np.gradient)

    z_list_t = np.transpose(z_list)

    x_grads = np.array([np.gradient(y) for y in z_list])
    y_grads = np.array([np.gradient(x) for x in z_list_t])

    x_grads_t = np.transpose(x_grads)

    for iteration in range(batches):

        try:
            index_prev_x = int(normalize(descending_x, -DOMAIN_MAX, DOMAIN_MAX) * len(z_list)) 
            print(index_prev_x)
            index_prev_y = int(normalize(descending_y, -DOMAIN_MAX, DOMAIN_MAX) * len(z_list))
            print(index_prev_y)

            #for the hessian we compute the second order with respect to
            #x, y, xy, yx
            
            #x_grads = np.array([np.gradient(y) for y in z_list])
            #x_grads = np.array(map(np.gradient, z_list))
            #x_grads = vec_gradient(z_list)
            x1_grad = x_grads[index_prev_y][index_prev_x]
            x2_grad = np.gradient(x_grads[index_prev_y])[index_prev_x]

            z_list_t = np.transpose(z_list)
            y1_grad = y_grads[index_prev_x][index_prev_y]
            print("y1_grad: " + str(y1_grad))
            y2_grad = np.gradient(y_grads[index_prev_x])[index_prev_y]
            
            xy_grad = np.gradient(x_grads_t[index_prev_x])[index_prev_y]
            #yx_grad is gonna be the same as xy_grad due to clairaut's theorem
            # so we won't make a new variable for it.

            hessian = np.matrix([[x2_grad, xy_grad], [xy_grad, y2_grad]])
            hessian_inv = np.linalg.inv(hessian)
            
            hessian_det = np.linalg.det(hessian_inv)

            descending_x -= (learning_rate * hessian_det * x1_grad) 
            descending_y -= (learning_rate * hessian_det * y1_grad)
            
            x_stops.append(descending_x)
            y_stops.append(descending_y)

            print("DESCENDING X: " + str(descending_x))
            print("DESCENDING Y: " + str(descending_y))
        
        except IndexError:
            print("ERROR: OUT OF BOUNDS")
            return

    plt.plot(x_stops, label='x_newton')
    plt.plot(y_stops, label='y_newton')
    plt.legend(loc='upper right') 

    return (descending_x, descending_y)


def gradient_descent(x_list, y_list, z_list, batches, epochs, learning_rate):

    random.seed(RAND_SEED)
    descending_x = x_list[random.randint(0, len(x_list)-1)] 
    random.seed(RAND_SEED)
    descending_y = y_list[random.randint(0, len(y_list)-1)]

    x_stops = [descending_x]
    y_stops = [descending_y] 

    print("STARTING LOCATIONS: X: " + str(descending_x) + " Y: " + str(descending_y))

    for epoch in range(epochs): 
        for iterations in range(batches):
               
            x_index = int(normalize(descending_x, -DOMAIN_MAX, DOMAIN_MAX) * len(z_list)) 
            print(descending_x)
            y_index = int(normalize(descending_y, -DOMAIN_MAX, DOMAIN_MAX) * len(z_list)) 


            #Compute partial with respect to x
            # first, fix the 'y' and grab that array:

            x_slice = z_list[x_index]
            partial_dx = np.gradient(x_slice) 
            #now that we have a 'gradient' (sort of) for each element in this slice we 
            # can pick out the one corresponding to our fixed x or y coord
            x_grad = partial_dx[x_index] 

        
            y_slice = np.transpose(z_list)[y_index]
            partial_dy = np.gradient(y_slice)
            y_grad = partial_dy[y_index]
            

            T_x = learning_rate * x_grad
            T_y = learning_rate * y_grad

            descending_x -= T_x 

            print("T_x" + str(T_x))
            print("Descending x: " + str(descending_x))
            #print("Descending x: {}".format(descending_x))
            descending_y -= T_y

                
            x_stops.append(descending_x)
            y_stops.append(descending_y)

            #print("X-grad: ", str(x_grad), "Y-grad: ", str(y_grad), "rand_x: ", str(x_list[rand_x]), "rand_y", str(y_list[rand_y]))
        
        #You NEED the below line
        #learning_rate =  learning_rate / 10
        
    x_label = 'x_standard'
    y_label = 'y_standard'
    
    plt.plot(x_stops, label=x_label)
    plt.plot(y_stops, label=y_label)
    plt.legend(loc='upper right') 

    return (descending_x, descending_y)


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
    
    random.seed(RAND_SEED)
    descending_x = x_list[random.randint(0, len(x_list)-1)] 
    random.seed(RAND_SEED)
    descending_y = y_list[random.randint(0, len(y_list)-1)]


    #descending_x = 0
    #descending_y = 0

    x_stops = [descending_x]
    y_stops = [descending_y] 

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
            #print("Descending x: {}".format(descending_x))
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
        

    if momentum: 
        x_label = 'x_momentum'
        y_label = 'y_momentum'
    else:
        x_label = 'x_standard'
        y_label = 'y_standard'
    
    plt.plot(x_stops, label=x_label)
    plt.plot(y_stops, label=y_label)
    plt.legend(loc='upper right') 

    return (descending_x, descending_y)



def plot_3d(X, Y, Z, minimum_coords):
    

    x_min, y_min = minimum_coords

    fig = plt.figure()
    ax = Axes3D(fig)


    print("=========================================================")


    ax.plot_surface(Y, X, Z, rstride=1, cstride=1, cmap="terrain", antialiased =True)

    #text = "Global Minimum: " + str(x_min) + " : " + str(y_min) + " : " + str(banana(x_min, y_min))
    text = "Global Minimum: " + str(x_min) + " : " + str(y_min) + " : " + str(FUNC(x_min, y_min))

    #ax.text(x_min, y_min, banana(x_min, y_min), text, size=20) 
    ax.text(x_min, y_min, FUNC(x_min, y_min), text, size=20) 
    plt.show()



def __main__():

    """
    if os.path.isfile('banana.obj'):
        fhandle = open('banana.obj', "rb")
        pickle.load( 
    """

    #x_list = [x for x in np.arange(-2, 2.0005, 0.0005)]
    x_list = [x for x in np.arange(-DOMAIN_MAX, DOMAIN_MAX+0.005, DOMAIN_DX)]
    
    #y_list = [x for x in np.arange(-2, 2.0005, 0.0005)]
    y_list = [y for y in np.arange(-DOMAIN_MAX, DOMAIN_MAX+0.005, DOMAIN_DX)]

    #Note: when plotting function it may be best to just use
    #a subset of the total samples so that it may render.

    Y, X = np.meshgrid(y_list, x_list)

    z_list = np.empty([len(x_list), len(y_list)])
    for y_index in range(len(y_list)):
        for x_index in range(len(x_list)):
            #z = banana(x_list[x_index], y_list[y_index])
            z = FUNC(x_list[x_index], y_list[y_index])
            z_list[y_index][x_index] = z
    
     
    print("Finshed computing Rosenbrock Banana")


    #minimum_coords_1 = sgd(x_list, y_list, z_list, 10, 1000, 0.0001)
    #minimum_coords_2 = sgd(x_list, y_list, z_list, 100, 100, 0.0001, True, 0.5)
    #minimum_coords_3 = newton(x_list, y_list, z_list, 10, 0.000001) 
    
    #minimum_coords_1 = sgd(x_list, y_list, z_list, 10000, 100000000000000000000000000000000000000000, 0.1)
    #minimum_coords_2 = sgd(x_list, y_list, z_list, 100, 500, 1, True, 0.5)

    #minimum_coords_1 = gradient_descent(x_list, y_list, z_list, 100, 1000, 0.1)
    minimum_coords_1 = newton(x_list, y_list, z_list, 10, 0.000000000001)

    
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
    
    #print("Regular sgd: " + str(banana(minimum_coords_1[0], minimum_coords_1[1])))
    #print("sgd with momentum: " + str(banana(minimum_coords_2[0], minimum_coords_2[1])))
    #print("Newton's method: " + str(banana(minimum_coords_3[0], minimum_coords_3[1])))

    print("Regular sgd: " + str(FUNC(minimum_coords_1[0], minimum_coords_1[1])))
    #print("sgd with momentum: " + str(FUNC(minimum_coords_2[0], minimum_coords_2[1])))
    #print("Newton's method: " + str(himmelblau(minimum_coords_3[0], minimum_coords_3[1])))

    plot_3d(X, Y, plot_z_list, minimum_coords_1)
    #plot_3d(X, Y, plot_z_list, minimum_coords_2)
    #plot_3d(X, Y, plot_z_list, minimum_coords_3)







__main__()
