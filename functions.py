import random
import numpy as np
import math


def ackley(x, y):

    """
    Ackley function
    Looks really cool (when it generates properly)
    for some reason this isn't generating properly.. :(

    """
       
    # TODO inconsistnet use of math library vs numpy??

    return (-20 * math.exp(-0.2 * math.sqrt(0.5 * ((x**2) + (y**2))))) - math.exp(0.5 * (math.cos(2*np.pi*x) + math.cos(2*np.pi*y))) + np.e + 20




def sphere_func(x, y): 
    
    """
    Sphere function for evaluation
    """

    #return sum([x_i**2 for x_i in np.arange(0, x+1, 0.001)]) 
    return (x**2) + (y**2)
    


def himmelblau(x, y): 
    
    """ 
    Himmelblau's function
    
    Hilly landscape. 
    Nonconvex, but all four minimums are the same. 
    """
    
    return (((x**2) + y - 11)**2) + ((x + (y**2) - 7)**2)




def banana(x, y): 

    """
    Rosenbrock banana function
    classic benchmark for testing 
    optimization algorithms

    Convex, min in the middle of the valley

    """

    #x, y should be a single pair.


    a = 1 
    b = 2 

    return ((a-x)**2) + (b*(y-(x**2)))**2







