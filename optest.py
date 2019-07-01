import os.path
import pickle
import numpy as np
import math

from functions import *

#from sympy import symbols, diff


# Random seed for starting location
# of each descent technique
RAND_SEED = random.randint(1, 1000)

#set testing landscape

def mark_topography(x_stops, y_stops, z_list):

    """ This function is necessary for visualization beause plotting
        doesn't plot every single point used in the function.

        So we must 'pop up' a bunch of points in the surrounding area.
    """
    orig_z_list = z_list[:]
    

    for point in range(int(len(x_stops))):
        print("POINT:", str(x_stops[point]), str(y_stops[point]))
        x_ind = get_domain_index(x_stops[point], len(z_list))
        y_ind = get_domain_index(y_stops[point], len(z_list))
        print("x_ind", str(x_ind), " y_ind", y_ind)
        try:

            if z_list[x_ind][y_ind] <= (orig_z_list[x_ind][y_ind]): z_list[x_ind][y_ind] *= 2
            print("Increased val")
        except IndexError:
            print("index error")
            continue
            
        for buf in range(-100, 100):
            
            try:
                if z_list[x_ind][y_ind] <= 100:
                    z_list[x_ind+buf][y_ind] += 50
                    z_list[x_ind][y_ind+buf] += 50
                    z_list[x_ind+buf][y_ind+buf] += 50
                    z_list[x_ind-buf][y_ind+buf] += 50
            except IndexError:
                print("index error 2")
                continue
    
    return z_list


def nesterov_momentum(x_list, y_list, z_list, batches, epochs, learning_rate, alpha=0.0):

    """
    Nesterov momentum:

    Starts off at a certain random point.

    Makes a random initial jump

    Takes the gradient at the point it jumped to and corrects

    Take a big jump in the direction of previous gradient

    """







def diff_evol(FUNC, x_list, y_list, z_list, batches=5, learning_rate=None):

    pop_size = 1000
    
    #known as differential weight, range [0,2]
    f = 1e-100
    
    #cross probability- chance that a given point will be compared
    # to a linear combination of other random points.
    # it is only set equal to the new point if the new point is a better
    # solution, of course
    cross_prob = 0.9

    sample = [(random.choice(x_list), random.choice(y_list)) for x in range(pop_size)]

    min_point = sample[0]
    min_indices = get_domain_index(min_point[0], len(z_list)), get_domain_index(min_point[1], len(z_list))
    min_val = z_list[min_indices[0]][min_indices[1]]
    print("MIN VAL", min_val)

    x_stops = [min_point[0]]
    y_stops = [min_point[1]]
    z_stops = [min_val]


    for iteration in range(batches):
        for point in sample:

            a = (point[0], point[1])
            b = (point[0], point[1])
            c = (point[0], point[1])

            while (a == point) or (b == point) or (c == point):
                a_index, b_index, c_index = np.random.choice([x for x in range(len(sample))], 3, replace=False)
                a = sample[a_index]
                b = sample[b_index]
                c = sample[c_index]



            # TODO: None of these should be equal to point. Check for this. 
            rand_x = random.uniform(0.0, 1.0)
            rand_y = random.uniform(0.0, 1.0)

            #just initializing
            update_x = 0
            update_y = 0
            if rand_x < cross_prob:
                update_x = a[0] + f * (b[0] - c[0])
            if rand_y < cross_prob:
                update_y = a[1] + f * (b[1] - c[1])
            #print("UPDATES", update_x, update_y)

            try:
                update_x_index = get_domain_index(update_x, len(z_list))
                update_y_index = get_domain_index(update_y, len(z_list))


                point_x_index = get_domain_index(point[0], len(z_list))
                point_y_index = get_domain_index(point[1], len(z_list))

            
            
                if z_list[update_x_index][update_y_index] <= z_list[point_x_index][point_y_index]:
                    point = (update_x, update_y)

                if z_list[point_x_index][point_y_index] < min_val:
                    min_val = z_list[point_x_index][point_y_index]
                    min_point = point[0], point[1]
                    print("NEW MIN", min_point)
            except IndexError:
                continue 
        x_stops.append(min_point[0])
        y_stops.append(min_point[1])
        z_stops.append(FUNC(min_point[0], min_point[1]))
        print("Diffevol min: ", z_stops[-1])
    

    final_x, final_y = min_point
    final_z = FUNC(final_x, final_y)
    return ((final_x, final_y, final_z), (x_stops, y_stops, z_stops))




def nelder_mead(FUNC, x_list, y_list, z_list, batches=30):
    
    """
    Downhill simplex method
    """
    print("NELDER MEAD OPTIMIZATION")
    #lower_bound = int((len(x_list)/3) - (len(x_list)/3.5))
    lower_bound = 0
    print(lower_bound)
    #upper_bound = int((len(x_list)/3) + (len(x_list)/3.5))
    upper_bound = len(x_list)-1
    print(upper_bound)

    domain_subsection = x_list[lower_bound:upper_bound]
    print(str(domain_subsection[0]), str(domain_subsection[-1]))
    
    # each point is a three-tuple (x,y,z)
    
    p1_x = random.choice(domain_subsection)
    print(str(p1_x))
    p1_y = random.choice(domain_subsection)
    print(str(p1_y))


    p1 = [p1_x, p1_y, FUNC(p1_x, p1_y)]
        
    simplex_distance_offset = DOMAIN_MAX/2


    
    
    p2 = [p1[0] + simplex_distance_offset, p1[1], 0] 
    p2[2] = FUNC(p2[0], p2[1])
   
    p3 = [p1[0], p1[1] + simplex_distance_offset, 0]
    p3[2] = FUNC(p3[0], p3[1])
    
          
    simplex = [p1,p2,p3]
    print("UNSORTED: " + str(simplex))


    x_stops = []
    y_stops = []
    z_stops = []
    
    # sorts based off of the height (z-coordinate)
    # The simplex array should always be sorted
    
    for iteration in range(batches):
        
        simplex.sort(key = lambda x: x[2])
        #print("SORTED: " + str(simplex))

        #compute the centroid from the points in the simplex sans p3
        sums = [sum(x) for x in zip(*simplex[:-1])]
        p0 = [sums[0]/2, sums[1]/2, FUNC(sums[0]/2, sums[1]/2)]
        x_stops.append(p0[0])
        y_stops.append(p0[1])
        z_stops.append(p0[2])
        print("Z is::", p0[2])

        #compute reflected point (reflection of the fourth point across the centroid p0
        #coeff of reflection initialized to 1
        A = 1
        pr = [0, 0, 0]
        pr[0] = p0[0] + A*(p0[0] - simplex[-1][0]) 
        pr[1] = p0[1] + A*(p0[1] - simplex[-1][1])
        pr[2] = FUNC(pr[0], pr[1])
         
        if (pr[2] < simplex[-2][2]) and (pr[2] >= simplex[0][2]):
            #print("REFLECTION")
            simplex[-2] = pr
            continue

        elif pr[2] < simplex[0][2]:
            #print("EXPANSION")
            # coefficient gamma (G) must be > 1
            G = 2
            pe = [0, 0, 0]
            pe[0] = p0[0] + G*(pr[0] - p0[0])
            pe[1] = p0[1] + G*(pr[1] - p0[1])
            pe[2] = FUNC(pe[0],pe[1]) 
        
            if pe[2] < pr[2]:
                simplex[-1] = pe
            else:
                simplex[-1] = pr

            continue 
    
        else:
            # it is certain at this point that the reflected point is worse
            # than the second worst point
            #print("CONTRACTION")
            rho = 0.5

            pc = [0, 0, 0]

            pc[0] = p0[0] + rho*(simplex[-1][0] - p0[0])
            pc[1] = p0[1] + rho*(simplex[-1][1] - p0[1])
            pc[2] = FUNC(pc[0], pc[1])
    
            if pc[2] < simplex[-1][2]:
                simplex[-1] = pc
                continue

            else:
                # 'shrink' step
                # replace all points except the first one
                #print("SHRINK")
                sigma = 0.5

                best_point = simplex[0]

                for point in simplex[1:]:
                    point[0] = best_point[0] + sigma*(point[0] - best_point[0]) 
                    point[1] = best_point[1] + sigma*(point[1] - best_point[1]) 
            
        
    #z_list = mark_topography(x_list, y_list, z_list) 
    
    #plt.plot(x_stops, label="amoeba-x")
    #plt.plot(y_stops, label="amoeba-y")
    #plt.legend(loc='upper right') 
    final_z = p0[2]
    print(x_stops)
    #print(z_stops[:100])
        
    return ((p0[0], p0[1], final_z), (x_stops, y_stops, z_stops))




def newton(FUNC, x_list, y_list, z_list, batches=5, learning_rate=1e-9):

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
    print("NEWTON'S METHOD OPTIMIZATION")


    random.seed(RAND_SEED)
    descending_x = x_list[random.randint(0, len(x_list)-1)] 
    random.seed(RAND_SEED)
    descending_y = y_list[random.randint(0, len(y_list)-1)]
	
    x_stops = [descending_x]
    y_stops = [descending_y] 
    z_stops = [FUNC(descending_x, descending_y)]

    print("Newton STARTING LOCATIONS: X: " + str(descending_x) + " Y: " + str(descending_y))

    #vec_gradient = np.vectorize(np.gradient)

    z_list_t = np.transpose(z_list)

    x_grads = np.array([np.gradient(y) for y in z_list])
    y_grads = np.array([np.gradient(x) for x in z_list_t])

    x_grads_t = np.transpose(x_grads)

    for iteration in range(batches):

        try:
            index_prev_x = get_domain_index(descending_x, len(z_list))
            #print(index_prev_x)
            index_prev_y = get_domain_index(descending_y, len(z_list))
            #print(index_prev_y)

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
            #do you really need hessian though if changing x and y individually? I guess you do.
            descending_x -= (learning_rate * hessian_det * x1_grad) 
            descending_y -= (learning_rate * hessian_det * y1_grad)

            
            x_stops.append(descending_x)
            y_stops.append(descending_y)
            z_stops.append(FUNC(descending_x, descending_y))

        
        except IndexError:
            print("ERROR: OUT OF BOUNDS")
            learning_rate /= 10
            #return
    final_z = z_stops[-1]
    return ((descending_x, descending_y, final_z), (x_stops, y_stops, z_stops))


def gradient_descent(FUNC, x_list, y_list, z_list, epochs=1000, batches=10, learning_rate=0.1):
    
    print("GRADIENT DESCENT OPTIMIZATION")

    random.seed(RAND_SEED)
    descending_x = x_list[random.randint(0, len(x_list)-1)] 
    random.seed(RAND_SEED)
    descending_y = y_list[random.randint(0, len(y_list)-1)]

    z_stops = [FUNC(descending_x, descending_y)]

    x_stops = [descending_x]
    y_stops = [descending_y] 

    #print("STARTING LOCATIONS: X: " + str(descending_x) + " Y: " + str(descending_y))

    for epoch in range(epochs): 
        for iterations in range(batches):
               
            x_index = get_domain_index(descending_x, len(z_list)) 
            print(descending_x)
            y_index = get_domain_index(descending_y, len(z_list))  


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

            #print("T_x" + str(T_x))
            #print("Descending x: " + str(descending_x))
            #print("Descending x: {}".format(descending_x))
            descending_y -= T_y

                
            x_stops.append(descending_x)
            y_stops.append(descending_y)
            z_stops.append(FUNC(descending_x, descending_y))

            #print("X-grad: ", str(x_grad), "Y-grad: ", str(y_grad), "rand_x: ", str(x_list[rand_x]), "rand_y", str(y_list[rand_y]))
        
        #You NEED the below line
        #learning_rate =  learning_rate / 10
        
    x_label = 'x_standard'
    y_label = 'y_standard'
    

    #z_list = mark_topography(x_stops, y_stops, z_list)
    final_z = FUNC(descending_x, descending_y) 
    print("Marked topography")

    return ((descending_x, descending_y, final_z), (x_stops, y_stops, z_stops))


def sgd(FUNC, x_list, y_list, z_list, batches, epochs=1000, learning_rate=0.1, momentum=False, alpha=0.0):

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
            #print("T_x" + str(T_x))
            #print("Descending x: " + str(descending_x))
            #print("Descending x: {}".format(descending_x))
            descending_y -= T_y


            if momentum:
                descending_x += (alpha) * (prev_update_x)
                #print("Prev_update_x: " + str(prev_update_x))
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


def simulated_annealing(FUNC, x_list, y_list, z_list, batches):
    pass

def stochastic_tunneling(FUNC, x_list, y_list, z_list, batches):
    return 

def custom(FUNC, x_list, y_list, z_list, batches):
    pass






"""


def __main__():


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


    minimum_coords_1, z_list = gradient_descent(x_list, y_list, z_list, 1000, 10, 0.1)


    #lower offset gives a more accurate function plot 
    offset = 100
    
    
    
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



"""
