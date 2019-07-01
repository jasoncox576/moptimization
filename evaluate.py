from optest import *
from plot import *


#FUNC: Current function being used
#FUNCS: List of all optimization benchmarks
FUNC = ackley
FUNCTS = [sphere_func, ackley, himmelblau, banana] 
#OP_ALGS = [gradient_descent, sgd, nesterov_momentum, newton, nelder_mead, diff_evol, stochastic_tunneling, simulated_annealing, custom] 
#OP_NAMES = ['gradient descent', 'sgd', 'nesterov momentum', 'newton\'s method', 'differential evolution', 'stochastic tunneling', 'simulated annealing', 'custom']

OP_ALGS = [gradient_descent, nelder_mead, newton, diff_evol]
OP_NAMES = ['gradient_descent', 'nelder_mead', 'newton', 'diff_evol']




#X,Y are the same for all functions, so just generate once and keep as global.
x_list = [x for x in np.arange(-DOMAIN_MAX, DOMAIN_MAX+0.005, DOMAIN_DX)]
y_list = [y for y in np.arange(-DOMAIN_MAX, DOMAIN_MAX+0.005, DOMAIN_DX)]


def generate_domain(func):
    z_list = np.empty([len(x_list), len(y_list)])
    for y_index in range(len(y_list)):
        for x_index in range(len(x_list)):
            z = func(x_list[x_index], y_list[y_index])
            z_list[y_index][x_index] = z
    
    return z_list


def evaluate_all():
    for func in FUNCTS:
        z_list = generate_domain(func)
        print("FUNC GENERATED")
        descend_list = []
        for OPTIMIZE in OP_ALGS:
            descends = evaluate_single(x_list, y_list, z_list, func, OPTIMIZE) 
            _, _, descend_z = descends
            descend_list.append(descend_z)
            
            print("OPTIMIZED") 

        plot_3x3(descend_list, OP_NAMES)





            






def evaluate_single(x_list, y_list, z_list, func, OPTIMIZE):
    final_coords, descends = OPTIMIZE(func,x_list, y_list, z_list)
    return descends
    #plot_2d(descends[0], descends[1], descends[2])
    #plot_3d(x_list, y_list, z_list, final_coords[:-1])
    




     


if __name__=='__main__':
    evaluate_all()
