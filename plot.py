from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot_3x3(zs, labels):

    fig, ax = plt.subplots(nrows=2, ncols=3)
    index=0
    for row in ax:
        for col in row:
            col.plot(zs[index])
            #plt.ylabel(labels[index])
            col.set_title(labels[index])
            if index < len(labels)-1:
                index += 1

    plt.show()



def plot_2d(xs, ys, zs):

    plt.subplot(2, 1, 1)
    plt.plot(zs, 'o-')
    plt.title('Coordinates over time')
    plt.ylabel('Z')

    plt.subplot(2, 1, 2)
    plt.plot(xs, '.-', label='X')
    plt.plot(ys, '.-', label='Y')
    plt.xlabel('Steps')
    plt.ylabel('X, Y')
    plt.legend(loc='upper right')

    plt.show()

def plot_3d(FUNC, X, Y, Z, minimum_coords):
    

    x_min, y_min = minimum_coords

    fig = plt.figure()
    ax = Axes3D(fig)

    Y, X = np.meshgrid(Y, X) 



    print("=========================================================")


    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="terrain", antialiased =True)
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="terrain")

    #text = "Global Minimum: " + str(x_min) + " : " + str(y_min) + " : " + str(banana(x_min, y_min))
    text = "Global Minimum: " + str(x_min) + " : " + str(y_min) + " : " + str(FUNC(x_min, y_min))

    #ax.text(x_min, y_min, banana(x_min, y_min), text, size=20) 
    ax.text(x_min, y_min, FUNC(x_min, y_min), text, size=20) 
    plt.show()






















