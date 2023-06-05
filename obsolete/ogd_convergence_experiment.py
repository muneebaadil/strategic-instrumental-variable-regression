import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib
import matplotlib.pyplot as plt
import sys

# Gradient Descent method with two ways of calculating gradient
def sgd(start, T, W, theta_star, z, g, learning_rate=0.01, tor=1e-05, flag=True, theta_min=-0.105):
    vec = start
    # list of theta
    Theta = []
    # population lost 
    total_loss = []
    # actual loss
    actual_loss = []
    # list of gradient
    gradient_lst = []
    dist_lst = []
    lr = learning_rate
    for i in range(T):
        learning_rate = lr/((i+1)**0.5)
        x = z[i] + W*W*vec
        y = x*theta_star + g[i]
        if flag:
            gradient = calc_gradient(x, y, vec, W, theta_star)
        else:
            gradient = calc_gradient_PP(x, y, vec, W, theta_star)
        if np.abs(gradient) < tor:
            break
        gradient_lst.append(gradient)
        vec = vec - learning_rate*gradient
        Theta.append(vec)
        actual_loss.append((y - x*vec)**2)
        total_loss.append(risk_func(vec, W, theta_star))
        dist_lst.append(norm(vec - (theta_min)))
    return Theta, total_loss, actual_loss, gradient_lst, dist_lst

# 1-D setting gradient
def calc_gradient(x, y, theta, W, theta_star):
    gradient = 2*(x*theta - y)*(x+W*W*(theta-theta_star))
    return gradient

# Performative prediction simple gradient 
def calc_gradient_PP(x, y, theta, W, theta_star):
    gradient = 2*(x*theta - y)*x
    return gradient

# Population risk
def risk_func(theta, W, theta_star, var_g=15.0, var_z=0.3, cov_zg=-6.5):
    res = var_g + var_z*(theta - theta_star)**2 + W**4 * theta**2 * (theta - theta_star)**2 - 2*cov_zg*(theta - theta_star)
    return res

def plot_loss(Theta, Theta_PP, total_loss, total_loss_PP, T=1000, start_PP=0.5, W=3, theta_star=1):    
    # Plot the function
    num_points = 100
    X = np.linspace(-1,2,num_points)
    Y = var_g + var_z*(X - theta_star)**2 + W**4 * X**2 * (X - theta_star)**2 - 2*cov_zg*(X - theta_star)
    fig = plt.figure(figsize=(6,4),tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.ylim([0,20])
    plt.xlim([-1,1.5])
    plt.yticks(np.linspace(0,20,5))
    plt.plot(X,Y)
    
    plt.scatter(Theta,total_loss, c='blue', marker='*', label='SGD')
    plt.annotate('Global Minima', size='large', ha='center', va='bottom', xytext=(-0.7,3), xy=(-0.2, 2.4), arrowprops= {'facecolor':'black', 'width':0.5})
    plt.annotate('SGD Direction', size='large', ha='center', va='bottom', xytext=(0.6,8), xy=(0.45, 4), arrowprops= {'facecolor':'black', 'arrowstyle':'->'})
    plt.scatter(start_PP, risk_func(start_PP, W, theta_star), c='red', marker='D', s=150, label='Starting point')

    plt.scatter(Theta_PP, total_loss_PP, c='green', marker='o', label='SSGD')
    plt.annotate('Local Minima', size='large', ha='center', va='bottom', xytext=(1.5,12.5), xy=(1, 14.3), arrowprops= {'facecolor':'black', 'width':0.5})
    plt.annotate('SSGG Direction', size='large', ha='center', va='bottom', xytext=(0.35,15), xy=(0.95, 16), arrowprops= {'facecolor':'black', 'arrowstyle':'->'})
    plt.legend(bbox_to_anchor=(1,0), loc='lower right')
    plt.xlabel('Assessment Rule', fontsize=14)
    plt.ylabel('Population Risk',labelpad=5,loc='top', fontsize=14)
    # plt.savefig('SGD_SSGD_convergence_plot.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()

def plot_distance(dist_lst_mean, dist_lst_PP_mean, T=1000):
    dist_mean = np.mean(dist_lst_mean, axis=0)
    dist_mean_PP = np.mean(dist_lst_PP_mean, axis=0)
    dist_var = np.std(dist_lst_mean, axis=0)
    dist_var_PP = np.std(dist_lst_PP_mean, axis=0)
    fig = plt.figure(figsize=(6,4),tight_layout=True)
    plt.errorbar(range(T), dist_mean, yerr=dist_var, c='darkblue', ecolor='lightblue', label='SGD')
    plt.errorbar(range(T), dist_mean_PP, yerr=dist_var_PP, c='darkgreen', ecolor='lightgreen', label='SSGD')
    plt.legend(loc='best')
    plt.xlabel('Number of rounds', fontsize=14)
    plt.ylabel('Distance to ' +  r'$\theta_\min$', fontsize=14)
    # plt.savefig('SGD_SSGD_distance_plot.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()


def main(start, start_PP, theta_star, W=3, nu=0.001,var_z=0.3, var_g=15.0, cov_zg=-6.5, T=1000, epochs=10):
    dist_lst_mean = np.zeros((epochs, T))
    dist_lst_PP_mean = np.zeros((epochs, T))
    for i in range(epochs): 
        mu = [0,0]
        cov = [[var_z, cov_zg],[cov_zg, var_g]]
        z, g = np.random.multivariate_normal(mu, cov, T).T
        Theta, total_loss, actual_loss, gradient_lst, dist_lst = sgd(start,T,W,theta_star, z, g, learning_rate=nu)
        Theta_PP, total_loss_PP, actual_loss_PP, gradient_lst_PP, dist_lst_PP = sgd(start_PP, T, W, theta_star,z, g, learning_rate=nu, flag=False)  
        dist_lst_mean[i,:] = dist_lst
        dist_lst_PP_mean[i,:] = dist_lst_PP 
    if theta_star == 1.0:
        plot_loss(Theta, Theta_PP, total_loss, total_loss_PP)
    else:
        plot_distance(dist_lst_mean, dist_lst_PP_mean)

if __name__ == '__main__':
    # theta_star = 1 # non-invex
    # theta_star = 0.7 # intex
    theta_star = float(sys.argv[1])
    start = 0.5
    start_PP = 0.5
    W = 3
    T = 1000
    epochs=10
    nu = 0.001
    var_z = 0.3
    var_g = 15.0
    cov_zg = -6.5
    main(start, start_PP, theta_star)

