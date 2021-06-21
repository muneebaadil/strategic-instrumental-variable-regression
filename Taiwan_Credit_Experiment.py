import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
from sklearn import preprocessing
import pickle as pkl
import time


def read_input():
    ### load data & construct initial (z) & observable (x) features

    ## load data
    data = pd.read_csv('UCI_Credit_Card.csv',index_col=0)
    data = data.rename(columns={'PAY_0':'PAY_1', 'default.payment.next.month':'DEFAULT'})

    unmodifiable_features = data[['SEX', 'EDUCATION', 'MARRIAGE', 'AGE']]

    ## construct 6 features (ratio & difference btween bill & payment amount per month)
    # create new feature: ratio btween BILL_AMT & PAY_AMT 
    data['PAY_RATIO1'] = data['PAY_AMT1']/data['BILL_AMT1']
    data['PAY_RATIO2'] = data['PAY_AMT2']/data['BILL_AMT2']
    data['PAY_RATIO3'] = data['PAY_AMT3']/data['BILL_AMT3']
    data['PAY_RATIO4'] = data['PAY_AMT4']/data['BILL_AMT4']
    data['PAY_RATIO5'] = data['PAY_AMT5']/data['BILL_AMT5']
    data['PAY_RATIO6'] = data['PAY_AMT6']/data['BILL_AMT6']

    # create new feature: difference btween PAY_AMT & BILL_AMT
    data['PAY_DIFF1'] = data['PAY_AMT1'] - data['BILL_AMT1']
    data['PAY_DIFF2'] = data['PAY_AMT2'] - data['BILL_AMT2']
    data['PAY_DIFF3'] = data['PAY_AMT3'] - data['BILL_AMT3']
    data['PAY_DIFF4'] = data['PAY_AMT4'] - data['BILL_AMT4']
    data['PAY_DIFF5'] = data['PAY_AMT5'] - data['BILL_AMT5']
    data['PAY_DIFF6'] = data['PAY_AMT6'] - data['BILL_AMT6']

    # clean up NAN values
    data_scaled = data.fillna(1)
    data_scaled.replace([np.inf, -np.inf], 1, inplace=True)

    ### construct real-valued outcomes (y), set initial (z) & observed (x) features
    ## let z = pay vs. bill ratios & differences for the first 3 months
    z = data_scaled[['PAY_DIFF4', 'PAY_DIFF5', 'PAY_DIFF6', 'PAY_RATIO4', 'PAY_RATIO5', 'PAY_RATIO6']].to_numpy()
    ## let x = pay vs. bill ratios & differences for the last 3 months
    x = data_scaled[['PAY_DIFF1', 'PAY_DIFF2', 'PAY_DIFF3', 'PAY_RATIO1', 'PAY_RATIO2', 'PAY_RATIO3']].to_numpy()
    ## outcome y is currently binary (1 = credit default, 0 = no default)
    y = data_scaled['DEFAULT'].to_numpy()

    ## create continuous y (denoting true rate of defaulting on credit card)
    # divide binary y in half (0s stay same and 1s become 0.5), then add noise ~ Unif(0, 0.5)
    y_scaled = y/2 + np.random.uniform(low=0, high=0.5, size=len(y)) # add uniform noise

    ## scale z and x to be zero mean
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    z_scaled = scaler.fit_transform(z)
    
    # OLS
    m = x_scaled.shape[1]
    x_sum = np.zeros([m,m])
    xy_sum = np.zeros(m)
    for i in range(len(x)):
        x_sum += np.outer(x_scaled[i], x_scaled[i])
        xy_sum += x_scaled[i]*y_scaled[i]
    theta_star = np.matmul(inv(x_sum), xy_sum)

    # supposed play noises throughout 
    theta = np.random.randn(z_scaled.shape[0], m)

    theta_sum = np.zeros([m,m])
    thetax_sum = np.zeros([m,m])
    thetay_sum = np.zeros(m)
    for i in range(len(x)):
        theta_sum += np.outer(theta[i], theta[i])
        thetax_sum += np.outer(theta[i], x_scaled[i])
        thetay_sum += theta[i]*y_scaled[i]
    EWW = np.matmul(inv(theta_sum), thetax_sum)
    return data_scaled, x_scaled, y_scaled, z_scaled, theta_star, EWW



def test_params(data, EWW, Z, theta_star):
    # split data into young and old people
    young = data.loc[data['AGE']<35]
    elder = data.loc[data['AGE']>=35]
    split_indices = elder.index - 1
    
    # Creating confounding effect in dataset
    Z[young.index-1] -= (np.random.randn(len(young), Z.shape[1]) + 0.6 )
    Z[elder.index-1] += np.random.randn(len(elder), Z.shape[1]) + 0.6
    scaler = preprocessing.StandardScaler()
    Z = scaler.fit_transform(Z)
    
    T = data.shape[0]
    # confounder g
    g = np.ones(T)*0.4   
    g[split_indices]*=-1
    g += np.random.normal(0.2212, 0.2, T) # 0.2212 = sample default probability

    # assessment rule: assume playing random noise 
    m = EWW.shape[0]
    theta = np.random.randn(T, m)
    # effort conversion matrix WW^T
    WW_lst = []
    # add noise to EWW
    for i in range(T):
        noise = np.zeros((m, m))
        for j in range(m):
            for k in range(m):
                mu = EWW[j,k]/2
                var = np.abs(EWW[j,k]/5)
                noise[j,k] = np.random.normal(mu, var)
        if i in split_indices:
            WW = EWW.copy()
            WW += noise
        else:
            WW = EWW.copy()
            WW -= noise
        WW_lst.append(WW)
    WW_lst = np.array(WW_lst)
    # create observable features x
    x_noised = np.zeros([T, m])
    for i in range(T):
        x_noised[i] = Z[i] + np.matmul(WW_lst[i], theta[i])
    # create outcome y
    #y_noised = np.matmul(x_noised, theta_star) + g
    y_noised = np.clip(np.matmul(x_noised,theta_star)+g,0,1)#truncated to [0,1]
    return x_noised, y_noised, theta, young.index, elder.index

def ols(X,Y,T):
    X_tilde = np.hstack((X,np.ones((len(X),1)))) 

    m = X.shape[1]
    x_sum = np.zeros([m+1,m+1])
    xy_sum = np.zeros(m+1)

    for i in range(T):
      x_sum += np.outer(X_tilde[i],X_tilde[i])
      xy_sum += X_tilde[i]*Y[i]

    theta_hat_ols = np.matmul(inv(x_sum),xy_sum)
    return theta_hat_ols[:m]

def tsls(X,Y,Theta,T):
    m = X.shape[1]
    Theta_tilde = np.hstack((Theta, np.ones((len(Theta),1))))
    
    theta_tilde_sum = np.zeros([m+1,m+1])
    thetax_tilde_sum = np.zeros([m+1,m])
    thetay_tilde_sum = np.zeros(m+1)
    
    for i in range(T):
        theta_tilde_sum += np.outer(Theta_tilde[i], Theta_tilde[i])
        thetax_tilde_sum += np.outer(Theta_tilde[i], X[i])
        thetay_tilde_sum += Theta_tilde[i]*Y[i]
    
    omega_hat = np.matmul(inv(theta_tilde_sum), thetax_tilde_sum)
    z_bar = omega_hat[m,:]
    omega_hat = omega_hat[:m,:m]
    
    lmbda_hat = np.matmul(inv(theta_tilde_sum), thetay_tilde_sum)
    gztheta_bar = lmbda_hat[m]
    lmbda_hat = lmbda_hat[:m]
    
    estimate = np.matmul(inv(omega_hat), lmbda_hat)
    return estimate

def run_test_params(X,Y,Theta,theta_star,T=30000,k=1000):
    estimates_lst = np.zeros([int(T/k),2,X.shape[1]])
    error_lst = np.zeros([int(T/k),2])
    i=0
    for t in range(k,T+1,k):
        # estimates
        ols_estimate = ols(X,Y,t)
        tsls_estimate = tsls(X,Y,Theta,t)
        estimates_lst[i,:] += [ols_estimate,tsls_estimate]

        # errors
        ols_error = norm(theta_star-ols_estimate)
        tsls_error = norm(theta_star-tsls_estimate)
        error_lst[i] = [ols_error,tsls_error]

        i+=1
    return estimates_lst, error_lst



def plot_error_estimation(error_lst_mean, T=30000, k=100):
    fig = plt.figure(figsize=(6,4),constrained_layout=True)

    # run OLS & 2SLS 10 times, plot mean & std deviation of L-2 norm of 
    # the estimation errors ||theta* - theta_hat|| over 30000 rounds
    ols_mean = np.mean(error_lst_mean, axis=0)[:,0] # mean OLS error
    tsls_mean = np.mean(error_lst_mean, axis=0)[:,1] # mean 2SLS error
    ols_std = np.std(error_lst_mean,axis=0)[:,0] # std OLS error
    tsls_std = np.std(error_lst_mean,axis=0)[:,1] # std 2SLS error

    # error plots with standard deviation errror bars
    plt.errorbar(list(range(k,T+1,k)), ols_mean, yerr=ols_std, color='darkorange', ecolor='wheat', label='OLS',elinewidth=10)
    plt.errorbar(list(range(k,T+1,k)), tsls_mean, yerr=tsls_std, color='darkblue', ecolor='lightblue', label='2SLS',elinewidth=10)

    # O(1/sqrt(T)) line
    plt.plot(range(1,T+1), 3/np.sqrt(range(1,T+1)), ls='--', color='red', linewidth=3, label='3/sqrt(T)')
    plt.plot(range(1,T+1), 3/np.sqrt(range(1,T+1)), ls='-', color = 'white', linewidth = 4)
    plt.plot(range(1,T+1), 3/np.sqrt(range(1,T+1)), ls='--', color = 'red', linewidth = 4)

    plt.ylim(0,0.15)
    plt.xlim(2000,T)

    plt.xlabel('Number of applicants (rounds)', fontsize=16)
    plt.ylabel('Credit effect estimate error', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(bbox_to_anchor=(1,1), loc='best', fontsize=14)
    #plt.title('Error rate of OLS and 2SLS')

    plt.savefig('error_estimation_credit.png', dpi=500, bbox_inches='tight')
    plt.show()



def plot_dist_default_prob(y_noised, young_index, elder_index):
    ## plot default probability (outcome y)
    # all
    plt.hist(y_noised,bins='auto',label='all')
    plt.axvline(x=np.mean(y_noised),color='blue',linestyle='--', linewidth = 3, label='all mean')
    plt.axvline(x=np.mean(y_noised), linestyle='-', color = 'blue', linewidth = 4)
    plt.axvline(x=np.mean(y_noised), linestyle='--', color = 'white', linewidth = 4)

    # young
    plt.hist(y_noised[young_index-1],bins='auto', color='#2ca02c', label='young', alpha=0.95)
    plt.axvline(x=np.mean(y_noised[young_index-1]), linestyle='--', color = 'green', linewidth = 3, label='young mean')
    plt.axvline(x=np.mean(y_noised[young_index-1]),color='green',linestyle='-', linewidth = 4)
    plt.axvline(x=np.mean(y_noised[young_index-1]), linestyle='--', color = 'white', linewidth = 4)


    # elder
    plt.hist(y_noised[elder_index-1],bins='auto',label='elder', alpha=.75)
    plt.axvline(x=np.mean(y_noised[elder_index-1]),color='orange',linestyle='--', linewidth = 3, label='elder mean')
    plt.axvline(x=np.mean(y_noised[elder_index-1]), linestyle='-', color = 'orange', linewidth = 4)
    plt.axvline(x=np.mean(y_noised[elder_index-1]), linestyle='--', color = 'white', linewidth = 4)

    plt.xlim(0,1)
    plt.yscale('log')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('True probability of defaulting', fontsize=14)
    plt.ylabel('Number of applicants (log scale)', fontsize=14)

    #plt.legend(bbox_to_anchor=(.342, 1.03), loc='upper left', fontsize=12, ncol=3)
    plt.legend(bbox_to_anchor=(.16, 1), loc='best', fontsize=12, ncol=2)

    plt.savefig('all_outcome_credit.png', dpi=500, bbox_inches='tight')
    plt.show()



def plot_causal_estimate(estimates_lst_mean, theta_star, T=30000, k=100):
    fig = plt.figure(figsize=(6,4),constrained_layout=True)
    month = 0
    ## estimates of causal effect of first month payment on default probability
    ols_mean = np.mean(estimates_lst_mean, axis=0)[:,0,month] # mean OLS error
    tsls_mean = np.mean(estimates_lst_mean, axis=0)[:,1,month] # mean 2SLS error
    ols_std = np.std(estimates_lst_mean,axis=0)[:,0,month] # std OLS error
    tsls_std = np.std(estimates_lst_mean,axis=0)[:,1,month] # std 2SLS error

    # error plots with standard deviation errror bars
    plt.errorbar(list(range(k,T+1,k)), ols_mean, yerr=ols_std, color='darkorange', ecolor='wheat', label='OLS estimate',elinewidth=10)
    plt.errorbar(list(range(k,T+1,k)), tsls_mean, yerr=tsls_std, color='darkblue', ecolor='lightblue', label='2SLS estimate',elinewidth=10)

    # 1/sqrt(T) line
    month = 3
    plt.axhline(theta_star[month], ls='--', color='red', linewidth=3, label='True effect on default rate')
    plt.axhline(theta_star[month], ls='-', color = 'white', linewidth = 4)
    plt.axhline(theta_star[month], ls='--', color = 'red', linewidth = 4)

    plt.ylim(-0.07,0.04)
    plt.xlim(2000,T)

    plt.xlabel('Number of applicants (rounds)', fontsize=16)
    plt.ylabel('Effect of first month payment \n on default probability', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    #plt.legend(bbox_to_anchor=(1,1), loc='best', fontsize=14)
    plt.legend(fontsize=14)
    #plt.title('Error rate of OLS and 2SLS')

    plt.savefig('estimate_convergence_credit.png', dpi=500, bbox_inches='tight')
    plt.show()


def main(T=30000, epochs=10, k=100):
    data_scaled, x_scaled, y_scaled, z_scaled, theta_star, EWW = read_input()
    

    estimates_lst_mean = np.zeros((epochs, int(T/k), 2, z_scaled.shape[1]))
    error_lst_mean = np.zeros((epochs, int(T/k), 2))

    for i in range(epochs):
        np.random.seed(i+10)
        EWW_noised = EWW + np.random.randn(EWW.shape[0], EWW.shape[1])
        x_noised, y_noised, theta, young_index, elder_index = test_params(data_scaled, EWW_noised, z_scaled, theta_star)
        ols_estimate = ols(x_noised,y_noised,T=30000)
        tsls_estimate = tsls(x_noised,y_noised,theta,T=30000)
#         print("OLS Gap: ",norm(theta_star-ols_estimate))
#         print("2SLS Gap: ",norm(theta_star-tsls_estimate))
#         print("USABLE: ", norm(theta_star-tsls_estimate) < norm(theta_star-ols_estimate))
        estimates_lst, error_lst = run_test_params(x_noised, y_noised, theta, theta_star, T=30000, k=k)
        estimates_lst_mean[i,:] = estimates_lst
        error_lst_mean[i,:] = error_lst
    # Generate Estimation Error plot
    plot_error_estimation(error_lst_mean, T=T, k=k)
    # Generate effect of first month payment on default probability estimation plot
    plot_causal_estimate(estimates_lst_mean, theta_star=theta_star, T=T, k=k)
    # Generate default probability distribution shift plots
    plot_dist_default_prob(y_noised, young_index, elder_index)


if __name__ == "__main__":
    main()


