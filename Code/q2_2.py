'''
implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv
from numpy.linalg import det

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    
    for i in range(10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        mean_digit = np.mean(i_digits,0)
        means[i,:] = mean_digit
    
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    
    means = compute_mean_mles(train_data, train_labels)
    
    d = 64
    
    for i in range(10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        
        for j in range(d):
            for k in range(d):
                var_1 = i_digits[:,j]-means[i,j]
                var_2 = i_digits[:,k]-means[i,k]
                var = np.mean(var_1*var_2,0)
            
                covariances[i,j,k] = var
                
                if j==k:
                    covariances[i,j,k]+=0.01
            
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    
    cov_all = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        
        cov_all.append(cov_diag.reshape(8,8))
        
    cov_concat = np.concatenate(cov_all, 1)
    
    
    cov_concat = np.log(cov_concat)
    
    plt.imshow(cov_concat, cmap='gray')
    plt.show()
        

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    
    n = digits.shape[0]
    d = 64
    
    density = np.zeros((n,10))
    
    for i in range(n):
        
        for k in range(10):
            
            det_k=det(covariances[k])
            inv_k=inv(covariances[k])
            #print (inv_k.shape)
        
            tmp = (2*np.pi)**(-d/2) *(det_k)**(-0.5)
            
            dif = digits[i] - means[k]
            
            mul_1 = np.dot(dif.transpose(),inv_k)
            mul_2 = np.dot(mul_1,dif)
        
            tmp2 = np.exp(-0.5 *mul_2)
            
            density[i,k] = np.log(tmp*tmp2)
            
    
    
    
    return density

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    
    alpha = 0.1
    
    log_gen = generative_likelihood(digits, means, covariances)
    
    gen = np.exp(log_gen)
    
    total_prob = (np.sum((gen*alpha),1)).reshape(-1,1)
    
    log_total = np.log(total_prob)
    
    log_con = log_gen + np.log(0.1) - log_total
        
    
    return log_con

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    
    N = digits.shape[0]
    total=0
    
    for i in range(N):
        total += cond_likelihood[i,int(labels[i])]
        
    avg = total/N
        
        
    return avg

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    
    N = digits.shape[0]
    
    class_label = []
    
    for i in range(N):
        
        label = np.argmax(cond_likelihood[i,:])
        
        class_label.append(label)
    
    return class_label

def classification_accuracy(predict, labels):
    '''
    Evaluate the classification accuracy of Gaussian Classifier on the given 'eval_data'
    using the labels
    '''
    
    N = labels.shape[0]
    accuracy=0
    
    for i in range(N):
        if predict[i]==labels[i]:
            accuracy+=1
    
    accuracy /=N
    
    return accuracy
    

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    
    plot_cov_diagonal(covariances)
        
    #print (means)
    #print (covariances)
    
    #den = generative_likelihood(train_data, means, covariances)
    
    #con = conditional_likelihood(train_data, means, covariances)
    
    train_avg = avg_conditional_likelihood(train_data, train_labels , means, covariances)
    
    test_avg = avg_conditional_likelihood(test_data, test_labels , means, covariances)
    
    print (train_avg)
    
    print (test_avg)
    
    train_post = classify_data(train_data, means, covariances)
    train_a = classification_accuracy(train_post, train_labels)
    print (train_a)
    
    test_post = classify_data(test_data, means, covariances)
    test_a = classification_accuracy(test_post, test_labels)
    print (test_a)
    

    
    

    # Evaluation

if __name__ == '__main__':
    main()