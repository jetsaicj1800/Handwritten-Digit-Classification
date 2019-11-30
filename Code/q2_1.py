'''
implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        return the digit label provided by the algorithm
        '''
        
        
        
        distance = self.l2_distance(test_point)
        
        sort = np.argsort(distance)
        
        tie = True
        
        while tie:
            
            bag =[]
            
            #sort the l2 distance and find the max argument
            for i in range(0,k):
                index=sort[i]
                bag.append(self.train_labels[index])
                
            counts =np.bincount(bag)
            digit = np.argmax(counts)
            
            tie_flag = False
            
            #determine whether there are multiple max arguments
            for i in range(0,counts.size):
                if i!=digit:
                    if counts[i] == digit:
                        tie_flag = True
                        k = k-1
                        break
                    
            #exit the loop if there is only one max argument
            if not tie_flag:
                tie = False
            
        return digit

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k
    '''
    
    avg =[]
    
    for k in k_range:

        kf = KFold(n_splits=10)
        
        test_accuracy=[]
        
        for train_index, test_index in kf.split(train_data):
  
            knn = KNearestNeighbor(train_data[train_index,:], train_labels[train_index])
            test_tmp = classification_accuracy(knn, k, train_data[test_index,:], train_labels[test_index])
            test_accuracy.append(test_tmp)

        avg.append(np.mean(test_accuracy))
        
    
    y = np.arange(15) +1
    
    plt.plot(y[:],avg[:])
    
    plt.xlabel("K")
    plt.ylabel("accuracy")
    plt.title("accuracy vs K")
    
    
    plt.show()
        
    return avg
            

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    
    size = eval_labels.shape[0]
    correct = 0
    
    for i in range(0,size):
        
        predict_label = knn.query_knn(eval_data[i,:],k)
        if predict_label == eval_labels[i]:
            correct+=1

    accuracy = correct/float(size)
    
    return accuracy
    

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    #predicted_label = knn.query_knn(test_data[0], 100)
    
    #print(predicted_label)
    
    a_test = classification_accuracy(knn, 1, test_data, test_labels)
    a_train = classification_accuracy(knn, 1, train_data, train_labels)
    
    b_test = classification_accuracy(knn, 15, test_data, test_labels)
    b_train = classification_accuracy(knn, 15, train_data, train_labels)
    
    print(a_test, a_train, b_test, b_train)
    
    
    avg = cross_validation(train_data, train_labels)
    
    print (avg)
    
    c_test = classification_accuracy(knn, 3, test_data, test_labels)
    c_train = classification_accuracy(knn, 3, train_data, train_labels)
    
    print (c_test, c_train)
    
    
    

if __name__ == '__main__':
    main()