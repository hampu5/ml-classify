import numpy as np

def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1 + np.exp(-x))

def relu(x, deriv=False):
    if deriv:
        return np.greater(x, 0).astype(int)
    return np.maximum(x, 0)

num_epochs = 60000

#initialize weights
syn0 = 2*np.random.random((4,3)) - 1
syn1 = 2*np.random.random((3,1)) - 1


X_train = np.array([1, 1, 0, 0])
y_train = np.array([1])


#Step 3 Train Model

for j in range(num_epochs):
    #feed forward through layers 0,1, and 2
    k0 = X_train
    k1 = sigmoid(np.dot(k0, syn0))
    k2 = sigmoid(np.dot(k1, syn1))
    
    #how much did we miss the target value?
    k2_error = y_train - k2
    
    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(k2_error))))
    
    #in what direction is the target value?
    k2_delta = k2_error * sigmoid(k2, deriv=True)
    
    #how much did each k1 value contribute to k2 error
    k1_error = np.dot(k2_delta, syn1.T)
    
    k1_delta = k1_error * sigmoid(k1, deriv=True)
    
    print(k1)
    print(k2_delta)
    print(syn1)
    asd = np.dot(k1, k2_delta[0])
    print(asd)
    syn1 += asd.T
    syn0 += np.dot(k0, k1_delta)