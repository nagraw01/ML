import numpy as np

class TwoLayersNN (object):
    """" TwoLayersNN classifier """

    def __init__ (self, inputDim, hiddenDim, outputDim):
        self.params = dict()
        self.params['w1'] = None
        self.params['b1'] = None
        self.params['w2'] = None
        self.params['b2'] = None
        #########################################################################
        # TODO: 20 points                                                       #
        # - Generate a random NN weight matrix to use to compute loss.          #
        # - By using dictionary (self.params) to store value                    #
        #   with standard normal distribution and Standard deviation = 0.0001.  #
        #########################################################################
        self.params['w1'] = np.random.normal(0, .0001, size=(inputDim, hiddenDim))
        self.params['b1'] = np.zeros(hiddenDim)
        self.params['w2'] = np.random.normal(0, .0001, size=(hiddenDim, outputDim))
        self.params['b2']= np.zeros(outputDim)


        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        TwoLayersNN loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to each parameter (w1, b1, w2, b2)
        """
        loss = 0.0
        grads = dict()
        grads['w1'] = None
        grads['b1'] = None
        grads['w2'] = None
        grads['b2'] = None

        #############################################################################
        # TODO: 40 points                                                           #
        # - Compute the NN loss and store to loss variable.                         #
        # - Compute gradient for each parameter and store to grads variable.        #
        # - Use Leaky RELU Activation at hidden and output neurons                  #
        # - Use Softmax loss
        # Note:                                                                     #
        # - Use L2 regularization                                                   #
        # Hint:                                                                     #
        # - Do forward pass and calculate loss value                                #
        # - Do backward pass and calculate derivatives for each weight and bias     #
        #############################################################################

        N = x.shape[0]
        W1,b1 = self.params['w1'],self.params['b1']
        W2, b2 = self.params['w2'], self.params['b2']

        s1 = x.dot(W1)+b1 #h1 == s1
        z1 = np.maximum(s1,.01*s1)#a1 == z1
        s2 = z1.dot(W2) + b2#h2 == s2
        z2 = np.maximum(s2,.01*s2)#a2 == z2

        S = z2.copy()# scores = S
        S -= np.max(S, axis=1).reshape(N, 1)
        Sp = S[np.arange(N), y].reshape(N, 1)# s_pred = Sp
        ltemp = (np.exp(Sp)) / (np.sum(np.exp(S), axis=1).reshape(N, 1))#prob = ltemp
        li = -1 * (np.log(ltemp))#l_i = li

        loss = np.sum(li) / N
        loss +=  reg *(np.sum(W1**2) + np.sum(W2**2))

        expS = np.exp(S)#exp_scores = expS
        P = expS / np.sum(expS, axis=1, keepdims=True)#probs == P

        zo = P#Dout == zo
        zo[range(N),y]-=1
        dz2=zo#da2 == dz2
        zo/=N
        ds2=dz2.copy()#dz2 == ds2
        ds2[[z2 < 0]]=.01
        ds2[[z2 >= 0] ]=1
        ds2*=dz2
        dw2 = np.dot(z1.T, ds2)
        dw2 += (reg *(W2)**2)
        dz1 = ds2.dot(W2.T)#da1 == dz1
        ds1=dz1.copy()#dz1 == ds1
        ds1[z1<0]=.01
        ds1[z1>=0]=1
        ds1*=dz1

        dw1 = np.dot(x.T,ds1)
        dw1 += (reg * (W1**2))
        db2= np.sum(ds2,axis=0)
        db1= np.sum(ds1,axis=0)

        grads['w1'] = dw1.copy()
        grads['w2'] = dw2.copy()
        grads['b1'] = db1.copy()
        grads['b2'] = db2.copy()
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, grads

    def train (self, x, y, lr=5e-3, reg=5e-3, iterations=100, batchSize=200, decay=0.95, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iterations):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (batchSize, D)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################
            batchID = np.random.choice(np.arange(x.shape[0]), batchSize, replace=False)
            xBatch = x[batchID]
            yBatch = y[batchID]
            loss, grads = self.calLoss(x=xBatch, y=yBatch, reg=reg)

            self.params['w1']-= (lr* grads['w1'])
            self.params['w2']-= (lr* grads['w2'])
            self.params['b1']-= (lr* grads['b1'])
            self.params['b2']-= (lr* grads['b2'])

            lossHistory.append(loss)

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            # Decay learning rate
            lr *= decay
            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Store the predict output in yPred                                    #
        ###########################################################################

        z1 = x.dot(self.params['w1']) + self.params['b1']
        a1 = np.maximum(z1, .01 * z1)
        z2 = a1.dot(self.params['w2']) + self.params['b2']
        a2 = np.maximum(z2, .01 * z2)
        yPred = np.argmax(a2, axis=1)



        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################
        y_pred = self.predict(x)
        acc = 100*(np.mean(y == y_pred))



        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc



