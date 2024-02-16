import numpy as np

class TrainModel:
    #model training
    def fit(self,X_train,X_test,Y_train,Y_test,alpha,epochs):
        n_in = X_train
        n_out = Y_train
        Y = Y_train
        n = n_in.shape[0]  # this equals to 36
        # calculate the standard deviation
        std = np.sqrt(2.0 / n_out.shape[0])
        # calculate the weights
        W = np.random.randn(n_out.shape[0], n_in.shape[0]) * std

        # initialize bias 'b' as zero
        b = np.zeros((n_out.shape[0], 1))
        # define a variable to store the error calculated in each iteration

        # set number of epochs
        epochs=epochs

        # iteration
        for i in range(epochs):
            # initialize the intermediate variable
            Z = np.dot(W, n_in) + b

            # forward-propagation
            # ReLU activation
            Y_hat = np.maximum(0, Z)

            # define the sample size 'n'
            # note that the true value 'Y' is already defined in the above calculation - no need to define it again.
            n = Y_hat.shape[1]

            # calculate the mean squared error or loss function
            E = (1 / n) * np.sum(np.square(Y_hat - Y))
            # print(f"Epoch:{i} loss:{E}" ) ## uncomment this line if you want to see loss for each epoch

            # backward-propagation
            # derivative of the loss function
            dE = (2 / n) * (Y_hat - Y)

            # derivative of Relu Activation function
            dY_hat = (Z > 0)

            # input data
            dZ = n_in

            # derivative of loss function with respect to weights
            dW = np.dot((dE * dY_hat), dZ.T)

            # derivative of loss function with respect to bias
            db = np.sum((dE * dY_hat), axis=1, keepdims=True)

            # update weights
            W = W - (dW * alpha)

            # update bias
            b = b - (db * alpha)
        return W, b

    #mean squared error
    def mean_sqaured_error(self,W,b,X_test,Y_test):
        Z_predict = np.dot(W, X_test) + b

        # activation function
        Y_hat_predict = np.maximum(0, Z_predict)

        # define the size
        n = X_test.shape[1]

        # error prediction
        E_predict = (1 / n) * np.sum(np.square(Y_hat_predict - Y_test))
        return E_predict
