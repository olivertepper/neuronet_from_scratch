import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Preprocessing:

    #initializing data
    def __init__(self,data):
        self.data=data


    #columns to keep
    def cols_to_keep(self,cols):
        self.data=self.data[cols]
        return self.data

    #convert to numerical value
    def convert_to_num(self,col):
        # extract the numerical value for the 'Propert Type'
        self.data[col] = self.data[col].str.extract('(\d+)')
        return self.data


    #binary conversion of category
    def convert_to_binary(self,col):
        # %%
        # convert the 'Swimming Pool' variable from string to binary
        self.data[col] = self.data[col].str.lower().str.strip().map(
            {'yes': True, 'no': False}
        )

        return self.data

    #dropping missing values
    def remove_missing_values(self):
        nan_rows = self.data[self.data.isnull().any(axis=1)]
        self.data=self.data.drop(nan_rows.index.values)
        return self.data

    #one hot encoding of categorical feature
    def one_hot_encode(self,col):
        area_encoded = pd.get_dummies(self.data['Sub-Area'].str.lower().str.strip(), prefix='area', dtype='float64')
        self.data = pd.concat([self.data, area_encoded], axis=1)
        self.data = self.data.drop('Sub-Area', axis = 1)
        return self.data


    #converting values to float
    def convert_to_float(self,cols):
        for i in cols:
            self.data[i]=self.data[i].astype(float,errors='raise')
        return self.data


    #min max scaler
    def min_max_scale(self,col):
        scaler = MinMaxScaler()
        self.data[col] = scaler.fit_transform(self.data[[col]])
        return self.data

    #train test split of the data
    def train_test_split(self,target_col,test_size):
        X=self.data.drop(target_col,axis=1)
        Y=self.data[target_col]
        Y = Y.values.reshape((Y.shape[0], 1))
        size=round(X.shape[0]*(1-test_size))
        X_train = X[:size].T
        Y_train = Y[:size].T
        X_test = X[size:].T
        Y_test = Y[size:].T
        return X_train,X_test,Y_train,Y_test








    


