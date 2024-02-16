from ML_Pipeline.Preprocessing import Preprocessing
from ML_Pipeline.Neural_Network import TrainModel
import pandas as pd
import warnings
warnings.simplefilter("ignore")

#reading data
data=pd.read_excel('Input/Real_Estate_Data.xlsx')

#columns to keep
cols_to_keep=['Propert Type','Property Area in Sq. Ft.','Sub-Area','Swimming Pool','Price in Millions'] #name of columns to keep
df=Preprocessing(data).cols_to_keep(cols_to_keep)

#converting column to numeric
df=Preprocessing(df).convert_to_num('Propert Type')

#converting column to binary
df=Preprocessing(df).convert_to_binary('Swimming Pool')

#one hot encoding
df=Preprocessing(df).one_hot_encode('Sub-Area')

#removing missing values
df=Preprocessing(df).remove_missing_values()

#converting to float
cols=['Propert Type','Property Area in Sq. Ft.','Swimming Pool','Price in Millions']
df=Preprocessing(df).convert_to_float(cols)

#scaling feature
df=Preprocessing(df).min_max_scale('Property Area in Sq. Ft.')

#splitting data into train test
target_col='Price in Millions' #put name of target column here
test_size=0.3 #put test size here
X_train,X_test,Y_train,Y_test=Preprocessing(df).train_test_split(target_col,test_size)

#training model
learning_rate=0.01 #put learning rate here
epochs=100 #put epochs here
w,b=TrainModel().fit(X_train,X_test,Y_train,Y_test,learning_rate,epochs)

#calculating  mean squared error on test set
error=TrainModel().mean_sqaured_error(w,b,X_test, Y_test)
print(f'Mean squared error is: {error:.3f}')
