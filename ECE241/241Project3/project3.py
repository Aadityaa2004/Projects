import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

class Model:
    
        def __init__(self,file):

                """
                In this function, we are passing the file and calling other functions in this class.
                we also initialise the self.features from the csv file to apply it further.
                """      

                self.file = file
                self.training_df = self.load_training_df()
                self.features = self.training_df[['LotFrontage','LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'BsmtFinSF1', 'BsmtUnfSF',
                                                  'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                                                  'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                                                  'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                                                  'OpenPorchSF', 'EnclosedPorch', 'MoSold', 'Price']]
                
                self.weights = np.random.uniform(low=0,high=1,size=26).tolist()
                #self.weights = np.ones(26)
                self.prediction = self.pred()


        def load_training_df(self):

                """
                Under this 'load_training_df' function the file path is passed and this function is
                used to open and read the csv file
                """                      

                return pd.read_csv(self.file)  # Import the dataset.

        def first_ten_items(self):

                """
                This function, 'first_ten_items' returns the first 10 rows of the opened csv file
                """           

                return (self.training_df.head(10))
        
        def size(self):

                """
                The function 'size' returns the number of entries this file contains. 
                """        

                size = self.training_df.size
                return 'The dataframe %s has %s entries' %(self.file, size)
        
        def mean_price(self):

                """
                The function 'mean_price' returns the Mean of all prices from the file that has been input by the user. 
                """          

                mean_price = self.training_df['Price'].mean()
                return 'The mean of all prices is %s' %(mean_price)
        
        def min_and_max_price(self):

                """
                The function 'min_and_max_price' returns the calculated maximum and minimum value of the price. 
                """               

                min_price = self.training_df['Price'].min()
                max_price = self.training_df['Price'].max()
                return 'The maximum price is %s and the minimum price is %s' %(max_price, min_price)
                    
        def standard_deviation(self):

                """
                This function is implemented to calculate the standard deviation of the price.
                """             

                std_dev_price = self.training_df['Price'].std()
                return 'The standard deviation is %s' %(std_dev_price)
        
        def hist_price(self):

                """
                This function plots the histogram of Prices.
                """            

                plt.figure(figsize=(10,6))
                plt.hist(self.training_df['Price'],bins=30,color='skyblue')
                plt.title('Histogram of Price')
                plt.xlabel('price')
                plt.ylabel('frequency')
                plt.show()
                plt.close
                
        def pair_wise_scatter_plot(self):

                """
                Plots a pair wise scatter plot of different columns.
                Here we use GrLivArea, BedroomAbvGr, TotalBsmtSF, FullBath for the plots.
                """                   

                plot = sns.pairplot(data = self.training_df, 
                                    vars= ['GrLivArea', 'BedroomAbvGr', 'TotalBsmtSF', 'FullBath'],
                                    kind='scatter', diag_kind='auto', size=1.5)

                plt.show()
                plt.close                

        def pred(self):

                """
                Calculates the predicted price. this does so by multiplying the feature matrix with weights.
                """         

                prediction = self.features.multiply(self.weights).sum(axis=1)
                return prediction
        
        def loss(self):

                """
                Calculates the loss based on a set of predicted sale price and the correct sale price.
                """       

                prediction = self.pred()         
                loss = (prediction - self.training_df['Price']).pow(2).multiply(1/self.training_df['Price'].size)
                return loss
        
        def gradient(self):

                '''
                calculates the gradient of loss function based on the predicted price and the correct price.
                '''

                prediction = self.pred()
                Y_y = (prediction - self.training_df['Price'])
                feature_vector_transposed = self.features.T
                gradient = feature_vector_transposed.multiply(Y_y).multiply(2 / self.training_df['Price'].size).sum(axis=1)
                
                return gradient.T.to_list()

        def update(self, alpha):

                '''
                Under the 'update' function given the alpha value which is the learning rate, 
                it uses to update the weights based on the gradient.
                '''

                gradient = self.gradient()

                # Iterate over the indices and values of self.weights and gradient
                for i in range(len(self.weights)): 
                    # Update each weight using the corresponding gradient value
                    self.weights[i] = self.weights[i] - (alpha * gradient[i]) 

                return self.weights
        
        def TrainModel(self, alpha, num_iterations=500):

                """
                This function is implimented to train a model that is passes under this class.
                Number of iterations here is set to 500.
                Here first alpha (learning rate) is passed. Then we calculate the current_loss. 
                """

                mse = [] 
                for i in range(num_iterations):
                    self.update(alpha)
                    current_loss = self.loss().sum()  # Calculate the current total loss
                    current_loss_sum = current_loss.sum()
                    mse.append(current_loss)
                    # Add a stopping condition (you can customize this based on your needs)
                    if current_loss < 0.1:
                        print('Loss is below threshold: STOPPING TRAINING')
                        break
                return mse




# ========================================================================================================================        
# ========================================================================================================================


# =====================
''' TEST FUNCTIONS '''
# =====================

input = rf'/Users/aaditya/Documents/UMASS/ECE241/Project 3/test.csv'

model1 = Model(input)
model2 = Model(input)
model3 = Model(input)

# ==============================
''' INITIALIZING STATEMENTS '''
# ==============================

first_ten_items = model1.first_ten_items()
size = model1.size()
mean_price = model1.mean_price()
min_and_max_price = model1.min_and_max_price()
std_dev = model1.standard_deviation()

# ==================================================================================================

hist_price = model1.hist_price()
scatter_plot = model1.pair_wise_scatter_plot()

# ==================================================================================================


prediciton = model1.pred()
loss = model1.loss()
gradient = model1.gradient()

# ==================================================================================================

update = model1.TrainModel(0.2)
iteration_update = range(len(update))
plt.plot(iteration_update,update,label = '0.2')

print('The Final sum of MSE is:', sum(update))
plt.title(label='ECE 241')
plt.legend()
plt.show()


# ========================
''' PRINT STATEMENTS '''
# ========================
        
print("========================================================================================================")
print("========================================================================================================")

print('The first ten items are',first_ten_items)

print("========================================================================================================")
print("========================================================================================================")

print(size)

print("========================================================================================================")
print("========================================================================================================")

print(mean_price)

print("========================================================================================================")
print("========================================================================================================")

print(min_and_max_price)

print("========================================================================================================")
print("========================================================================================================")

print(std_dev)

print("========================================================================================================")
print("========================================================================================================")

print('The prediction is',prediciton)

print("========================================================================================================")
print("========================================================================================================")

print('the loss is',loss)

print("========================================================================================================")
print("========================================================================================================")

print('The prediction is',gradient)

print("========================================================================================================")
print("========================================================================================================")


x = 10**-8
y = 10**-8.5

run1 = model2.TrainModel(x)
run2 = model3.TrainModel(y)

sum_run_1 = model2.loss().sum()
sum_run_2 = model3.loss().sum()

iteration_run1 = range(len(run1))
iteration_run2 = range(len(run2))

plt.plot(iteration_run1,run1,label = x)
plt.plot(iteration_run2,run2,label = y)

print('The Final MSE is:', sum_run_1, sum_run_2)
plt.title(label='ECE 241')
plt.legend()
plt.show()


print("========================================================================================================")
print("========================================================================================================")

