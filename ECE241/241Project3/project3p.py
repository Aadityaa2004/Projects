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
                :return:
                """                      
                self.file = file
                self.training_df = self.load_training_df()
                self.features = self.training_df[['LotFrontage','LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'BsmtFinSF1', 'BsmtUnfSF',
                                                  'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                                                  'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                                                  'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                                                  'OpenPorchSF', 'EnclosedPorch', 'MoSold', 'Price']]
                
                #self.weights = np.random.uniform(low=0,high=1,size=26).tolist()
                self.weights = np.ones(26)
                self.prediction = self.pred()


        def load_training_df(self):
                """
                :return:
                """                      
                return pd.read_csv(self.file)  # Import the dataset.

        def first_ten_items(self):
                """
                :return:
                """                      
                return (self.training_df.head(10))
        
        def size(self):
                """
                :return:
                """                      
                size = self.training_df.size
                return 'The dataframe %s has %s entries' %(self.file, size)
        
        def mean_price(self):
                """
                :return:
                """                      
                mean_price = self.training_df['Price'].mean()
                return 'The mean of all prices is %s' %(mean_price)
        
        def min_and_max_price(self):
                """
                :return:
                """                      
                min_price = self.training_df['Price'].min()
                max_price = self.training_df['Price'].max()
                return 'The maximum price is %s and the minimum price is %s' %(max_price, min_price)
                    
        def standard_deviation(self):
                """
                :return:
                """                      
                std_dev_price = self.training_df['Price'].std()
                return 'The standard deviation is %s' %(std_dev_price)
        
        def hist_price(self):
                """
                :return:
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
                :return:
                """                      
                plot = sns.pairplot(data = self.training_df, 
                                    vars= ['GrLivArea', 'BedroomAbvGr', 'TotalBsmtSF', 'FullBath'],
                                    kind='scatter',
                                    diag_kind='auto',
                                    hue='Price'
                                    )
                plt.show()
                plt.close                

        def pred(self):
                """
                :return:
                """                
                prediction = self.features.multiply(self.weights)
                return prediction.sum(axis=1)
        
        def loss(self):
                """
                :return:
                """                
                pred = self.pred()
                loss = (pred - self.training_df['Price']).pow(2).multiply(1/self.training_df['Price'].size)
                return loss
        
        def gradient(self):
                """
                return: Graident of the Function
                """                
                pred = self.pred()
                
                Y_y = (pred - self.training_df['Price'])
                feature_vector_transposed = self.features.T
                gradient = feature_vector_transposed.multiply(Y_y).multiply(2 / self.training_df['Price'].size).sum(axis=1)

                return gradient.T.to_list()

        def update(self, alpha):
                """
                :param alpha:
                :return:
                """

                gradient = self.gradient()

                for i in range(len(self.weights)): # Iterate over the indices and values of self.weights and gradient
                    self.weights[i] = self.weights[i] - (alpha * gradient[i]) # Update each weight using the corresponding gradient value

                return self.weights                
                # gradient = self.gradient()
                # for i, j in zip(range(len(self.weights)), gradient):
                #     self.weights[i] = self.weights[i] - (alpha * j)
                
                # return self.weights

        def TrainModel(self, alpha, num_iterations=500):
                """
                :param alpha:
                :param num_iterations:
                :return: None
                """
                mse = []
                for _ in range(num_iterations):
                    self.update(alpha)
                    current_loss = self.loss().sum()  # Calculate the current total loss
                    mse.append(current_loss)
                    # Add a stopping condition (you can customize this based on your needs)
                    if current_loss < 0.1:
                        print("Stopping training. Loss is below threshold.")
                        break

                print(f"Training completed after {num_iterations} iterations.")
                print("Final weights:", self.weights)

                print(self.pred())

                return mse
        
        


        

        
        
'''
We have seen the Batch Gradient Descent. We have also seen the Stochastic Gradient Descent.
Batch Gradient Descent can be used for smoother curves. SGD can be used when the dataset is large. 
Batch Gradient Descent converges directly to minima. SGD converges faster for larger datasets. 
But, since in SGD we use only one example at a time, we cannot implement the vectorized implementation on it. 
This can slow down the computations. To tackle this problem, a mixture of Batch Gradient Descent and SGD is used.

Neither we use all the dataset all at once nor we use the single example at a time. 
We use a batch of a fixed number of training examples which is less than the actual dataset and call it a mini-batch. 
Doing this helps us achieve the advantages of both the former variants we saw. 
So, after creating the mini-batches of fixed size, we do the following steps in one epoch:

    - Pick a mini-batch
    - Feed it to Neural Network
    - Calculate the mean gradient of the mini-batch
    - Use the mean gradient we calculated in step 3 to update the weights
    - Repeat steps 1 to 4 for the mini-batches we created
    - Just like SGD, the average cost over the epochs in mini-batch gradient descent fluctuates because we are averaging a small number of examples at a time.
'''


'''def build_model(my_learning_rate):
  """Create and compile a simple linear regression model."""

  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  # Describe the topography of the model.
  # The topography of a simple linear regression model
  # is a single node in a single layer.
  model.add(tf.keras.layers.Dense(units=1,
                                  input_shape=(1,)))

  # Compile the model topography into code that TensorFlow can efficiently
  # execute. Configure training to minimize the model's mean squared error.
  model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model


def train_model(model, df, feature, label, epochs, batch_size):
  """Train the model by feeding it data."""

  # Feed the model the feature and the label.
  # The model will train for the specified number of epochs.
  history = model.fit(x=df[feature],
                      y=df[label],
                      batch_size=batch_size,
                      epochs=epochs)

  # Gather the trained model's weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch

  # Isolate the error for each epoch.
  hist = pd.DataFrame(history.history)

  # To track the progression of training, we're going to take a snapshot
  # of the model's root mean squared error at each epoch.
  rmse = hist["root_mean_squared_error"]

  return trained_weight, trained_bias, epochs, rmse

print("Defined the build_model and train_model functions.")

print("Defined the build_model and train_model functions.")

def plot_the_model(trained_weight, trained_bias, feature, label):
  """Plot the trained model against 200 random training examples."""

  # Label the axes.
  plt.xlabel(feature)
  plt.ylabel(label)

  # Create a scatter plot from 200 random points of the dataset.
  random_examples = training_df.sample(n=200)
  plt.scatter(random_examples[feature], random_examples[label])

  # Create a red line representing the model. The red line starts
  # at coordinates (x0, y0) and ends at coordinates (x1, y1).
  x0 = 0
  y0 = trained_bias
  x1 = random_examples[feature].max()
  y1 = trained_bias + (trained_weight * x1)
  plt.plot([x0, x1], [y0, y1], c='r')

  # Render the scatter plot and the red line.
  plt.show()

def plot_the_loss_curve(epochs, rmse):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.97, rmse.max()])
  plt.show()

print("Defined the plot_the_model and plot_the_loss_curve functions.")



def predict_house_values(n, feature, label):
  """Predict house values based on a feature."""

  batch = training_df[feature][10000:10000 + n]
  predicted_values = my_model.predict_on_batch(x=batch)

  print("feature   label          predicted")
  print("  value   value          value")
  print("          in thousand$   in thousand$")
  print("--------------------------------------")
  for i in range(n):
    print ("%5.0f %6.0f %15.0f" % (training_df[feature][10000 + i],
                                   training_df[label][10000 + i],
                                   predicted_values[i][0] ))
'''