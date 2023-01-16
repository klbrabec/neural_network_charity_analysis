# Neural Networking Charity Analysis 
This project has been designed to utilize neural networks and deep learning in order to determine the potential for applicants for loan funding on fulfilling the loan successfully.  

## Process 
### Initial Model Construction 

The initial dataset for this project was provided by Alphabet.  The initial pass at modeling was built around the SUCCESS variable.  All data was included, with the classification variable and ap plication type being binned to minimize potential observation errors.  Once completed, the dataset was split into test and train subsets, scaled, and then fit.  The initial neural network construction used a callback function that saved data every 5 epochs through the test data. 


```
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train_scaled[0])
nodes_hidden_layer1 = 80
nodes_hidden_layer2 = 30

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer1, activation="relu", input_dim=number_input_features))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

```
The results of the first pass using this model returned an accuracy measurement of 73.12%.   Based on this result, additonal work was done to optimize the accuracy of the model. 

### Optimization Process 
#### Optimization Attempt 1 
The first optimization trial was based on the amount the application was applying for.   Data was preprocessed in the same manner as for the initial attempt.  Reviewing the data, the overall income of the applicants, the amount that was being requested, and the type of applications submitted all were processed and 'binned' in order to reduce observation errors and further refine the data.  The Dataframe was merged together, and the 'IS SUCCESSFUL' column was dropped (this is the target value)  The data was split into train and test data and then processed through a deep learning model with 2 hidden layers.  (This also implemented a callback module that tracks the results every 5 Epochs) 

```
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train_scaled[0])
nodes_hidden_layer1 = 100
nodes_hidden_layer2 = 30

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer1, activation="relu", input_dim=number_input_features))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))


```
This model was processed and returned an overall accuracy of 72.97%.  Fairly close to the initial model's accuracy.

#### Optimization Attempt 2 
The second optimization trial was executed by adding a t hird hidden layer to the model.  This was run on the same data model as the above attempt.  

```
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train_scaled[0])
nodes_hidden_layer1 = 80
nodes_hidden_layer2 = 30
nodes_hidden_layer3 = 10

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer1, activation="relu", input_dim=number_input_features))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer2, activation="relu"))

# Third hidden layer
nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer3, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
```

This model returned an accuracy of 72.82%.  Again fairly close to the initial model's accuracy. 


#### Optimization Attempt 3 
The third optimization trial was executed by changing the activation from relu to tanh.  This model was processed against the same dataset as above.  
```
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train_scaled[0])
nodes_hidden_layer1 = 80
nodes_hidden_layer2 = 30

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer1, activation="tanh", input_dim=number_input_features))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer2, activation="tanh"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
```
This model returned an accuracy of 72.70.  

## Results Discussion 
### Data Pre Processing 
What variables are considered the target(s) for the model?
 - The 'IS SUCCESSFUL' variable would be the target for the model that is being built in this exercise.  We are attempting to determine if a borrower will be successful in repaying their loan. 

What variables are considered to be the features for your model? 
- The following variables can be considered features: 
    - Application Type 
    - Affiliation
    - Classification
    - Income
    - Asking Amount

What variables are neither targets nor features and should be removed from the input data? 
- On initial examination - the EIN and NAME fields are neither targets nor features and were removed.  
- Additional columns should be considered for removal in future iterations of this study, including Use_Case, Organization, Status, Special_considerations.  These columns were not removed initially, but could be removed in the future as they do not appear to provide additional insight into the success of a loan. 

### Compiling, Training and Evaluating the Model 
How many neurons, layers and activation functions did you select for your neural network?
- For the initial model, two layers were built.  The first layer used 80  nodes, and the second used 30.  The activation function implemented was relu. This format was used as suggested best practice for the data provided.  Note:  For this and all following models, the number of nodes selected was arbitrary, based on previous experience with building models. 
- The first optimization model used two layers.  The first layer was expanded to use 100 nodes with the second using 30.  The activation function implemented was relu.   This model was attempted in order to see if increasing the number of nodes in the first layer would improve the performance of the model. (There was no significant improvement)
- The second optimization model used three layers.  The first layer implemented 80 nodes, the second 30, and the third 10.  The activation function implemented was relu.  This model was attempted in order to see if a third layer of nodes would increase the performance of the model. (There was no significant improvement) 
- The third optimization model used two layers.  The first layer implemented 80 nodes, the second 30.  The activation function implemetned was tanh.  This model was attempted to see if modifying the activation function would improve the performance of the model. (There was no significant improvement)

Were you able to achieve the target model performance? 
- For all models built, none achieved the target performance of 75%. 

What steps did you take to try and increase model performance? 
- Please see above.  Attempts included increasing nodes in the hidden layers, increasing the number of overall layers and changing the activation function. 

## Summary 
While all attempts performed fairly well, none reached the target performance threshold of 75%.    
Initial Model - 73.12%
Optimization Model 1 - 72.97%
Optimization Model 2 - 72.82%
Optimization Model 3 - 72.70%

In order to reach the target threshold, it is suggested that the first model and second model be further reviewed and refined to remove columns that do not directly relate to the success results (as indicated above)  Additionally, changing the activation functions from relu to tanu should also be explored.  

