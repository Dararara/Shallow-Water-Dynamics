# Shallow-Water-Dynamics

## train.py
This includes the code to call all the key functions, including generating the data, loading the data, training the model, plotting the comparision of the result, doing the inverse problem and plotting the result. To people who want to run these files, just run this program is enough.

## Inverse.py
Include a function to do the inverse predict: 
</br>
Predict output(pred_out), the corresponding input(din), and one input for the inverse problem(test_in).
</br>
Return the output of inverse problem from input.

## nerual_network.py
### cnn(...)
Construct the cnn with one convolution layer, one flatten layer and one hidden dense layer.
After construction, the function will compile and train the model using given input and output, meanwhile save the model with best val_loss among all the epochs.

## plot.py
### pred_draw(...)
Draw the 3d plot for two input matrices, with the error difference plot.

## predict.py
Generate array of the predict output and true output, randomly choose some of observations.

## random_data_generator.py
Generate surface data after specific steps for training, given a randomly initialized smooth surface as the initial state.

## random_sequence_generator.py
Generate sequences of surfaces for training, given a randomly initialized smooth surface as the initial state.

##random_sequence_cnn_t1_train.py
Train CNN by random initialized sequences. The interval between input and output surface is 1 time step.
</br>
This module is not included in the train.py.
