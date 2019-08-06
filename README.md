# Shallow-Water-Dynamics
## Inverse.py
include a function to do the inverse predict, need: 
</br>
predict output(pred_out), the corresponding input(din), and one input for the inverse problem(test_in)
</br>
return the output of inverse problem from input

## nerual_network.py
### cnn(...)
the function that construct the cnn with one convolution layer, one flatten layer and one hidden dense layer
after construct, it will compile and train the model using given input and output, it will save the model with best val_loss among all the epochs

## plot.py
### pred_draw(...)
draw the 3d plot for two input matrices, with the error difference plot

##  predict.py
code to generate array of the predict output and true output, randomly choose some of observations.

## random_data_generator.py
generate data for training, initial is a random smooth surface

## train.py
file that read the input and output, split data into train set and test set, and train the model

