from keras.models import Sequential, Input
from keras.layers import Dense, Conv2D, Flatten,Concatenate, MaxPooling2D, Dropout, Conv2DTranspose
from keras import callbacks
import numpy as np

#train the cnn model by random initialized sequences
#seq_file: the name/path of the sequence data
#model_file: the name of the model file(.h5)
def sequence_cnn_t1_train(seq_file,model_file):

    #load data
    train_sequence=np.load(seq_file)
    print(train_sequence.shape)
    #number of sequences
    sample = train_sequence.shape[0]
    #length of the sequence
    step = train_sequence.shape[1]
    # nx,ny: grid size
    nx = train_sequence.shape[2]
    ny = train_sequence.shape[3]

    #add the static surface to the training data
    zero = np.zeros((1,step,nx,ny,1))
    train_sequence = np.concatenate((zero,train_sequence),axis=0)

    #divide the sequences into input/output pairs
    #where input is the eta of time t and output is the eta of time t+1
    seq_input = np.zeros((sample+1, step-1, nx, ny, 1))
    seq_output = np.zeros((sample+1, step-1, nx, ny, 1))
    for s in range(sample+1):
        for i in range(step-1):
            seq_input[s, i, :, :, :] = train_sequence[s, i, :, :, :]
            seq_output[s, i, :, :, :] = train_sequence[s, i + 1, :, :, :]
    seq_input = seq_input.reshape(((sample+1) * (step-1), nx, ny, 1))
    seq_output = seq_output.reshape(((sample+1) * (step-1), nx*ny))
    print(seq_input.shape)
    print(seq_output.shape)

    #cnn
    mycall = callbacks.EarlyStopping(patience=5, monitor='loss', mode = 'min', min_delta=0.0001)
    model = Sequential()
    model.add(Conv2D(filters=1, kernel_size=(3, 3), activation='linear'))
    model.add(Flatten())
    model.add(Dense(841, activation='tanh'))
    model.add(Dense(961, activation='linear'))
    model.compile(optimizer='Adam', loss='mae', metrics=['mse', 'mae'])
    model.fit(shuffle=True,batch_size=64, epochs=500, callbacks=[mycall], x=seq_input, y = seq_output)
    model.save(model_file)

#sequence_cnn_t1_train('random_sequence_init0.npy','random_cnn_t1_1_3_3_64.h5')
