# from __future__ import absolute_import, division, print_function
import numpy as np, random
import keras
from keras import ops
from keras.callbacks import Callback
keras.utils.set_random_seed(1)
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from keras.metrics import R2Score, RootMeanSquaredError, MeanSquaredLogarithmicError
from keras.layers import (
    Input, Activation, Dense, Dropout, Flatten,
    Permute, Reshape, SpatialDropout1D)
from keras.layers import TimeDistributed, LeakyReLU
from keras.layers import Conv1D, Conv2D, MaxPooling2D, AveragePooling1D, MaxPooling1D, BatchNormalization
from keras.regularizers import l1, l2
from keras.optimizers import Adam, AdamW, Adadelta, SGD, RMSprop, Adagrad, Adamax, Nadam, Ftrl, Lion
from keras.initializers import GlorotUniform, HeUniform, GlorotNormal, HeNormal
import matplotlib.pyplot as plt
import pandas as pd

def one_hot_encode(sequences):
    # horizontal one-hot encoding
    sequence_length = len(sequences[0])
    integer_type = np.int32
    integer_array = LabelEncoder().fit(np.array(('ACGTN',)).view(integer_type)).transform(
        sequences.view(integer_type)).reshape(len(sequences), sequence_length)
    one_hot_encoding = OneHotEncoder(
        sparse_output=False, categories=[[0,1,2,3,4]]*sequence_length, dtype=integer_type).fit_transform(integer_array)

    return one_hot_encoding.reshape(
            len(sequences), 1, sequence_length, 5).swapaxes(2, 3)[:, :, [0, 1, 2, 3], :]


def pad_sequence(seq, max_length):
    if len(seq) > max_length:
        #diff = len(seq) - max_length
        ## diff%2 returns 1 if odd
        #trim_length = int(diff / 2)
        #seq = seq[trim_length : -(trim_length + diff%2)]
        seq = seq[-max_length:].upper()
    else:
        #seq = seq.center(max_length, 'N')
        seq = 'N'*(max_length - len(seq)) + seq

    return seq

def process_seqs(data, seq_length):
    seqs = data.Promoter.values.astype(str)
    seqs = [s.upper() for s in seqs]
    padded_seqs = np.array([pad_sequence(s, seq_length) for s in seqs])
    X = one_hot_encode(padded_seqs)
    y = data.Activity.values
    return X, y

def add_noise(X, y):
    # Add noise to inputs
    npromoters,_,_,nseq = X.shape
    seq_idx = np.arange(nseq)
    for p in range(npromoters):
        # Random shift -20 +20 nt
        shift = np.random.randint(-10, 10)
        if shift<0:
            X[p,0,:,-shift:] = X[p,0,:,:shift]
            X[p,0,:,:-shift] = 0
        elif shift>0:
            X[p,0,:,:-shift] = X[p,0,:,shift:]
            X[p,0,:,-shift:] = 0
    return X,y

def bootstrap_uniform(X, y, N):
    ly = np.log(1 + y)
    nbins = 10

    # Compute histogram and bin edges
    hist, edges = np.histogram(ly, bins=nbins)
    print(hist)

    # Calculate bin indices for each data point
    bin_idx = np.digitize(ly, bins=edges[:-1]) - 1

    # Compute frequencies of each bin
    freq = hist[bin_idx]

    nprom,_,nt,nseq = X.shape
    Xbs = np.zeros((0,1,nt,nseq))
    ybs = np.zeros((0,))
    n = N // nbins
    for b in range(nbins):
        ybin = y[bin_idx==b]
        Xbin = X[bin_idx==b]
        len_bin = len(ybin)
        nchoice = n # - len(ybin)
        if nchoice>0 and len_bin>0:
            bootstrap_idx = np.random.choice(np.arange(len_bin), nchoice, replace=True)
            ybs = np.concatenate([ybs, ybin[bootstrap_idx]])
            Xbs = np.concatenate([Xbs, Xbin[bootstrap_idx]])

    Xbs,ybs = add_noise(Xbs, ybs)
    return Xbs,ybs

# Hyperparameters
N = 400000
lr = 1e-4
input_dropout = 0.
conv_dropout = 0.
dense_dropout = 0.5
reg = 0.

# Model architecure
seq_length = 150
num_layers = 3
num_dense = 1000
conv_width = 150
filt_width = 10

if __name__ == '__main__':
    # Load training and validation data
    col_names = ['Promoter', 'expn_med', 'expn_med_fitted', 'expn_med_fitted_scaled', 'Start', 'End', 'Name', 'Active']
    all_tss = pd.read_csv('/home/yfang/dissertation/tss_scramble_peak_expression_model_format .txt', sep='\t',
                          header=None, names=col_names)

    # Remove background from training and validation data
    bg = all_tss[all_tss.Name.str.contains('neg_control')].expn_med_fitted.mean()
    bg_std = all_tss[all_tss.Name.str.contains('neg_control')].expn_med_fitted.std()
    all_tss['Activity'] = all_tss.expn_med_fitted - bg
    all_tss = all_tss[all_tss.Activity>-1]
    all_tss = all_tss

    results = pd.DataFrame()
    results_train = pd.DataFrame()


    # Select valiidation data uniformly across target range
    nbins = 10
    ly = np.log(1 + all_tss.Activity.values)
    # Compute histogram and bin edges
    hist, edges = np.histogram(ly, bins=nbins)
    # Calculate bin indices for each data point
    bin_idx = np.digitize(ly, bins=edges[:-1]) - 1
    idx = np.arange(len(ly))
    valid_prom_seqs = []
    for b in range(nbins):
        bidx = idx[bin_idx==b]
        if len(bidx)>1000:
            prom_idx  = np.random.choice(bidx, min(int(len(bidx)*0.5), 200), replace=False)
            for p in prom_idx:
                valid_prom_seqs.append(all_tss.Promoter.values[p])
    valid_idx = all_tss.Promoter.isin(valid_prom_seqs)
    valid_data = all_tss[valid_idx]
    valid_data = valid_data.dropna()

    # Select training as remaining samples
    train_data = all_tss[~valid_idx]

    # Load test data
    test = pd.read_csv('/home/yfang/dissertation/salis.txt', sep='\t')
    test = test.dropna()

    # One hot encode input DNA sequence
    X,y = process_seqs(train_data, seq_length)
    X_train,y_train = X,y
    X_valid,y_valid = process_seqs(valid_data, seq_length)
    X_test,y_test = process_seqs(test, seq_length)

    # Augment training data
    X_train, y_train = bootstrap_uniform(X_train, y_train, N)

    # Fit for log activity + 1
    y_test = np.log(1 + y_test)
    y_train = np.log(1 + y_train)
    y_valid = np.log(1 + y_valid)

    # Construct CNN
    model = Sequential()
    model.add(Input(shape=(4, seq_length)))
    model.add(Dropout(rate=input_dropout))
    for l in range(num_layers):
        model.add(Conv1D(conv_width, filt_width, 
            activation='linear', 
                    data_format='channels_first',
                    kernel_regularizer=l2(reg),
                    padding='same',
                    kernel_initializer=GlorotNormal()))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding="valid", data_format='channels_first'))
        model.add(SpatialDropout1D(rate=conv_dropout))

    model.add(Flatten())
    for w in [num_dense]:
        model.add(Dense(w, activation='relu', 
                        use_bias=True, 
                        kernel_regularizer=l2(reg),
                        kernel_initializer=GlorotNormal()))
        model.add(Dropout(rate=dense_dropout))
    model.add(Dense(1, activation='linear', 
                use_bias=True, 
                kernel_regularizer=l2(reg),
                kernel_initializer=GlorotNormal()))

    # Train model
    opt = Adam(learning_rate=lr)
    rmse_metric = RootMeanSquaredError()
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=[rmse_metric])
    model.summary()

    es_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=5,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=3,
    )
    callbacks = [es_callback]

    history =  model.fit(X_train.reshape((len(X_train),4,seq_length)), 
            y_train.reshape((len(y_train),)), 
            validation_data=(X_valid.reshape((len(X_valid),4,seq_length)), y_valid.reshape((len(y_valid),))), 
            batch_size=128, 
            epochs=50, 
            verbose=1, 
            shuffle=True,
            callbacks=callbacks) #, sample_weight=weight) # np.exp(-y_train / y_train.mean()) / y_train.mean()) #np.exp(-y_train))

    #  Save model
    model.save_weights('model.weights.h5')
    open('model.arch.json', 'w').write(model.to_json())

    #  Predict validation data
    predictions = model.predict(X_valid.reshape((len(X_valid),4,seq_length)))
    for p in range(len(predictions)):
        true_activity = y_valid[p]
        pred = predictions[p][0]
        result = pd.DataFrame([{'Promoter':valid_data.Promoter.values[p], 'True':true_activity, 'Pred':pred}])
        results = pd.concat([results, result])
    results.to_csv('valid_results.txt', sep='\t', index=False)

    # Save training fits
    predictions = model.predict(X_train.reshape((len(X_train),4,seq_length)))
    pred = np.array([p[0] for p in predictions])
    true_activity = y_train
    result = pd.DataFrame(columns=['True', 'Pred'])
    result['True'] = true_activity
    result['Pred'] = pred
    results_train = pd.concat([results_train, result])
    results_train.to_csv('train_results.txt', sep='\t', index=False)

    # Predict validation data
    predictions = model.predict(X_test.reshape((len(X_test),4,seq_length)))
    results = pd.DataFrame()
    true_activity = y_test
    pred = np.array([p[0] for p in predictions])
    result = pd.DataFrame(columns=['True', 'Pred'])
    result['True'] = true_activity
    result['Pred'] = pred
    result['Promoter'] = test.Promoter.values
    results = pd.concat([results, result])
    results.to_csv('test_results.txt', sep='\t', index=False)

