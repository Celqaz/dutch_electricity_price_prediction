import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout, Flatten, Input, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.fft import fft, ifft


import tensorflow as tf
import numpy as np
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

#!/usr/bin/env python
# coding: utf-8

# # Data Processing

# In[14]:


import pandas as pd
import numpy as np
# model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_percentage_error


# In[13]:


# Load data
def load_data(dataset = 'df_19_24_cleaned'):
    data = pd.read_pickle(f'../data/{dataset}.pkl') 
    print(data.info())
    return data


# In[19]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
def data_scaler(data):
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    print('Data is scaled')
    return data_scaled


# In[12]:


# 1. Train-Test Split (keeping all hourly data points in the last 7 days of each month for testing)
def train_test_split_7(data):
    test_indices = data.index.to_series().groupby([data.index.year, data.index.month]).apply(lambda x: x[-24*7:])
    test_data = data.loc[test_indices]
    train_data = data.drop(test_indices)
    print(f'Shape of train_data: {train_data.shape}')
    print(f'Shape of test_data: {test_data.shape}')
    return train_data, test_data


# ## LSTM

# In[27]:


def create_sequences(data, seq_length=24, target_column='price'):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)  # Include all features in X
        y.append(data[target_column].iloc[i+seq_length])  # Target is still the original 'price'
    return np.array(X), np.array(y)

def create_sequences_2(data, seq_length=24, features = ['price']):
    # features = ['price', 'wind_energy_generation', 'solar_energy_generation', 'total_load']
    
    # Convert to numpy array for easier slicing
    data_array = data[features].values
    
    # Initialize lists for sequences and labels
    sequences = []
    labels = []
    
    # Create sequences
    for i in range(len(data_array) - seq_length):
        # Sequence of 24 time steps
        sequences.append(data_array[i:i + seq_length])
        
        # The label is the price at the next time step after the sequence
        labels.append(data_array[i + seq_length, 0])  # Assuming `price` is the first column
    
    # Convert lists to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)

    return  sequences, labels


# In[15]:


def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dropout(0.2),  # Dropout to prevent overfitting
        Dense(1)  # Output layer with a single neuron for regression
    ])
    model.compile(optimizer='adam', loss='mae')
    return model


# ## Evaluation

# In[17]:


# Define sMAPE function for evaluation
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


# In[28]:


def scaler_inverse(y_test_scaled, y_preds_scaled, X_test):
    y_test_original = scaler.inverse_transform(
        np.concatenate((y_test_scaled.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]

    y_preds_original = scaler.inverse_transform(
        np.concatenate((y_preds_scaled, X_test[:, -1, 1:]), axis=1))[:, 0]

    return y_test_original, y_preds_original
    
def scaler_inverse_2(y_test_scaled, y_preds_scaled, num_features = 1):
    # Reshape predictions and true values for inverse transformation
    y_preds_scaled = y_preds_scaled.reshape(-1, 1)
    y_test_scaled = y_test_scaled.reshape(-1, 1)
    
    # Extend with zeros for other features to match scaler's input shape
    # num_features = len(features)
    zeros = np.zeros((len(y_preds_scaled), num_features - 1))
    predictions_extended = np.concatenate([y_preds_scaled, zeros], axis=1)
    # test
    y_test_extended = np.concatenate([y_test, zeros], axis=1)
    
    # Inverse transform
    y_preds_original = scaler.inverse_transform(predictions_extended)[:, 0]  # Only take price column
    y_test_original = scaler.inverse_transform(y_test_extended)[:, 0]      

    return y_test_original, y_preds_original


# In[2]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
def eva(y_test, y_pred, X_test):
    
    # Inverse scale predictions and actual values
    y_pred_rescaled = scaler.inverse_transform(
        np.concatenate((y_pred, X_test[:, -1, 1:]), axis=1)
    )[:, 0]
    y_test_rescaled = scaler.inverse_transform(
        np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1)
    )[:, 0]
    
    # Calculate MAE
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                   
    def smape(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    
    smape_value = smape(y_test_rescaled, y_pred_rescaled)
    print(f"Symmetric Mean Absolute Percentage Error (sMAPE): {smape_value:.2f}")
    return y_test_rescaled, y_pred_rescaled


# In[9]:


def eva_s(y_test_rescaled, y_pred_rescaled):
    
    # Calculate MAE
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                   
    def smape(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    
    smape_value = smape(y_test_rescaled, y_pred_rescaled)
    print(f"Symmetric Mean Absolute Percentage Error (sMAPE): {smape_value:.2f}")

    def smape_cus(y_true, y_pred):
       return sum(2 * abs(y_true-y_pred) / ( abs(y_true) + abs(y_pred))) / 24 / 305 * 100
    
    smape_t = smape_cus(y_test_rescaled, y_pred_rescaled)

    print(f"sMAPE: {smape_t:.2f}")
    
    return mae, rmse, smape_value, smape_t
    # return y_test_rescaled, y_pred_rescaled


# In[6]:


def promo_prect(m_before, m_now, decrease = True ):
    m1 = np.array(m_before)
    m2 = np.array(m_now)
    res = (m1 - m2) / m1 * 100
    print(res)


# # EWT

# In[ ]:


import ewtpy

# Resued for https://pypi.org/project/ewtpy/ under MIT license
# source code: https://github.com/vrcarva/ewtpy/blob/master/ewtpy/ewtpy.py

def ewt_decompose(data, K, log = 0, detect = "locmax", completion = 0, reg = 'average', lengthFilter = 10,sigmaFilter = 5):
    ewt,  mfb ,boundaries = ewtpy.EWT1D(data, 
                                        N = K, 
                                        log = log, 
                                        detect = detect, 
                                        completion = completion, 
                                        reg = reg, 
                                        lengthFilter = lengthFilter,
                                        sigmaFilter = sigmaFilter)
    

    combined_signal = np.sum(ewt, axis=1)
    return combined_signal, ewt


def plot_ewt(ewt, label = None ,start =None, end = None, ): 
    n = ewt.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(12, 9))
    for i in range(n):
        axes[i].plot(ewt[start:end,i])
        # axes[i].set_title(f'{name} EWT Component {i + 1}')
    
    # Set a shared ylabel for the entire plot
    fig.text(-0.001, 0.5, label, va='center', rotation='vertical', fontsize=12)
    
    plt.tight_layout()
    plt.show()


import numpy as np
import ewtpy
import matplotlib.pyplot as plt

# T = 1000
# t = np.arange(1, T+1) / T
# f = np.cos(2 * np.pi * 0.8 * t) + 2 * np.cos(2 * np.pi * 10 * t) + 0.8 * np.cos(2 * np.pi * 100 * t)

def ewt_sureshrink_denoise(data, K, start=None, end=None, plot=False):
    # Perform EWT decomposition
    ewt, mfb, boundaries = ewtpy.EWT1D(data, N=K)
    
    # Apply SureShrink thresholding to each component
    denoised_components = np.zeros_like(ewt)
    for i in range(K):
        component = ewt[:, i]
        
        # Calculate threshold based on the component's median absolute deviation (MAD)
        sigma = np.median(np.abs(component - np.median(component))) / 0.6745
        threshold_val = sigma * np.sqrt(2 * np.log(len(component)))
        
        # Apply soft thresholding to the component
        denoised_components[:, i] = threshold(component, value=threshold_val, mode='soft')
    
    # Reconstruct the denoised signal by summing the denoised components
    denoised_signal = np.sum(denoised_components, axis=1)
    
    # Plotting each component (optional)
    if plot:
        fig, axes = plt.subplots(K + 1, 1, figsize=(12, 9))
        for i in range(K):
            axes[i].plot(denoised_components[start:end, i], label=f'Denoised EWT Component {i + 1}')
            axes[i].legend()
        axes[-1].plot(denoised_signal[start:end], label='Reconstructed Signal', color='black')
        axes[-1].legend()
        plt.tight_layout()
        plt.show()
    
    return denoised_signal, denoised_components


# ## Stat Tests

# In[1]:


from statsmodels.tsa.stattools import adfuller


# In[2]:


def adf_test(data):
    adf_test = adfuller(data, regression='c')
    print('ADF Statistic: {:.6f}\np-value: {:.6f}\n#Lags used: {}'
          .format(adf_test[0], adf_test[1], adf_test[2]))
    for key, value in adf_test[4].items():
        print('Critical Value ({}): {:.6f}'.format(key, value))


# In[1]:


def dm_test_cus(actual_lst, pred1_lst, pred2_lst, h = 1, crit="MSE", power = 2):
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_act = len(actual_lst)
        len_p1  = len(pred1_lst)
        len_p2  = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt,msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt,msg)  
        # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")  
        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True
        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            is_actual_ok = compiled_regex(str(abs(actual)))
            is_pred1_ok = compiled_regex(str(abs(pred1)))
            is_pred2_ok = compiled_regex(str(abs(pred2)))
            if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):  
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return (rt,msg)
        return (rt,msg)
    
    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    
    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    
    rt = dm_return(DM = DM_stat, p_value = p_value)
    
    return rt


# ## Plot

# In[7]:


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def plot_acf_pacf(data):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6))
    plot_acf(data, lags=50, ax=ax1)
    plot_pacf(data, lags=50, ax=ax2)
    plt.tight_layout()
    plt.show()


# In[9]:


def compare_preds(test, preds):
    fig, ax = plt.subplots(figsize = (12,6))
    # ax.plot(train['date'], train['data'], 'g-.', label='Train')
    ax.plot(test, 'b-', label='Test')
    ax.plot(preds, 'r--', label='Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Electricity Price')
    # ax.axvspan(80, 83, color='#808080', alpha=0.2)
    ax.legend(loc=2)
    
    
    fig.autofmt_xdate()
    plt.tight_layout()


# ## Scaler

# In[4]:


class MedScaler:
    def __init__(self):
        self.median = None
        self.sum_abs_dev = None

    def fit(self, data):
        # Calculate the median and sum of absolute deviations from the median
        data = np.array(data)
        self.median = np.median(data)
        self.sum_abs_dev = np.sum(np.abs(data - self.median))

    def transform(self, data):
        # Apply the custom normalization formula
        data = np.array(data)
        return (data - self.median) / self.sum_abs_dev

    def inverse_transform(self, scaled_data):
        # Reverse the transformation to get the original data
        scaled_data = np.array(scaled_data)
        return scaled_data * self.sum_abs_dev + self.median




# ====================================


df_ori = load_data('df_actual_22_24_cleaned')





SEQ = 24





features = ['price'] 
df_ewt = df_ori.loc['2021': '2024',  features]

def train_test_split_th(df, features =['price'], train_period = ['2022', '2023'], test_period = ['2024', '2024']): 
    train_start, train_end = train_period
    test_start, test_end = test_period
    df_ewt = df_ori.loc[train_start: test_end,  features]
    
    train_df = df.loc[train_start: train_end]
    test_df = df.loc[test_start: test_end]

    return train_df, test_df




train_df_ewt, test_df_ewt = train_test_split_th(df_ori, features =['price'], train_period = ['2022', '2023'], test_period =  ['2024', '2024']) 



def ewt_trans(data, level = 4, lengthFilter=10, reg='none', sigmaFilter=5):

    # return reconstructed signal and components
    sig_combined, sig_comps = ewt_decompose(data, level, lengthFilter= lengthFilter, reg = reg, sigmaFilter = sigmaFilter)
    
    eva_s(data.to_numpy(), sig_combined)
    # fig, ax= plt.subplots(figsize=(16, 8))
    # plt.plot(data.to_numpy()[:200], label = 'Actual',alpha = 0.7)
    # plt.plot(sig_combined[:200], label = 'Simulated', alpha = 0.7)
    # ax.legend(loc=2)

    return sig_comps

sig_comps = ewt_trans(df_ewt['price'], level = 6, lengthFilter= 1, reg = 'none', sigmaFilter=5)








# main 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
import numpy as np
import random

param_space = {
    'lstm_units': [32, 64, 128],
    'dropout': [0.1, 0.2, 0.3],
    'epochs': [10, 20, 30],
    'batch_size': [16, 32, 64],
}

def lstm_model_randomcv(train_data, test_data, param_space = param_space , n_iter=2, n_splits=3):
    # Normalize data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
    test_scaled = scaler.transform(test_data.reshape(-1, 1))

    def create_seq(data, seq_length=24):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length, :])  # Previous `window_size` steps across all components
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_seq(train_scaled, 24)
    X_test, y_test = create_seq(test_scaled, 24)

    # Randomly sample hyperparameters from param_space
    param_combinations = [
        {
            'lstm_units': random.choice(param_space['lstm_units']),
            'dropout': random.choice(param_space['dropout']),
            'epochs': random.choice(param_space['epochs']),
            'batch_size': random.choice(param_space['batch_size']),
        }
        for _ in range(n_iter)
    ]

    # Cross-validation setup
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for params in param_combinations:
        print(f"Testing Hyperparameters: {params}")
        cv_rmse_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            print(f"  Fold {fold + 1}/{n_splits}")
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            # Define the LSTM model
            model = Sequential([
                Input(shape=(X_train_fold.shape[1], X_train_fold.shape[2])),
                LSTM(units=params['lstm_units'], return_sequences=True),
                Dropout(params['dropout']),
                LSTM(units=params['lstm_units'] // 2, return_sequences=False),
                Dense(units=1)
            ])

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X_train_fold, y_train_fold, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

            # Validate the model
            y_val_pred = model.predict(X_val_fold)
            y_val_rescaled = scaler.inverse_transform(y_val_pred.reshape(-1, 1))
            y_val_actual = scaler.inverse_transform(y_val_fold.reshape(-1, 1))

            # Evaluate performance using sMAPE
            # smape_score = smape(y_val_actual.reshape(-1), y_val_rescaled.reshape(-1))
            # cv_smape_scores.append(smape_score)
            # Evaluate performance
            rmse = np.sqrt(np.mean((y_val_actual - y_val_rescaled) ** 2))
            cv_rmse_scores.append(rmse)

        # Average CV sMAPE for current hyperparameters
        avg_rmse = np.mean(cv_rmse_scores)
        print(f"  Average CV sMAPE: {avg_rmse:.2f}")
        results.append({'params': params, 'cv_smape': avg_rmse})

    # Find the best hyperparameters
    best_result = min(results, key=lambda x: x['cv_smape'])
    print("Best Hyperparameters:", best_result['params'])
    print("Best CV sMAPE:", best_result['cv_smape'])

    return best_result['params']


trian_len = train_df_ewt.shape[0] 
train_comps = sig_comps[:trian_len, :]
test_comps = sig_comps[trian_len:, :]





models_invetory = {
    # 'LSTM': lstm_model_ewt,
    # 'GRU' : gru_model_ewt,
    # 'CNN_R' : cnn_model_ewt_revised
    'LSTM_CROSS': lstm_model_randomcv
}

for model_name, model_func in models_invetory.items():
    print('Model:', model_name)
    # root_dir = f'ewt_models/{model_name}'
    # print(root_dir)

    # container
    models = []
    test_preds =[]
    metrics = []
    train_preds = []
    best_results = []
    for i in range(sig_comps.shape[1]):
        # for i in range(1):
        print('Training on components:',i)
        train_data = train_comps[:,i]
        test_data = test_comps[:,i]
        # model, y_test_preds, performance, trian_pred_rescaled = model_func(train_data, test_data)
        best_paras = model_func(train_data, test_data)

        # best_paras
        np.save(f"{model_name}_best_paras_{i}.npy", np.array(best_results))
        print(f'Sub {i} saved.')



















