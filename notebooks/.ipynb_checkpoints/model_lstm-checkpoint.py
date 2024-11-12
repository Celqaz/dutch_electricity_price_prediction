import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error



df_all = pd.read_pickle('../data/df_19_24_cleaned.pkl') 
print('Read data.\n')

df = df_all['2022': '2024']


df = df_all['2022': '2024']

# Feature Selection


import pandas as pd

# Define the lags you want to explore
lags = [1, 2]
# features = ['price','wind_onshore', 'wind_offshore', 'solar', 'total_load']
features = ['wind_onshore', 'wind_offshore', 'solar', 'total_load']
# features = ['wind_onshore', 'wind_offshore', 'solar', 'total_load']
# features = ['price']

# Create lagged features for each feature at specified lags
for feature in features:
    for lag in lags:
        df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
    print(f'Added lagged feature {feature}')

print('\n')

def day_flag(data):
    data['DayOfWeek'] = data.index.dayofweek              # Day of the week (0=Monday, 6=Sunday)
    # data['WeekOfYear'] = data.index.isocalendar().week    # ISO week number of the year
    # data['Day'] = data.index.day                          # Day of the month
    # data['Month'] = data.index.month                      # Month of the year
    # data['Year'] = data.index.year                        # Year
    data['PeriodOfDay'] = data.index.hour                 # Hour of the day (0-23)
    # Add cyclic transformations for HourOfDay and DayOfWeek
    data['HourOfDay_sin'] = np.sin(2 * np.pi * data['PeriodOfDay'] / 24)
    data['HourOfDay_cos'] = np.cos(2 * np.pi * data['PeriodOfDay'] / 24)
    
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
    
    # Drop original cyclic features if you want to avoid redundancy
    data.drop(['PeriodOfDay', 'DayOfWeek'], axis=1, inplace=True)
    print('Created day flag.\n')
    return data

df = day_flag(df)

# utils

def eva(y_test, y_pred):
    
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

# LSTM

df = df.dropna()
features = df.columns.tolist()
df = df[features]


train_df = df['2022': '2023']
test_df = df['2024': '2024']

print('Train Test Split.\n')

# 32.19
# 28.14
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df[features])
test_scaled = scaler.transform(test_df[features])

print('Scaled data.')

# Step 3: Create Sequences Function
def create_sequences_2(data, seq_length=24, target_column='price'):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)  # Include all features in X
        y.append(data[target_column].iloc[i+seq_length])  # Target is still the original 'price'
    return np.array(X), np.array(y)

# Convert scaled data to DataFrame for consistency in `create_sequences` function
train_scaled_df = pd.DataFrame(train_scaled, columns=features)
test_scaled_df = pd.DataFrame(test_scaled, columns=features)

# Generate sequences for LSTM model
X_train, y_train = create_sequences_2(train_scaled_df)
X_test, y_test = create_sequences_2(test_scaled_df)
print('Sequences generated.\n')

# Step 4: Build the LSTM Model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))

print('Model Compiling ... \n')
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping can be omitted if not needed; here, it’s kept with training loss monitoring.

early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Fit the model without validation data
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Step 6: Evaluate the Model on the Test Set
# Calculate the test loss
# test_loss = model.evaluate(X_test, y_test, verbose=1)
# print(f"Test Loss: {test_loss}")

print('Training finished. \n')

y_pred = model.predict(X_test)

print('Performance: \n')
eva(y_test, y_pred)
