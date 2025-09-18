import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

def fit_LSTM(reconstruct, time, n_forecasts, original_series, plot=True):
    # Function to create sequences
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    # Function to make forecasts
    def make_forecasts(model, historical_data, seq_length, n_forecasts):
        forecasts = []
        input_seq = historical_data[-seq_length:]
        for _ in range(n_forecasts):
            x = input_seq[-seq_length:]
            x = x.reshape((1, seq_length, 1))
            yhat = model.predict(x, verbose=0)
            forecasts.append(yhat[0])
            input_seq = np.append(input_seq, yhat[0])
        return forecasts

    # Define the custom log MSE loss function
    def log_mse(y_true, y_pred):
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        return tf.math.log(mse + tf.keras.backend.epsilon())  # Add epsilon to avoid log(0)

    def plot_lstm_forecasts(original_series, reconstruction, forecasts, time, n_forecasts):
        """
        Plots the original time series data, its reconstruction, and LSTM forecasts.
    
        Parameters:
        - original_series: Original time series data.
        - reconstruction: Reconstructed series after transformation or feature extraction.
        - forecasts: Forecasted values from the LSTM model.
        - time: The time series index for the original and reconstructed data.
        - n_forecasts: Number of forecast periods.
        """
        # Create forecast years starting from 1 year after the last date in `time`
        forecast_years = list(range(time.max() + 1, time.max() + n_forecasts + 1, 1))
    
        plt.figure(figsize=(12, 6))
    
        # Plot original series
        plt.plot(time, original_series, color='blue', label='Original Series', linewidth=2)
    
        # Plot reconstruction
        plt.plot(time, reconstruction, color='black', label='Reconstructed Series', linewidth=2)
    
        # Plot LSTM forecasts
        plt.plot(forecast_years, forecasts, label='LSTM Forecast', linestyle='--', marker='o', color='red')
    
        plt.title('LSTM Forecast: Original and Reconstructed Series')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    
    # Given `reconstruct` is time series data
    data = np.array(reconstruct).reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    seq_length = 3  # Number of years for the sequence
    X, y = create_sequences(data_scaled, seq_length)
    
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(40, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=log_mse)
    
    # Fit the model
    model.fit(X, y, epochs=100, verbose=0)
    
    # Make forecasts
    forecasts_scaled = make_forecasts(model, data_scaled, seq_length, n_forecasts)
    forecasts = scaler.inverse_transform(forecasts_scaled)

    if plot:
        # Plot LSTM forecasts
        plot_lstm_forecasts(original_series, reconstruct, forecasts, time, n_forecasts)

    # Evaluate the model on the training data to get the log MSE value
    log_mse_value = model.evaluate(X, y, verbose=0)
    print(f'Log Mean Squared Error: {log_mse_value}')

    return forecasts, log_mse_value


def fit_LSTM1(reconstruct, time, n_forecasts, original_series, plot=True):
    # Function to create sequences
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data)-seq_length):
            x = data[i:(i+seq_length)]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    # Function to make forecasts
    def make_forecasts(model, historical_data, seq_length, n_forecasts):
        forecasts = []
        input_seq = historical_data[-seq_length:]
        for _ in range(n_forecasts):
            x = input_seq[-seq_length:]
            x = x.reshape((1, seq_length, 1))
            yhat = model.predict(x, verbose=0)
            forecasts.append(yhat[0])
            input_seq = np.append(input_seq, yhat[0])
        return forecasts

    def plot_lstm_forecasts(original_series, reconstruction, forecasts, time, n_forecasts):
        """
        Plots the original time series data, its reconstruction, and LSTM forecasts.
    
        Parameters:
        - original_series: Original time series data.
        - reconstruction: Reconstructed series after transformation or feature extraction.
        - forecasts: Forecasted values from the LSTM model.
        - time: The time series index for the original and reconstructed data.
        - n_forecasts: Number of forecast periods.
        """
        # Create forecast years starting from 1 year after the last date in `time`
        forecast_years = list(range(time.max()+1, time.max()+n_forecasts+1, 1))
    
        plt.figure(figsize=(12, 6))
    
        # Plot original series
        plt.plot(time, original_series, color='blue', label='Original Series', linewidth=2)
    
        # Plot reconstruction
        plt.plot(time, reconstruction, color='black', label='Reconstructed Series', linewidth=2)
    
        # Plot LSTM forecasts
        plt.plot(forecast_years, forecasts, label='LSTM Forecast', linestyle='--', marker='o', color='red')
    
        plt.title('LSTM Forecast: Original and Reconstructed Series')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    
    # Given `reconstruct` is time series data
    data = np.array(reconstruct).reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    seq_length = 3  # Number of years for the sequence
    X, y = create_sequences(data_scaled, seq_length)
    
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Fit the model
    model.fit(X, y, epochs=100, verbose=0)
    
    # Make forecasts
    forecasts_scaled = make_forecasts(model, data_scaled, seq_length, n_forecasts)
    forecasts = scaler.inverse_transform(forecasts_scaled)

    if plot:
        # Plot LSTM forecasts
        plot_lstm_forecasts(original_series, reconstruct, forecasts, time, n_forecasts)

    return forecasts

def fit_arima_model(ts, max_ar=3, max_ma=3, max_d=3, plot=False):
    """
    Fits an ARIMA model to time series data and selects the best model based on AIC and BIC criteria, 
    choosing the model with fewer total parameters (AR+D+MA) if AIC and BIC criteria suggest different models.
    If the total order is the same but the models are different, it defaults to the BIC model.

    Parameters:
    - ts: The time series data as a Pandas Series.
    - max_ar: Maximum auto-regressive order to consider.
    - max_ma: Maximum moving average order to consider.
    - max_d: Maximum degree of differencing to consider.

    Returns:
    - The best ARIMA model based on the described criteria.
    """
    models_params = []
    aic_values = []
    bic_values = []

    for ar in range(1, max_ar + 1):
        for d in range(max_d + 1):
            for ma in range(1, max_ma + 1):
                try:
                    model = sm.tsa.ARIMA(ts, order=(ar, d, ma)).fit()
                    models_params.append((ar, d, ma))
                    aic_values.append(model.aic)
                    bic_values.append(model.bic)
                except Exception as e:
                    print(f"Failed to fit ARIMA({ar},{d},{ma}): {e}")
                    continue

    # Find the indices of the models with the lowest AIC and BIC
    best_aic_index = aic_values.index(min(aic_values))
    best_bic_index = bic_values.index(min(bic_values))

    # Print the models with the lowest AIC and BIC
    print(f"Model with lowest AIC: ARIMA{models_params[best_aic_index]} with AIC: {aic_values[best_aic_index]}")
    print(f"Model with lowest BIC: ARIMA{models_params[best_bic_index]} with BIC: {bic_values[best_bic_index]}")

    # Choose between AIC and BIC model based on total order, default to BIC if equal
    best_aic_order = sum(models_params[best_aic_index])
    best_bic_order = sum(models_params[best_bic_index])

    if best_aic_order < best_bic_order:
        best_model_index = best_aic_index
    else:
        # This defaults to BIC in case of a tie or BIC having fewer parameters
        best_model_index = best_bic_index

    best_params = models_params[best_model_index]
    best_model = sm.tsa.ARIMA(ts, order=best_params).fit()

    print(f"Selected model: ARIMA{best_params} with AIC: {aic_values[best_model_index]} and BIC: {bic_values[best_model_index]}")

    # Plotting original AIC and BIC values
    if plot:
        fig, ax1 = plt.subplots(figsize=(12, 6))  # Increase figure size for better readability
    
        color = 'tab:red'
        ax1.set_xlabel('Model')
        ax1.set_ylabel('AIC', color=color)
        ax1.plot(range(len(models_params)), aic_values, color=color, label='AIC', marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
    
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('BIC', color=color)  # we already handled the x-label with ax1
        ax2.plot(range(len(models_params)), bic_values, color=color, label='BIC', marker='x')
        ax2.tick_params(axis='y', labelcolor=color)
    
        # Setting x-axis labels explicitly with rotation and alignment
        ax1.set_xticks(range(len(models_params)))  # Ensure we have a tick for each model
        ax1.set_xticklabels([f'ARIMA{params}' for params in models_params], rotation=45, ha='right', fontsize=8)
    
        plt.draw()  # Explicitly redraw the figure to apply the rotation and alignment
        fig.tight_layout()  # Adjust layout to make room for the rotated x-labels
        plt.title('AIC and BIC Values for Each Model')
        plt.show()

    return best_model, best_params, aic_values[best_model_index], bic_values[best_model_index]


'''
def fit_arima_model(ts, max_ar=3, max_ma=3, max_d=3, plot = False):
    """
    Fits an ARIMA model to time series data and selects the best model based on AIC, BIC, and a combined score.

    Parameters:
    - ts: The time series data as a Pandas Series.
    - max_ar: Maximum auto-regressive order to consider.
    - max_ma: Maximum moving average order to consider.
    - max_d: Maximum degree of differencing to consider.

    Returns:
    - The best ARIMA model based on a combined normalized AIC and BIC score.

    The function iterates over specified ranges for AR, MA, and D parameters, fits ARIMA models,
    and evaluates them using AIC and BIC criteria. It prints the best models according to AIC,
    BIC, and a combined normalized score, and plots the normalized AIC and BIC values for comparison.
    """
    models_params = []
    aic_values = []
    bic_values = []

    for ar in range(1, max_ar + 1):
        for d in range(max_d + 1):
            for ma in range(max_ma + 1):
                try:
                    model = sm.tsa.ARIMA(ts, order=(ar, d, ma)).fit()
                    models_params.append((ar, d, ma))
                    aic_values.append(model.aic)
                    bic_values.append(model.bic)
                except Exception as e:
                    print(f"Failed to fit ARIMA({ar},{d},{ma}): {e}")
                    continue

    # Find the indices of the best AIC and BIC
    best_aic_index = aic_values.index(min(aic_values))
    best_bic_index = bic_values.index(min(bic_values))
    best_aic_params = models_params[best_aic_index]
    best_bic_params = models_params[best_bic_index]

    # Normalizing and combining scores
    min_aic = min(aic_values)
    min_bic = min(bic_values)
    normalized_aic = [aic / min_aic for aic in aic_values]
    normalized_bic = [bic / min_bic for bic in bic_values]
    combined_scores = [aic + bic for aic, bic in zip(normalized_aic, normalized_bic)]
    best_combined_index = combined_scores.index(min(combined_scores))
    best_combined_params = models_params[best_combined_index]

    # Fitting the best model based on combined score
    best_model = sm.tsa.ARIMA(ts, order=best_combined_params).fit()

    print(f"Best AIC model: ARIMA{best_aic_params} with AIC: {min(aic_values)}")
    print(f"Best BIC model: ARIMA{best_bic_params} with BIC: {min(bic_values)}")
    print(f"Best combined score model: ARIMA{best_combined_params} based on combined normalized score")

    # Plotting function
    def plot_model_criteria():
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = range(len(models_params))
        ax.plot(indices, normalized_aic, label='Normalized AIC', marker='o')
        ax.plot(indices, normalized_bic, label='Normalized BIC', marker='x')
        plt.xticks(indices, [f'ARIMA{params}' for params in models_params], rotation=90)
        plt.xlabel('Model')
        plt.ylabel('Normalized Criterion Value')
        plt.title('Normalized AIC and BIC Values for Each Model')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Call the plotting function
    if plot:
        plot_model_criteria()

    return best_model
'''


def plot_all_forecasts(original_rescale, forecasts, data):
    """
    Plots the original time series data, its wavelet reconstruction, and forecasts with their mean.

    Parameters:
    - original_rescale: Reconstructed series after wavelet transformation.
    - forecasts: 3D array containing forecasted values for multiple simulations and series.
    - data: Original time series data, transformed or normalized for comparison.

    This function visualizes the original time series data alongside its reconstruction and future forecasts.
    It plots the original data, the wavelet reconstructed series, and individual forecasts. Additionally,
    it calculates and plots the mean of all forecast simulations to project a potential future trend.
    """
    
    plt.figure(figsize=(12, 6))

    # Plot original normalized data
    plt.plot(data, color='blue', label='Original Transformed Series', linewidth=2)

    # Length of the original time series
    original_length = original_rescale.shape[0]

    # Plot the reconstructed original time series
    plt.plot(original_rescale, color='black', label='Reconstructed Wavelet Extraction', linewidth=2)

    # Forecast time axis starts after the original series
    forecast_time_axis = np.arange(original_length, original_length + forecasts.shape[2])

    # Plot each reconstructed forecast
    num_series = forecasts.shape[1]
    for i in range(forecasts.shape[0]):
        plt.plot(forecast_time_axis, forecasts[i,num_series-1,:], alpha=0.3)  # Adjust alpha for better visibility

    # Plot mean of simulations
    mean_sim = np.mean(forecasts[:,num_series-1,:], axis = 0)
    plt.plot(forecast_time_axis, mean_sim, color = 'red', label='Projected Mean ARMA Wavelet')
    
    plt.title('ARIMA Forecasts: Original and Reconstructed Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def reconstructed_forecasts(signal_ibc_exp, forecasted_ibc_exp, og_data, data_unit, model_type, run_name, folder, plot=False, save=False):
    """
    Plots reconstructed and forecasted simulations alongside original time series data.

    Parameters:
    - signal_ibc_exp: Array of the original time series data after signal processing.
    - forecasted_ibc_exp: 2D array of forecasted simulations for the processed signal.
    - og_data: Array of the original, unprocessed time series data.
    - save: Boolean indicating whether to save the average forecast to a CSV file.

    This function visualizes the original data, the processed signal, and multiple forecasted
    simulations with their average. Optionally saves the average forecast to "SF_Forecast.csv".
    """
    
    # Length of the original time series (signal_ibc_exp)
    original_length = len(signal_ibc_exp)

    if plot:
        plt.figure(figsize=(12, 6))

    if model_type == 'LSTM':
        df = pd.DataFrame(forecasted_ibc_exp, columns=['Forecast'])
        forecast_x_values = np.arange(original_length, original_length + forecasted_ibc_exp.shape[0])
        if plot:
            plt.plot(forecast_x_values, forecasted_ibc_exp, color='#EE6677', label='Average Forecasted Signal')
    
    else:       
        # Plot each simulation for forecasted_ibc_exp with offset
        for i in range(forecasted_ibc_exp.shape[0]):
            forecast_x_values = np.arange(original_length, original_length + forecasted_ibc_exp.shape[1])
            if plot:
                plt.plot(forecast_x_values, forecasted_ibc_exp[i, :], color='#004488', alpha=0.1)  # Low opacity
        
        # Calculate and plot the average of all simulations with offset
        avg_forecasted = np.mean(forecasted_ibc_exp, axis=0)
        if plot:
            plt.plot(forecast_x_values, avg_forecasted, color='#EE6677', label='Average Forecasted Signal', linewidth=2)
    
        df = pd.DataFrame(avg_forecasted, columns=['Forecast'])
    
    # Export average of sims if save
    if save:
        df.to_csv(f"{folder}/Signal_Forecast_{run_name}.csv", index=False)
    if plot:
        # Plot signal_ibc_exp
        plt.plot(np.arange(original_length), signal_ibc_exp, color='#228833', label='Signal with Trend', linewidth=2)
        
        # Plot original data
        plt.plot(np.arange(len(og_data)), og_data, color='#004488', label='Original Timeseries', linewidth=2)
        
        plt.title('Reconstructed and Forecasted Simulations from WARM')
        plt.xlabel('Time Periods')
        plt.ylabel(f'Discharge ({data_unit})')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.show()

    return df