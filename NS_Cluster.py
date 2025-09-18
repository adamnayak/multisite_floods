import numpy as np
import pandas as pd
import random
from scipy.stats import genpareto, gamma, expon, poisson
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, norm, rankdata
from scipy.optimize import minimize
#from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.neighbors import NearestNeighbors
from scipy.stats import genpareto, gamma, expon, kstest, chisquare



def simulate_storm_frequencies(sim, years, future_signal, result_df, params, KNN_sampling = True, plot_pair=False, plot_RV=False):
    """
    Simulates storm statistics for a given simulation ID and forecast data.

    Args:
        sim (int): The simulation ID.
        years (np.array): Array of years for which statistics are to be simulated.
        reconstruct_forecast (pd.DataFrame): DataFrame containing the forecast data.
        result_df (pd.DataFrame): DataFrame used for KNN bootstrap sampling.
        plot_pair (bool): Flag to determine whether to plot pairplots.
        plot_RV (bool): Flag to determine whether to plot univariate pdfs.

    Returns:
        pd.DataFrame: DataFrame containing the simulation results with storm statistics.
        pd.DataFrame: DataFrame containing the chosen distributions for each variable.
    """
    
    # Create parent distribution
    parent = pd.DataFrame({
        'sim': sim,
        'year': years,
        'signal': future_signal['signal'] #['Forecast']  # Assign signal directly
    })

    parent['Intensity'] = 0
    parent['Duration'] = 0

    n_samples = len(result_df['Intensity'].values)
    
    frequency_params = params['frequency'][0]
    signal_params = params['signal_freq'][0]
    freq_dist = params['frequency'][1]
    signal_dist = params['signal_freq'][1]
        
    if signal_dist == 'Expon':
        pass
    elif signal_dist == 'GPD':
        s_gpd_c = signal_params['gpd_c']
        s_gpd_loc = signal_params['gpd_loc']
    elif signal_dist == 'Gamma':
        s_gamma_alpha = signal_params['gamma_alpha']
        s_gamma_loc = signal_params['gamma_loc']
    else:
        _, signal_samples, _ = fit_logspline_density(result_df['Signal'], n_samples, model=signal_params, plot=False) 

    if freq_dist != 'Poisson':
        _, frequency_samples, _ = fit_logspline_density(result_df['Frequency'], n_samples, model=frequency_params, plot=False) 
    
    for index, row in parent.iterrows():
        curr_signal = parent.loc[index, 'signal']
        
        if freq_dist == 'Poisson':
            freq_scale = np.maximum(0, future_signal.loc[index, 'Scale_Freq'])
            frequency_samples = np.random.poisson(freq_scale, n_samples)
        
        if signal_dist == 'GPD':
            sig_scale = np.maximum(0, future_signal.loc[index, 'Scale_Sig'])
            signal_samples = genpareto.rvs(scale=sig_scale, c=s_gpd_c, loc=s_gpd_loc, size=n_samples)
        if signal_dist == 'Expon':
            sig_scale = np.maximum(0, future_signal.loc[index, 'Scale_Sig'])
            signal_samples = np.random.exponential(scale=sig_scale, size=n_samples)
        if signal_dist == 'Gamma':
            sig_scale = np.maximum(0, future_signal.loc[index, 'Scale_Sig'])
            signal_samples = gamma.rvs(scale=sig_scale, a=s_gamma_alpha, loc=s_gamma_loc, size=n_samples)
        
        freq_sample = {
            'Signal': signal_samples,
            'Frequency': np.maximum(0, frequency_samples)
        }
        
        freq_copula = fit_empirical_copula(result_df, freq_sample)

        if KNN_sampling:
            freq_copula = KNN_bootstrap(curr_signal, freq_copula, sample_size=n_samples)
        
        freq_sample = freq_copula.sample(1).iloc[0]
        
        parent.loc[index, 'Frequency'] = freq_sample['Frequency']
        parent.loc[index, 'unique_storm_id'] = f"{sim}_{row['year']}_0"

    return parent



def simulate_storm_statistics(sim, parent, result_df, future_signal, params, KNN_sampling = True, plot_pair=False, plot_RV=False):
    """
    Simulates storm statistics for a given simulation ID and forecast data.

    Args:
        sim (int): The simulation ID.
        years (np.array): Array of years for which statistics are to be simulated.
        reconstruct_forecast (pd.DataFrame): DataFrame containing the forecast data.
        result_df (pd.DataFrame): DataFrame used for KNN bootstrap sampling.
        plot_pair (bool): Flag to determine whether to plot pairplots.
        plot_RV (bool): Flag to determine whether to plot univariate pdfs.

    Returns:
        pd.DataFrame: DataFrame containing the simulation results with storm statistics.
        pd.DataFrame: DataFrame containing the chosen distributions for each variable.
    """    

    # Initialize Dataframe to update for frequencies greater than 0    
    filtered_parent = parent[(parent['storm_index'] > 0)]

    n_samples = len(result_df['Intensity'].values)
    
    duration_params = params['duration'][0]
    intensity_params = params['intensity'][0]
    signal_params = params['signal'][0]
    intensity_dist = params['intensity'][1]
    dur_dist = params['duration'][1]
    signal_dist = params['signal'][1]
    
    if dur_dist == 'Gamma':
        dur_gamma_alpha = duration_params['gamma_alpha']
        dur_gamma_loc = duration_params['gamma_loc']
    elif dur_dist == 'GPD':
        dur_gpd_c = duration_params['gpd_c']
        dur_gpd_loc = duration_params['gpd_loc']
    elif dur_dist == 'Expon':
        pass
    else:
        _, duration_samples, _ = fit_logspline_density(result_df['Duration'], n_samples, model=duration_params, plot=False) 
    

    if intensity_dist == 'Expon':
        pass
    elif intensity_dist == 'GPD':
        gpd_c = intensity_params['gpd_c']
        gpd_loc = intensity_params['gpd_loc']
    elif intensity_dist == 'Gamma':
        gamma_alpha = intensity_params['gamma_alpha']
        gamma_loc = intensity_params['gamma_loc']
    else:
        _, intensity_samples, _ = fit_logspline_density(result_df['Intensity'], n_samples, model=intensity_params, plot=False) 

    if signal_dist == 'Expon':
        pass
    elif signal_dist == 'GPD':
        s_gpd_c = signal_params['gpd_c']
        s_gpd_loc = signal_params['gpd_loc']
    elif signal_dist == 'Gamma':
        s_gamma_alpha = signal_params['gamma_alpha']
        s_gamma_loc = signal_params['gamma_loc']
    else:
        _, signal_samples, _ = fit_logspline_density(result_df['Signal'], n_samples, model=signal_params, plot=False) 

    for index, row in filtered_parent.iterrows():
        curr_signal = parent.loc[index, 'signal']
        year = parent.loc[index, 'year']
        
        if dur_dist == 'GPD':
            dur_scale = np.maximum(0, future_signal['Scale_Dur'][future_signal['year'] == year])
            duration_samples = genpareto.rvs(scale=dur_scale, c=dur_gpd_c, loc=dur_gpd_loc, size=n_samples)
        elif dur_dist == 'Gamma':
            dur_scale = np.maximum(0, future_signal['Scale_Dur'][future_signal['year'] == year])
            duration_samples = gamma.rvs(scale=dur_scale, a=dur_gamma_alpha, loc=dur_gamma_loc, size=n_samples)
        elif dur_dist == 'Expon':
            dur_scale = np.maximum(0, future_signal['Scale_Dur'][future_signal['year'] == year])
            duration_samples = np.random.exponential(scale=dur_scale, size=n_samples)
        
        if intensity_dist == 'GPD':
            int_scale = np.maximum(0, future_signal['Scale_Int'][future_signal['year'] == year])
            intensity_samples = genpareto.rvs(scale=int_scale, c=gpd_c, loc=gpd_loc, size=n_samples)
        elif intensity_dist == 'Expon':
            int_scale = np.maximum(0, future_signal['Scale_Int'][future_signal['year'] == year])
            intensity_samples = np.random.exponential(scale=int_scale, size=n_samples)
        elif intensity_dist == 'Gamma':
            int_scale = np.maximum(0, future_signal['Scale_Int'][future_signal['year'] == year])
            intensity_samples = gamma.rvs(scale=int_scale, a=gamma_alpha, loc=gamma_loc, size=n_samples)

        if signal_dist == 'GPD':
            sig_scale = np.maximum(0, future_signal['Scale_Sig'][future_signal['year'] == year])
            signal_samples = genpareto.rvs(scale=sig_scale, c=s_gpd_c * signal_scaling, loc=s_gpd_loc, size=n_samples)
        elif signal_dist == 'Expon':
            sig_scale = np.maximum(0, future_signal['Scale_Sig'][future_signal['year'] == year])
            signal_samples = np.random.exponential(scale=sig_scale, size=n_samples)
        elif signal_dist == 'Gamma':
            sig_scale = np.maximum(0, future_signal['Scale_Sig'][future_signal['year'] == year])
            signal_samples = gamma.rvs(scale=sig_scale, a=s_gamma_alpha, loc=s_gamma_loc, size=n_samples)
        
        univariate_sample = {
            'Signal': signal_samples,
            'Intensity': np.maximum(0, intensity_samples),
            'Duration': np.maximum(0, duration_samples),
        }
        
        emp_copula = fit_empirical_copula(result_df, univariate_sample)
    
        if plot_pair:
            FIDS_pairplot(result_df, columns=univariate_sample.keys())
            FIDS_pairplot(emp_copula, columns=univariate_sample.keys())

        if KNN_sampling:
            emp_copula = KNN_bootstrap(curr_signal, emp_copula, sample_size=n_samples)
            
        sample = emp_copula.sample(1).iloc[0]
        parent.loc[index, ['Intensity', 'Duration']] = sample[['Intensity', 'Duration']]

    return parent



def KNN_bootstrap_vector(predict, summary, sample_size=100, neighbors="all"):
    # Using NearestNeighbors to find k nearest neighbors efficiently
    k = len(summary) if neighbors == 'all' else int(np.sqrt(len(predict)))
    nbrs = NearestNeighbors(n_neighbors=k).fit(summary[['Signal']].values.reshape(-1, 1))
    distances, indices = nbrs.kneighbors(predict[['Forecast']].values.reshape(-1, 1))

    bootstrapped_samples = []
    for i in range(len(predict)):
        weights = 1 / (np.argsort(distances[i]) + 1)
        weights /= weights.sum()
        sampled_indices = np.random.choice(indices[i], size=sample_size, p=weights)
        bootstrapped_samples.append(summary.loc[sampled_indices])
    
    return bootstrapped_samples

    

def KNN_bootstrap(predict, summary, sample_size, neighbors=None):
    """
    Performs KNN-based bootstrap sampling for a single prediction value.

    Parameters:
    - predict: A single prediction value (e.g., float or int).
    - summary: DataFrame containing the summary data with a 'Signal' column.
    - sample_size: The number of bootstrap samples to generate.
    - neighbors: Number of neighbors to consider; if 'all', use all neighbors.

    Returns:
    - A DataFrame containing the bootstrapped samples.
    """
    # Using NearestNeighbors to find k nearest neighbors efficiently
    k = len(summary) if neighbors == 'all' else int(np.sqrt(len(summary)))
    nbrs = NearestNeighbors(n_neighbors=k).fit(summary[['Signal']].values.reshape(-1, 1))

    # Finding the k nearest neighbors for the single prediction value
    distances, indices = nbrs.kneighbors(np.array([[predict]]))

    # Calculate weights based on distances
    weights = 1 / (np.argsort(distances[0]) + 1)
    weights /= weights.sum()

    # Perform weighted bootstrap sampling
    sampled_indices = np.random.choice(indices[0], size=sample_size, p=weights)
    bootstrapped_samples = summary.iloc[sampled_indices]

    return bootstrapped_samples



def KNN(predict, summary):
    """
    Performs a K-Nearest Neighbors (KNN) resampling based on the signal intensity.
    
    Args:
        predict (pd.DataFrame): A DataFrame containing the 'wave' feature for which we want to find nearest neighbors.
        summary (pd.DataFrame): A DataFrame containing historical data with 'Intensity', 'Duration', and 'Frequency' features.

    Returns:
        pd.DataFrame: A DataFrame of resampled rows from the summary DataFrame with added 'year' and 'sim' columns from the predict DataFrame.
    """
    
    # Determine the number of predictions to process
    N = predict.shape[0]
    
    # Calculate the number of neighbors, k, as the square root of number of observations
    k = int(np.sqrt(summary.shape[0]))
    
    # Initialize a list to collect resampled DataFrame rows
    resampled_rows = []

    # Iterate over each prediction
    for i in range(N):
        # Extract the 'wave' value for the current prediction
        signal_value = predict.iloc[i]["signal"]
        
        # Find the indices of the k nearest neighbors in 'summary' based on the absolute difference in 'Intensity'
        kNN_ind = np.argsort(np.abs(signal_value - summary["signal"]))[:k]

        # Calculate weights for each of the k neighbors inversely proportional to their rank (1 being closest)
        W = [(1/x) / np.sum(1 / np.arange(1, k+1)) for x in np.arange(1, k+1)]
        
        # Ensure the weights sum to 1.0
        assert np.isclose(np.sum(W), 1.0), 'weights should sum to 1'

        # Calculate the cumulative sum of weights to facilitate weighted random sampling
        cumW = np.cumsum(W)
        
        # Generate a random number and use it to select a neighbor based on the weighted distribution
        rnd = np.random.rand()
        sampled_index = next(x for x, val in enumerate(cumW) if val > rnd)
        
        # Get the index of the selected sample
        samp_ind = kNN_ind.iloc[sampled_index]

        # Extract the 'year' and 'sim' values from the current prediction
        year = predict.iloc[i]["year"]
        sim = predict.iloc[i]["sim"]

        # Create a new row from the selected sample with additional 'year', 'sim', and original 'wave' values
        resampled_row = summary.iloc[[samp_ind]][["signal", "mean_intensity", "mean_duration", "mean_frequency"]].copy()
        resampled_row.rename(columns={'signal': 'Scale_Sig', 'mean_intensity': 'Scale_Int', 'mean_duration': 'Scale_Dur', 'mean_frequency': 'Scale_Freq'}, inplace=True)
        resampled_row["year"] = year
        resampled_row["sim"] = sim
        resampled_row["signal"] = signal_value
        
        # Append the new row to the list of resampled rows
        resampled_rows.append(resampled_row)

    # Concatenate all resampled rows into a single DataFrame
    resampled_summary = pd.concat(resampled_rows, ignore_index=True)
    
    # Return the resampled DataFrame
    return resampled_summary



def KNN_MLE(predict, summary):
    """
    Performs a K-Nearest Neighbors (KNN) analysis using maximum likelihood estimates
    based on the signal intensity, duration, and frequency.

    Args:
        predict (pd.DataFrame): A DataFrame containing the 'wave' feature for which we want to find nearest neighbors.
        summary (pd.DataFrame): A DataFrame containing historical data with 'Intensity', 'Duration', and 'Frequency' features.

    Returns:
        pd.DataFrame: A DataFrame with estimated values for 'Intensity', 'Duration', and 'Frequency' for each prediction,
                      along with the 'year' and 'sim' columns from the predict DataFrame.
    """
    
    N = predict.shape[0]
    k = int(np.sqrt(summary.shape[0]))
    resampled_rows = []

    for i in range(N):
        signal_value = predict.iloc[i]["signal"]
        kNN_ind = np.argsort(np.abs(signal_value - summary["signal"]))[:k]
        W = [(1/x) / np.sum(1 / np.arange(1, k+1)) for x in np.arange(1, k+1)]
        
        assert np.isclose(np.sum(W), 1.0), 'weights should sum to 1'

        # Calculate weighted averages for 'Intensity', 'Duration', 'Frequency' instead of sampling
        signal_estimate = np.dot(W, summary.iloc[kNN_ind]["signal"])
        intensity_estimate = np.dot(W, summary.iloc[kNN_ind]["mean_intensity"])
        duration_estimate = np.dot(W, summary.iloc[kNN_ind]["mean_duration"])
        frequency_estimate = np.dot(W, summary.iloc[kNN_ind]["mean_frequency"])
        
        year = predict.iloc[i]["year"]
        sim = predict.iloc[i]["sim"]

        # Create a new row with the estimated values and original 'year', 'sim', and 'wave' values
        estimated_row = pd.DataFrame({
            "year": [year],
            "sim": [sim],
            "signal": [signal_value],
            "Scale_Sig": [signal_estimate],
            "Scale_Int": [intensity_estimate],
            "Scale_Dur": [duration_estimate],
            "Scale_Freq": [frequency_estimate]
        })
        
        resampled_rows.append(estimated_row)

    resampled_summary = pd.concat(resampled_rows, ignore_index=True)
    
    return resampled_summary



def expand_rows_based_on_frequency(parent, sim):
    """
    Expands each row of the DataFrame based on the 'Frequency' column and assigns unique storm IDs.

    Args:
        parent (pd.DataFrame): DataFrame containing storm data with a 'Frequency' column.
        sim (int): Simulation identifier used for setting unique storm IDs.

    Returns:
        pd.DataFrame: Expanded DataFrame with unique storm IDs that reset for each year.
    """
    # Ensure the DataFrame is sorted by year for consistent processing
    parent = parent.sort_values(by='year')

    # Repeat each row according to the 'Frequency'
    child = parent.loc[parent.index.repeat(parent['Frequency'])].reset_index(drop=True)

    # Generate a unique storm ID for each row, resetting count each year
    child['storm_index'] = child.groupby('year').cumcount() + 1  # Starts counting from 1

    # Create unique identifiers
    child['unique_storm_id'] = child.apply(
        lambda x: f"{sim}_{x['year']}_{x['storm_index']}" if x['Frequency'] > 0 else f"{sim}_{x['year']}_0", axis=1
    )

    # Retain original rows for years with 0 frequency but adjust their IDs
    zero_freq_rows = parent[parent['Frequency'] == 0]
    zero_freq_rows['storm_index'] = 0  # Set storm index to 0 for years with no storms
    zero_freq_rows['unique_storm_id'] = zero_freq_rows.apply(lambda x: f"{sim}_{x['year']}_0", axis=1)

    # Combine the rows with zero frequency with the expanded rows
    final_df = pd.concat([child, zero_freq_rows], ignore_index=True).sort_values(by='year')

    return final_df



def storm_trajectories(parent, bootstrap_curve, cluster_dict_std, plot=False):
    """
    Plots intensity trajectories for each unique storm based on their duration and intensities.

    Args:
        parent (pd.DataFrame): DataFrame containing storm data.
        bootstrap_curve (function): Function that generates intensity curves given standard deviation dictionary, duration, and peak intensity.
        cluster_dict_std (dict): Dictionary containing standard deviations for intensity curve generation.

    Returns:
        Updated dataframe with trajectories
    """
    # Filter out rows where 'Frequency', 'Duration', and 'Intensity' are all zero
    nonzero_mask = (parent[['Frequency', 'Duration', 'Intensity']].sum(axis=1) != 0)
    nonzero_parent = parent[nonzero_mask]

    # Repeat each row in DataFrame according to the storm duration
    series = nonzero_parent.loc[nonzero_parent.index.repeat(nonzero_parent['Duration'])].reset_index(drop=True)
    series['storm_day'] = series.groupby('unique_storm_id').cumcount() + 1
    series['daily_flow'] = None  # Initialize the column for daily flow

    # Iterate through each unique storm
    for index_storm in series['unique_storm_id'].unique():
        storm_data = series[series['unique_storm_id'] == index_storm]
        duration = storm_data['Duration'].iloc[0]
        peak_intensity = storm_data['Intensity'].iloc[0]

        # Generate the intensity curve
        intensities, no_error = bootstrap_curve(cluster_dict_std, duration, peak_intensity)
        if no_error == 1:
            print("Spline fitting error for trajectory")
            continue  # Skip if there's an error in spline fitting

        if len(intensities) != len(storm_data):
            continue  # Skip if lengths do not match

        # Assign generated intensities to the 'daily_flow' column for the current storm
        series.loc[storm_data.index, 'daily_flow'] = intensities

    # Add back the rows with zeros in 'Frequency', 'Duration', and 'Intensity'
    zero_rows = parent[~nonzero_mask].copy()
    zero_rows['daily_flow'] = 0
    zero_rows['storm_day'] = 0

    # Combine the processed rows with the zero rows
    combined_series = pd.concat([series, zero_rows], ignore_index=True).sort_index()

    if plot:
        # Plot each storm's intensity trajectory
        fig, ax = plt.subplots()
        for index_storm in series['unique_storm_id'].unique():
            storm_data = series[series['unique_storm_id'] == index_storm]
            ax.plot(storm_data['storm_day'], storm_data['daily_flow'], label=f'Storm {index_storm}')
    
        ax.set_xlabel('Storm Day')
        ax.set_ylabel('Intensity (Daily Flow)')
        ax.set_title('Storm Intensity Trajectories')
        plt.show()

    return combined_series



def fit_and_predict_spline(values, n_gen):
    """
    Fit a cubic spline to the given series and generate new data points based on new input length.
    
    Parameters:
    - values: A 1D array-like object containing the series values.
    - n_gen: The number of data points to generate.
    
    Returns:
    - new_values: A 1D array of the generated future data points.
    - no_error: Indicator if the process was successful (0) or not (1).
    """
    n = len(values)
    no_error = 0
    
    # Use the index as the time value
    t = np.arange(n)

    # Ensure n_gen is an integer
    n_gen = int(n_gen)
    
    try:
        # Fit the cubic spline
        spline = CubicSpline(t, values, bc_type='natural')
        
        # Determine the range for future points
        new_indices = np.linspace(0, n-1, n_gen)
        
        # Generate data points
        new_values = spline(new_indices)
        
    except Exception as e:
        print("Error fitting spline:", e)
        new_values = None
        no_error = 1
    
    return new_values, no_error



def find_closest_number(target, numbers):
    """
    Finds and returns the number closest to a target value from a list of numbers, except 1. 
    If the closest number is 1, it returns the second closest number.

    Parameters:
    - target: The target number to compare against.
    - numbers: A list of numbers to search through.

    Returns:
    - The number from the list that is closest to the target value, excluding 1 as the closest.
    """
    
    # Sort the numbers based on their absolute difference from the target, ignoring 1 if it's the closest
    sorted_numbers = sorted(numbers, key=lambda x: (abs(x - target), x == 1))
    
    # Find and return the first number in the sorted list that is not 1, 
    # which will either be the closest number to the target if the closest isn't 1,
    # or the second closest if 1 is the nearest.
    for number in sorted_numbers:
        if number != 1:
            return number
            


def bootstrap_curve(dict, duration, intensity):
    """
    Selects a random series from a dictionary based on the closest duration and scales it by intensity and specified duration.

    Parameters:
    - data: Dictionary with durations as keys and lists of series as values.
    - duration: Target duration to match or find the closest one in the dictionary keys.
    - intensity: Scaling factor to apply to the selected series.

    Returns:
    - A series from the dictionary corresponding to the closest duration, scaled by the given intensity.

    This function first finds the closest duration key to the given duration. Then, it randomly selects
    a series associated with this duration from the dictionary and scales it by the specified intensity.
    """
    no_error = 0
    keys = list(dict.keys())
    if duration not in keys:
        n_dur = find_closest_number(duration, keys)
        rand = random.randint(0, len(dict[n_dur])-1)
        series, no_error = fit_and_predict_spline(dict[n_dur][rand], duration) # if duration not in keys, fit to scale based on closest duration
        if no_error == 1:
            print("Spline fit failed. Proceeding with nearest duration bootstrap as proxy...")
            series = dict[n_dur][rand]*intensity
        else:
            series = series*intensity
    else:
        rand = random.randint(0, len(dict[duration])-1)
        series = dict[duration][rand]*intensity

    return series, no_error


def index_and_count_clusters(df, var_interest, base_signal, name, cluster_len=1):
    """
    Indexes and counts clusters of consecutive exceedances within a DataFrame.

    Parameters:
    - df: DataFrame containing the time series data with timestamps as index.
    - name: The column name in df indicating exceedance (1 for exceedance, 0 otherwise).
    - cluster_len: Maximum number of days allowed between exceedances to belong to the same cluster.

    Returns:
    - A tuple containing the DataFrame with an added 'Cluster_Index' column indicating each 
      exceedance's cluster membership, and a list of cluster sizes.

    The function iterates over the DataFrame, identifying clusters of exceedances based on the 
    specified day gap (cluster_len). It assigns a unique index to each cluster and counts the 
    size (number of exceedances) of each cluster.
    """
    
    cluster_idx = 0  # Initialize cluster indexing at 0
    prev_timestamp = pd.Timestamp.min  # Use a very early timestamp to ensure any valid timestamp will be after this
    cluster_sizes = []  # List to keep track of individual cluster sizes for direct mapping later
    last_exceedance_timestamp = pd.Timestamp.min  # Track the timestamp of the last exceedance

    df['Cluster_Index'] = 0  # Initialize cluster index column
    current_cluster_size = 0  # Track the size of the current cluster

    # Map annual signals to daily data based on the year
    df['Year'] = df.index.year
    ann_dict = df.set_index('Year')['Ann_Signal'].to_dict()
    df['Ann_Signal'] = df['Year'].map(ann_dict)
    
    # Determine if daily values exceed the annual values
    df['Exceeds_Wave'] = (df[var_interest] > df['Ann_Signal']).astype(int)      
    df['Exceeds_Std'] = (df[var_interest] > base_signal).astype(int)  # Boolean attribute for standard exceedances above base signal
        
    # Calculate the difference only when exceedance occurs; otherwise, set to 0
    df['Exceedance_Diff_Wave'] = np.where(df['Exceeds_Wave'] == 1, df[var_interest] , df['Ann_Signal'])
    df['Exceedance_Diff_Std'] = np.where(df['Exceeds_Std'] == 1, df[var_interest], base_signal)
    
    # Identify changes in exceedance state
    df['Exceeds_Change_Wave'] = df['Exceeds_Wave'].diff()
    df['Exceeds_Change_Std'] = df['Exceeds_Std'].diff()
    
    # Identify the start of a cluster
    df['Start_Cluster_Wave'] = (df['Exceeds_Change_Wave'] == 1)
    df['Start_Cluster_Std'] = (df['Exceeds_Change_Std'] == 1)
    
    # Identify the end of a cluster
    df['End_Cluster_Wave'] = (df['Exceeds_Change_Wave'] == -1)
    df['End_Cluster_Std'] = (df['Exceeds_Change_Std'] == -1)

    for idx, row in df.iterrows():
        current_timestamp = idx  # Assuming idx is a Timestamp since df.iterrows() was used

        # Initialize day_difference with a default value
        day_difference = float('inf')  # Assume a large day difference by default
    
        if last_exceedance_timestamp != pd.Timestamp.min:
            # Calculate the difference in days between the current day and the last exceedance
            # Only if the last exceedance timestamp is not the minimum timestamp
            day_difference = (current_timestamp - last_exceedance_timestamp).days
            # Cap the day difference at 365 days
            day_difference = min(day_difference, 365)
    
        if row['Exceeds_Std'] == 1:
            if day_difference > cluster_len:
                # Start a new cluster if day difference is more than cluster_len
                # or if this is the first exceedance being considered
                if current_cluster_size > 0:
                    # Finalize the current cluster before starting a new one
                    cluster_sizes.append(current_cluster_size)
                cluster_idx += 1
                current_cluster_size = 1  # Start new cluster size counting
            else:
                # Current exceedance is within cluster_len days of the last exceedance
                current_cluster_size += 1
    
            # Update the cluster index and the last exceedance timestamp
            df.at[idx, 'Cluster_Index'] = cluster_idx
            last_exceedance_timestamp = current_timestamp
        else:
            # Non-exceedance days do not immediately finalize a cluster,
            # but we don't update last_exceedance_timestamp here
            continue

    # Handle the last cluster's size at the end of the DataFrame
    if current_cluster_size > 0:
        cluster_sizes.append(current_cluster_size)

    # Create a summary DataFrame for standard threshold
    summary_std = df.groupby(df.index.year).agg(
        Total_Exceedances=('Exceeds_Std', 'sum'),
        Intensity = ('Ann_Signal','mean'),
        Frequency=('Cluster_Index', lambda x: len(set(x[x > 0]))),  # Only consider non-zero indices
    )
    
    summary_std['Duration'] = np.where(summary_std['Frequency'] == 0, 0, summary_std['Total_Exceedances'] / summary_std['Frequency'])

    return df, summary_std, cluster_sizes

def uni_adapted_NS_Process(n_simulations, 
                       years,
                       forecasted_agg_ibc,
                       reconstruct_forecast,
                       summary_std,
                       cluster_dict_std,
                       folder,
                       run_name,
                       base_signal,
                       percentile_50,
                       forecast_model,
                       intensity_dist,
                       dur_dist,
                       KNN_Type='default',
                       wave_for='all',
                       freq_size=1,
                       dur_size=1,
                       int_size=1,
                       steps=20,
                       intensity_params = None,
                       duration_params = None):
    
    if dur_dist == 'Gamma':
        dur_gamma_alpha = duration_params['gamma_alpha']
        dur_gamma_loc = duration_params['gamma_loc']
        dur_gamma_beta = duration_params['scale']
        med_dur = duration_params['med_dur']
    if dur_dist == 'GPD':
        dur_gpd_c = duration_params['gpd_c']
        dur_gpd_loc = duration_params['gpd_loc']
        dur_gpd_scale = duration_params['scale']
        med_dur = duration_params['med_dur']
    
    if intensity_dist == 'Gamma':
        gamma_alpha = intensity_params['gamma_alpha']
        gamma_loc = intensity_params['gamma_loc']
        gamma_beta = intensity_params['scale']
    if intensity_dist == 'GPD':
        gpd_c = intensity_params['gpd_c']
        gpd_loc = intensity_params['gpd_loc']
        gpd_scale = intensity_params['scale']
    
    all_data = []
    for sim in range(n_simulations):
        # Create parent distribution
        parent = pd.DataFrame({'sim': sim, 'year': years})
        
        # Pull wavelet threshold from wavelet
        if wave_for != 'mean' and forecast_model != 'LSTM':
            parent['wave'] = forecasted_agg_ibc[sim]
        else:
            parent['wave'] = reconstruct_forecast['Forecast'].values[:steps]

        if KNN_Type == 'MLE':
            parent = uni_KNN_MLE(parent, summary_std)
        else:
            parent = uni_KNN(parent, summary_std)
        
        # Sample from the Poisson distribution for how many storms in a year
        parent['no_storms'] = [np.round(np.mean(np.random.poisson(freq, freq_size))) for freq in parent['Frequency']]
        
        # Repeat each row in df according to the number of storms
        child = parent.loc[parent.index.repeat(parent['no_storms'])].reset_index(drop=True)
        
        # Create 'index_storms' to indicate the index of each storm
        child['index_storms'] = child.groupby(child.columns.drop('index_storms', errors='ignore').tolist()).cumcount() + 1
        
        # Create a unique identifier by combining year and storm index
        child['unique_storm_id'] = child['year'].astype(str) + "_" + child['index_storms'].astype(str)
    
        if len(child) > 0:
            # Calculate storm durations
            for i in range(len(child)):
                if dur_dist == 'GPD':
                    child.at[i,'storm_max_intensity'] = int(np.round(np.mean(genpareto.rvs(c=dur_gpd_c, loc=dur_gpd_loc, scale=child['Duration'][i]/med_dur*dur_gpd_scale, size=dur_size))))
                if dur_dist == 'Gamma':
                    shape_value = max(dur_gamma_alpha*child['Duration'][i]/med_dur, 0.1)  # Ensuring a minimum positive value

                    child.at[i, 'duration'] = int(np.round(np.mean(gamma.rvs(a=shape_value, loc=dur_gamma_loc, scale=dur_gamma_beta, size=dur_size))))
                else:
                    child.at[i, 'duration'] = int(np.round(np.mean(np.random.exponential(scale=child['Duration'][i], size=dur_size))))
            
                if intensity_dist == 'GPD':
                    # Sample from truncated genpareto distribution (at 50% flow) for moving wavelet threshold for max intensity of year
                    child.at[i,'storm_max_intensity'] = max(np.mean(genpareto.rvs(c=gpd_c, loc=gpd_loc, scale=child['Intensity'][i]/base_signal*gpd_scale, size=int_size)), percentile_50)
                if intensity_dist == 'Gamma':
                    # Sample from truncated gamma distribution (at 50% flow) for moving wavelet threshold for max intensity of year
                    child.at[i,'storm_max_intensity'] = max(np.mean(gamma.rvs(a=gamma_alpha*child['Intensity'][i]/base_signal, loc=gamma_loc, scale=gamma_beta, size=int_size)), percentile_50)
                else:
                    # Sample from truncated exponential distribution (at 50% flow) for moving wavelet threshold for max intensity of year
                    child.at[i,'storm_max_intensity'] = max(np.mean(np.random.exponential(scale=child['Intensity'][i], size=int_size)), percentile_50)
                
            # Repeat each row in df according to the storm duration
            series = child.loc[child.index.repeat(child['duration'])].reset_index(drop=True)
            
            # Create 'index_storms' to indicate the index of each storm
            series['storm_day'] = series.groupby(series.columns.drop('storm_day', errors='ignore').tolist()).cumcount() + 1
            
            # New column for intensities
            series['intensity'] = None  # Initialize the column
            
            # Iterate through each unique storm
            for index_storm in series['unique_storm_id'].unique():
                # Filter the DataFrame for the current storm
                storm_data = series[series['unique_storm_id'] == index_storm]
                # Get the duration, and peak intensity for the current storm
                duration = storm_data['duration'].iloc[0]
                peak_intensity = storm_data['storm_max_intensity'].iloc[0]
                # Generate the intensities
                intensities, no_error = bootstrap_curve(cluster_dict_std, duration, peak_intensity)
                
                # If spline fit fails, raise error and use nearest neighbor duration
                if no_error == 1:                    
                    # Identify the target rows in 'series'
                    target_rows = series['unique_storm_id'] == index_storm
                    # Length of the segment in 'series' that matches 'index_storm'
                    length_series_segment = series.loc[target_rows, 'intensity'].shape[0]
                    # Find length of 'intensities'
                    required_length = len(intensities)
                    
                    # Calculate the number of rows to drop
                    num_rows_to_drop = length_series_segment - required_length
                    
                    # Find the indices of the rows to keep
                    indices_to_keep = series.loc[target_rows].index[:-num_rows_to_drop]
                    
                    # Update 'series' to only include the rows we want to keep
                    series = series.loc[indices_to_keep.union(series.loc[~target_rows].index)]
                
                # Assign the generated intensities to the 'intensity' column for the current storm
                series.loc[series['unique_storm_id'] == index_storm, 'intensity'] = intensities      
        
            # Append the series dataframe to the list
            all_data.append(series)

    # Concatenate all DataFrames in the list at once
    all_sims = pd.concat(all_data, ignore_index=True)

    # Convert year and sim columns to integers
    all_sims = all_sims.astype({'sim': 'int', 'year': 'int'})
    all_sims = all_sims[['sim', 'year', 'index_storms', 'unique_storm_id', 'wave', 'Frequency', 'Intensity', 'Duration', 'no_storms', 'storm_max_intensity', 'duration', 'storm_day', 'intensity']]  # Reorder columns
    rename_dict = {
    'wave': 'forecasted_signal',
    'Frequency': 'lambda_F',
    'Intensity': 'gamma_I',
    'Duration': 'alpha_D',
    'duration': 'storm_duration'
    }
    all_sims = all_sims.rename(columns=rename_dict)

    # Save to CSV
    all_sims.to_csv(f"{folder}/All_Sims_{run_name}.csv", index=False)

    return all_sims, series


def uni_KNN(predict, summary):
    """
    Performs a K-Nearest Neighbors (KNN) resampling based on the signal intensity.
    
    Args:
        predict (pd.DataFrame): A DataFrame containing the 'wave' feature for which we want to find nearest neighbors.
        summary (pd.DataFrame): A DataFrame containing historical data with 'Intensity', 'Duration', and 'Frequency' features.

    Returns:
        pd.DataFrame: A DataFrame of resampled rows from the summary DataFrame with added 'year' and 'sim' columns from the predict DataFrame.
    """
    
    # Determine the number of predictions to process
    N = predict.shape[0]
    
    # Calculate the number of neighbors, k, as the square root of N
    k = int(np.sqrt(N))
    
    # Initialize a list to collect resampled DataFrame rows
    resampled_rows = []

    # Iterate over each prediction
    for i in range(N):
        # Extract the 'wave' value for the current prediction
        intensity_value = predict.iloc[i]["wave"]
        
        # Find the indices of the k nearest neighbors in 'summary' based on the absolute difference in 'Intensity'
        kNN_ind = np.argsort(np.abs(intensity_value - summary["Intensity"]))[:k]

        # Calculate weights for each of the k neighbors inversely proportional to their rank (1 being closest)
        W = [(1/x) / np.sum(1 / np.arange(1, k+1)) for x in np.arange(1, k+1)]
        
        # Ensure the weights sum to 1.0
        assert np.isclose(np.sum(W), 1.0), 'weights should sum to 1'

        # Calculate the cumulative sum of weights to facilitate weighted random sampling
        cumW = np.cumsum(W)
        
        # Generate a random number and use it to select a neighbor based on the weighted distribution
        rnd = np.random.rand()
        sampled_index = next(x for x, val in enumerate(cumW) if val > rnd)
        
        # Get the index of the selected sample
        samp_ind = kNN_ind.iloc[sampled_index]

        # Extract the 'year' and 'sim' values from the current prediction
        year = predict.iloc[i]["year"]
        sim = predict.iloc[i]["sim"]

        # Create a new row from the selected sample with additional 'year', 'sim', and original 'wave' values
        resampled_row = summary.iloc[[samp_ind]][["Intensity", "Duration", "Frequency"]].copy()
        resampled_row["year"] = year
        resampled_row["sim"] = sim
        resampled_row["wave"] = intensity_value
        
        # Append the new row to the list of resampled rows
        resampled_rows.append(resampled_row)

    # Concatenate all resampled rows into a single DataFrame
    resampled_summary = pd.concat(resampled_rows, ignore_index=True)
    
    # Return the resampled DataFrame
    return resampled_summary


def uni_KNN_MLE(predict, summary):
    """
    Performs a K-Nearest Neighbors (KNN) analysis using maximum likelihood estimates
    based on the signal intensity, duration, and frequency.

    Args:
        predict (pd.DataFrame): A DataFrame containing the 'wave' feature for which we want to find nearest neighbors.
        summary (pd.DataFrame): A DataFrame containing historical data with 'Intensity', 'Duration', and 'Frequency' features.

    Returns:
        pd.DataFrame: A DataFrame with estimated values for 'Intensity', 'Duration', and 'Frequency' for each prediction,
                      along with the 'year' and 'sim' columns from the predict DataFrame.
    """
    
    N = predict.shape[0]
    k = int(np.sqrt(N))
    resampled_rows = []

    for i in range(N):
        intensity_value = predict.iloc[i]["wave"]
        kNN_ind = np.argsort(np.abs(intensity_value - summary["Intensity"]))[:k]
        W = [(1/x) / np.sum(1 / np.arange(1, k+1)) for x in np.arange(1, k+1)]
        
        assert np.isclose(np.sum(W), 1.0), 'weights should sum to 1'

        # Calculate weighted averages for 'Intensity', 'Duration', 'Frequency' instead of sampling
        intensity_estimate = np.dot(W, summary.iloc[kNN_ind]["Intensity"])
        duration_estimate = np.dot(W, summary.iloc[kNN_ind]["Duration"])
        frequency_estimate = np.dot(W, summary.iloc[kNN_ind]["Frequency"])
        
        year = predict.iloc[i]["year"]
        sim = predict.iloc[i]["sim"]

        # Create a new row with the estimated values and original 'year', 'sim', and 'wave' values
        estimated_row = pd.DataFrame({
            "Intensity": [intensity_estimate],
            "Duration": [duration_estimate],
            "Frequency": [frequency_estimate],
            "year": [year],
            "sim": [sim],
            "wave": [intensity_value]
        })
        
        resampled_rows.append(estimated_row)

    resampled_summary = pd.concat(resampled_rows, ignore_index=True)
    
    return resampled_summary


def extract_exceedance_clusters(df, signal, var_interest, base_signal):
    # Add year and signal from annual dictionary
    df['Year'] = df.index.year
    ann_dict = signal.groupby('Year')['Ann_Signal'].median().to_dict()
    df['Ann_Signal'] = df['Year'].map(ann_dict)

    # Compute the annual maximum value for the variable of interest
    annual_max = df.groupby('Year')[var_interest].max().to_dict()

    # Determine if values exceed the baseline
    df['Exceeds_Std'] = (df[var_interest] > base_signal).astype(int)
    
    # Calculate exceedance difference
    df['Exceedance_Diff_Std'] = np.where(df['Exceeds_Std'] == 1, df[var_interest], 0)
    
    # Identify changes in exceedance state
    df['Exceeds_Change_Std'] = df['Exceeds_Std'].diff()
    
    # Identify the start and end of a cluster
    df['Start_Cluster_Std'] = (df['Exceeds_Change_Std'] == 1)
    df['End_Cluster_Std'] = (df['Exceeds_Change_Std'] == -1)

    # Initialize the result dataframe
    result = [] # storm level exceedances

    # Iterate over each year
    for year, group in df.groupby('Year'):
        # Detect clusters within the year
        group['cluster'] = (group['Start_Cluster_Std']).cumsum()
        clusters = group[group['Exceeds_Std'] == 1]
        
        # Summarize data for each cluster
        if not clusters.empty:
            for cluster_id, cluster in clusters.groupby('cluster'):
                duration = len(cluster)
                signal_value = ann_dict[year]
                max_intensity = cluster[var_interest].max()
                
                # Build the result entry
                result.append({
                    'Year': year,
                    'Signal': signal_value,
                    'Frequency': 1,  # Temporarily mark each cluster as a single event
                    'Cluster_ID': f"{year}.{cluster_id}",
                    'Duration': duration,
                    'Intensity': max_intensity
                })
        else:
            # Handle years with no exceedances
            result.append({
                'Year': year,
                'Signal': ann_dict[year],
                'Frequency': 0,
                'Cluster_ID': f"{year}.0",
                'Duration': 0,
                'Intensity': annual_max[year]
            })
    
    # Convert result list to DataFrame
    result_df = pd.DataFrame(result)

    # Aggregate frequency of clusters per year
    frequency_df = result_df.groupby('Year')['Frequency'].sum().reset_index()
    result_df = result_df.drop('Frequency', axis=1).merge(frequency_df, on='Year', how='left')

    # Calculate yearly statistics
    yearly_df = result_df.groupby('Year').agg(
        mean_duration=('Duration', 'mean'),
        mean_frequency=('Frequency', 'mean'),
        mean_intensity=('Intensity', 'mean'),
        signal=('Signal', 'mean')
    ).reset_index()
    
    return result_df, yearly_df



def identify_clusters(df, type):
    """
    Identifies clusters in the DataFrame based on 'Start_Cluster', 'End_Cluster', and 'Exceeds_Wave' flags.

    Parameters:
    - df: A pandas DataFrame with columns 'Start_Cluster', 'End_Cluster', 'Exceeds_Wave', and 'Exceedance_Diff_Wave'.

    Returns:
    - A list of clusters, where each cluster is a list of tuples. Each tuple contains the row index and the 'Exceedance_Diff_Wave' value.
    """
    clusters = []  # Initialize a list to hold cluster data
    current_cluster = []  # Temporary list to hold data of the current cluster

    exceeds = 'Exceeds_'+type
    exceeds_diff = 'Exceedance_Diff_'+type
    start = 'Start_Cluster_'+type
    end = 'End_Cluster_'+type

    # Loop through DataFrame
    for idx, row in df.iterrows():
        if row[start] or (row[exceeds] == 1 and not current_cluster):
            # Start of a new cluster
            current_cluster.append((idx, row[exceeds_diff]))
        elif row[exceeds] == 1:
            # Continuation of the current cluster
            current_cluster.append((idx, row[exceeds_diff]))
        elif row[end] and current_cluster:
            # End of the current cluster, add to clusters and reset current_cluster
            clusters.append(current_cluster)
            current_cluster = []

    # Check if the last cluster is closed
    if current_cluster:
        clusters.append(current_cluster)
    
    return clusters



def trajectory_dict_plot(df, plot_traj):
    """
    Analyzes standard exceedance clusters and optionally plots their trajectories.

    Parameters:
    - df: DataFrame containing the data to analyze.
    - plot_traj: Boolean indicating whether to plot the exceedance trajectories.
    """
    clusters_std = identify_clusters(df, 'Std')
    cluster_dict_std = {}

    if plot_traj:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    sigma_vec_std = []
    for cluster in clusters_std:
        if len(cluster) not in cluster_dict_std:
            cluster_dict_std[len(cluster)] = []
        clust_vals = np.array([t[1] for t in cluster])
        cluster_dict_std[len(cluster)].append(clust_vals / np.max(clust_vals))
        if 1 < len(cluster) <= 31:
            x_values = range(1, len(cluster) + 1)
            y_values = [value for _, value in cluster]
            if plot_traj:
                ax.plot(x_values, y_values, marker='o', linestyle='-')
            sigma_vec_std.append(np.std(y_values))

    if plot_traj:
        ax.set_xlabel('Exceedance Duration')
        ax.set_ylabel('Exceedance Value')
        ax.set_title("Standard Exceedance Clusters")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Calculate and print the average standard deviations
    sigma0_std = np.mean(sigma_vec_std) if sigma_vec_std else 0

    return cluster_dict_std, sigma0_std



def fit_empirical_copula(result_df, sample_data):
    """
    Fit an empirical copula by transforming data, sampling, and reordering based on ranks.

    Args:
    result_df (DataFrame): DataFrame containing the original data.
    sample_data (dict): Dictionary containing arrays of sampled data for each variable.

    Returns:
    DataFrame: Returns a DataFrame w_df containing the reordered sampled data.
    """
    # Determine the number of samples
    first_key = next(iter(sample_data))
    n_samples = len(result_df[first_key].values) 
    
    # Create a DataFrame for copula variables of interest using dictionary keys
    x_df = pd.DataFrame({key: result_df[key] for key in sample_data})

    # Transform each vector into ranks
    z_df = pd.DataFrame({key: rankdata(result_df[key].values, method='ordinal') for key in sample_data})

    # Sampled from logspline pdfs
    x_prime_df = pd.DataFrame(sample_data)

    # Sorting each column of samples independently
    R_prime_df = pd.DataFrame({col: x_prime_df[col].sort_values().values for col in x_prime_df})

    # Bootstrap sampling original ranks with replacement and adjust for zero-based indexing
    z_prime_df = z_df.sample(n=n_samples, replace=True).reset_index(drop=True)
    z_prime_df -= 1  # Adjust for zero-based indexing

    # Create an empty DataFrame with the same structure
    w_df = pd.DataFrame(index=z_prime_df.index, columns=z_prime_df.columns)

    # Populate w_df by referencing R_prime_df using indices from z_prime_df
    for column in z_prime_df.columns:
        w_df[column] = R_prime_df[column].iloc[z_prime_df[column]].values
    
    # Rounding values to the nearest integer if necessary
    if 'Frequency' in w_df.columns:
        w_df['Frequency'] = np.where(w_df['Frequency'].round() < 0, 0, w_df['Frequency'].round()).astype(int)
    if 'Duration' in w_df.columns:
        w_df['Duration'] = np.where(w_df['Duration'].round() < 0, 0, w_df['Duration'].round()).astype(int)

    return w_df



def fit_logspline_density(data, n_samples, model=None, plot=False):
    """
    Fits a logspline density estimator to the given univariate data, samples from it,
    and optionally plots the density estimation.

    Parameters:
    data (array-like): The univariate data for which to estimate the density function.
    n_samples (int): Number of samples to draw.
    plot (bool): If True, plot the density estimation against the histogram of the data.

    Returns:
    dict: A dictionary containing the model, sample array, density function, and plot (if created).
    """
    if model == None:
        # Fit the logspline model to the data
        model = KDEMultivariate(data, var_type='c', bw='cv_ml')
    
    # Define the density function
    def density_function(x):
        return model.pdf(x)

    # Compute the CDF from the PDF
    x = np.linspace(data.min(), data.max(), 1000)
    pdf_values = density_function(x)
    cdf_values = cumtrapz(pdf_values, x, initial=0)
    cdf_values /= cdf_values[-1]  # Normalize to make it a proper CDF

    # Create an interpolation of the inverse CDF
    inverse_cdf = interp1d(cdf_values, x, bounds_error=False, fill_value=(x[0], x[-1]))

    # Draw uniform random samples for inverse transform sampling
    uniform_samples = np.random.rand(n_samples)
    samples = inverse_cdf(uniform_samples)

    # Optionally plot the density estimation
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(x, pdf_values, label='Logspline Density Estimate')
        plt.hist(data, bins=30, density=True, alpha=0.5, label='Histogram of Data')
        plt.hist(samples, bins=30, density=True, alpha=0.5, label='Histogram of Samples', color='red')
        plt.title('Logspline Density Estimation and Samples')
        plt.legend()
        plt.show()
        
    return model, samples, "Logspline"



def fit_intensity_distribution(list_of_peaks, intensity_dist, plot_RV, name, n_sample, status=0):
    """
    Fits specified distribution to peaks of the exceedance difference wave signal
    and optionally plots the fit. Defaults to 'Exponential' distribution for any input
    other than 'GPD' or 'Gamma'.

    Parameters:
    - list_of_peaks: List of peak values from the exceedance difference wave signal.
    - intensity_dist: String indicating the distribution type ('GPD', 'Gamma', or others default to 'Exponential').
    - plot_RV: Boolean indicating whether to plot the resulting fit.
    - name: Name of fitting e.g. "Intensity", "Duration"
    - n_sample: Number of samples to generate from the fitted distribution.

    Returns:
    - A dictionary containing the distribution parameters.
    """
    if intensity_dist == 'GPD':
        c, loc, scale = genpareto.fit(list_of_peaks)
        intensity_params = {'gpd_c': c, 'gpd_loc': loc, 'scale': scale}
        distribution_func = genpareto
    elif intensity_dist == 'Gamma':
        alpha, loc, beta = gamma.fit(list_of_peaks)
        intensity_params = {'gamma_alpha': alpha, 'gamma_loc': loc, 'scale': beta}
        distribution_func = gamma
    else:  # Default to Exponential
        loc, scale = expon.fit(list_of_peaks, floc=0)
        intensity_params = {'expon_loc': loc, 'scale': scale}
        distribution_func = expon
        intensity_dist = "Expon"

    # Generate samples from the fitted distribution
    samples = distribution_func.rvs(*intensity_params.values(), size=n_sample)

    if plot_RV:
        plot_distribution_fit(list_of_peaks, samples, distribution_func, intensity_params, name)

    # Perform KS Test
    ks_stat, ks_p_value = kstest(list_of_peaks, distribution_func.cdf, args=tuple(intensity_params.values()))
    print(f"KS Test Statistic: {ks_stat}, p-value: {ks_p_value}")

    if ks_p_value < 0.05 and status != -1:
        print(f"Bad fit indicated by ks test for {intensity_dist}.")
        if status == 0:
            intensity_params, samples, intensity_dist, (ks_stat, ks_p_value) = fit_intensity_distribution(list_of_peaks, "Gamma", plot_RV, name, n_sample, status=1)
        elif status == 1:
            intensity_params, samples, intensity_dist, (ks_stat, ks_p_value) = fit_intensity_distribution(list_of_peaks, "GPD", plot_RV, name, n_sample, status=2)
        else:
            intensity_params, samples, intensity_dist = fit_logspline_density(list_of_peaks, n_sample, plot=plot_RV)
            (ks_stat, ks_p_value) = "N/A - Logspline default", "N/A - Logspline default"
            print(f"Defaulted to Logspline PDF.")
    else:
        if status == -1:
            print(f"Fit enforced by user.")
        else:
            print(f"Good fit indicated by ks test for {intensity_dist}.")

    return intensity_params, samples, intensity_dist, (ks_stat, ks_p_value)



def fit_duration_distribution(std_clusters, duration_dist, plot_RV, name, n_sample, status=0):
    """
    Fits specified distribution to peaks of the exceedance difference wave signal
    and optionally plots the fit. Defaults to 'Exponential' distribution for any input
    other than 'GPD' or 'Gamma'.

    Parameters:
    - std_clusters: DataFrame containing the exceedance difference wave data.
    - intensity_dist: String indicating the distribution type ('GPD', 'Gamma', or others default to 'Exponential').
    - plot_RV: Boolean indicating whether to plot the resulting fit.
    - name: Name of fitting eg. "Intensity", "Duration"

    Returns:
    - A dictionary containing the distribution parameters.
    """
    if duration_dist == 'GPD':
        c, loc, scale = genpareto.fit(std_clusters)
        dur_params = {'gpd_c': c, 'gpd_loc': loc, 'scale': scale}
        distribution_func = genpareto
    elif duration_dist == 'Gamma':
        alpha, loc, beta = gamma.fit(std_clusters)
        dur_params = {'gamma_alpha': alpha, 'gamma_loc': loc, 'scale': beta}
        distribution_func = gamma
    else:  # Default to Exponential
        loc, scale = expon.fit(std_clusters, floc=0)
        dur_params = {'expon_loc': loc, 'scale': scale}
        distribution_func = expon
        duration_dist = "Expon"

    # Generate samples from the fitted distribution
    samples = distribution_func.rvs(*dur_params.values(), size=n_sample)

    if plot_RV:
        plot_distribution_fit(std_clusters, samples, distribution_func, dur_params, name)

    # Perform Chi-Squared Test
    observed_freq, bin_edges = np.histogram(std_clusters, bins='auto', density=False)
    expected_freq = np.diff(distribution_func.cdf(bin_edges, *dur_params.values())) * len(std_clusters)
    
    # Normalize expected frequencies
    expected_freq = expected_freq * observed_freq.sum() / expected_freq.sum()

    # Ensure no expected frequency is zero
    expected_freq += 1e-10

    chi_stat, chi_p_value = chisquare(observed_freq, f_exp=expected_freq)
    print(f"Chi-Squared Test Statistic: {chi_stat}, p-value: {chi_p_value}")

    if chi_p_value < 0.05 and status != -1:
        print(f"Bad fit indicated by chi_squared test for {duration_dist}.")
        if status == 0:
            dur_params, samples, duration_dist, (chi_stat, chi_p_value) = fit_duration_distribution(std_clusters, "Gamma", plot_RV, name, n_sample, status=1)
        elif status == 1:
            dur_params, samples, duration_dist, (chi_stat, chi_p_value) = fit_duration_distribution(std_clusters, "GPD", plot_RV, name, n_sample, status=2)
        else:
            dur_params, samples, duration_dist  = fit_logspline_density(std_clusters, n_sample, plot=plot_RV)
            (chi_stat, chi_p_value) = "N/A - Logspline default", "N/A - Logspline default"
            print(f"Defaulted to Logspline PDF.")
        
    else:
        if status == -1:
            print(f"Fit enforced by user.")
        else:
            print(f"Good fit indicated by chi_squared test for {duration_dist}.")
    
    return dur_params, samples, duration_dist, (chi_stat, chi_p_value)



def fit_frequency_distribution(list_of_frequencies, frequency_dist, plot_RV, name, n_sample, status=0):
    """
    Fits a Poisson distribution to frequency data and optionally plots the fit.

    Parameters:
    - list_of_frequencies: List or array of frequency data to fit.
    - plot_RV: Boolean indicating whether to plot the resulting fit.
    - name: Name of the fitting eg. "Frequency"
    - n_sample: Number of samples to generate from the fitted distribution.

    Returns:
    - A dictionary containing the distribution parameters and samples.
    """
    # Fit the Poisson distribution (lambda is the mean of the data)
    lambda_val = np.mean(list_of_frequencies)
    frequency_params = {'lambda': lambda_val}

    # Generate samples from the fitted distribution
    samples = poisson.rvs(mu=lambda_val, size=n_sample)

    if plot_RV:
        plot_pois_distribution_fit(list_of_frequencies, samples, poisson, frequency_params, name)

    # Perform Chi-Squared Test
    min_val = np.min(list_of_frequencies)
    max_val = np.max(list_of_frequencies)
    bins = np.arange(min_val, max_val + 1.5) - 0.5  # Define bins to align with integer values

    observed_freq, bin_edges = np.histogram(list_of_frequencies, bins=bins, density=False)
    bin_centers = np.arange(min_val, max_val + 1)  # Bin centers are integer values
    expected_freq = poisson.pmf(bin_centers, mu=lambda_val) * len(list_of_frequencies)

    # Normalize expected frequencies to ensure the sum matches the sum of observed frequencies
    expected_freq = expected_freq * observed_freq.sum() / expected_freq.sum()

    # Ensure no expected frequency is zero
    expected_freq[expected_freq == 0] = 1e-10

    chi_stat, chi_p_value = chisquare(observed_freq, f_exp=expected_freq)
    print(f"Chi-Squared Test Statistic: {chi_stat}, p-value: {chi_p_value}")

    if chi_p_value < 0.05 and status != -1:
        print(f"Bad fit indicated by chi_squared test for {frequency_dist}.")
        list_of_frequencies = list_of_frequencies + np.random.uniform(-1, 1, size=len(list_of_frequencies))
        frequency_params, samples, frequency_dist  = fit_logspline_density(list_of_frequencies, n_sample, plot=plot_RV)
        (chi_stat, chi_p_value) = "N/A - Logspline default", "N/A - Logspline default"
        print(f"Defaulted to Logspline PDF.")
    else:
        frequency_dist = 'Poisson'
        if status == -1:
            print(f"Fit enforced by user.")
        else:
            print(f"Good fit indicated by chi_squared test for {frequency_dist}.")
    
    return frequency_params, samples, frequency_dist, (chi_stat, chi_p_value)


    
def plot_pois_distribution_fit(list_of_frequencies, samples, distribution, params, name):
    plt.figure(figsize=(10, 6))
    plt.hist(list_of_frequencies, bins='auto', alpha=0.5, label='Observed Data', density=True)
    plt.hist(samples, bins='auto', alpha=0.5, label='Fitted Data', density=True)
    plt.title(f'Fit of {name}')
    plt.legend()
    plt.show()



def plot_exceedances_over_time(df, data_unit):
    """
    Plots the distribution of exceedances over time for both the annual wavelet signal and standard exceedances.

    Parameters:
    - df: DataFrame containing the data for exceedances.
    - data_unit: String, the unit of measurement for the exceedances (e.g., 'm^3/s').
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 7), sharey=True)  # Two plots side by side with shared Y axis
    
    # Plot for Annual Signal Exceedances
    axs[0].plot(df.index, df['Exceedance_Diff_Wave'], color='blue')
    axs[0].set_title('Annual Wavelet Signal Exceedances')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel(f'Exceedances ({data_unit})')
    
    # Plot for Standard Exceedances
    axs[1].plot(df.index, df['Exceedance_Diff_Wave'], color='red')
    axs[1].set_title('Standard Exceedances')
    axs[1].set_xlabel('Time')
    
    plt.suptitle('Distribution of Exceedances Over Time')  # Super title for both subplots
    plt.show()


def plot_distribution_fit(data, samples, dist_func, params, name):
    """
    Plots the histogram of the data, samples, and the fitted distribution.

    Parameters:
    - data: List of peak values to fit.
    - samples: Samples generated from the fitted distribution.
    - dist_func: Distribution function from scipy.stats (e.g., genpareto, gamma, expon).
    - params: Distribution-specific parameters.
    - name: Name of fitting eg. "Intensity", "Duration"
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(min(data), max(data), 100)
    
    # Calculate the PDF from the distribution parameters
    if dist_func == expon:
        pdf_fitted = dist_func.pdf(x, loc=params['expon_loc'], scale=params['scale'])
    elif dist_func == gamma:
        pdf_fitted = dist_func.pdf(x, a=params['gamma_alpha'], loc=params['gamma_loc'], scale=params['scale'])
    elif dist_func == genpareto:
        pdf_fitted = dist_func.pdf(x, c=params['gpd_c'], loc=params['gpd_loc'], scale=params['scale'])
    elif dist_func == poisson:
        pdf_fitted = poisson.pmf(x, mu=params['lambda'])
    
    dist_name = dist_func.name.title()

    # Plot histogram of the data
    data_hist = ax.hist(data, bins=30, density=True, alpha=0.5, label='Data Histogram')
    # Plot histogram of the samples
    samples_hist = ax.hist(samples, bins=30, density=True, alpha=0.5, label='Samples Histogram', color='green')
    
    # Plot the fitted distribution curve
    ax.plot(x, pdf_fitted, 'r-', label=f'{name} Fitted {dist_name}')

    # Set the x and y axis limits based on the data and samples histograms
    ax.set_xlim([min(data_hist[1][0], samples_hist[1][0]), max(data_hist[1][-1], samples_hist[1][-1])])
    ax.set_ylim([0, max(max(data_hist[0]), max(samples_hist[0])) * 1.1])

    ax.set_title(f'{name} Fit with {dist_name}')
    ax.legend()

    plt.tight_layout()
    plt.show()



def plot_logspline_pdf(data, intensity_density_function):
    points = np.linspace(min(data), max(data), 100)
    estimated_densities = intensity_density_function(points)
    
    plt.figure(figsize=(8, 6))
    plt.plot(points, estimated_densities, label='Logspline Density Estimate')
    plt.hist(data, bins=30, density=True, alpha=0.5, label='Histogram')
    plt.title('Density Estimation using Logspline')
    plt.legend()
    plt.show()



def FIDS_pairplot(df, jitter_col=None, jitter_amount=(-0.5, 0.5), columns=None, plot_kws=None, height=3):
    """
    Create a custom pairplot with jitter applied to a specified column, scatter plots, and KDEs.

    Args:
    df (DataFrame): The input DataFrame.
    jitter_col (str): The column to apply jitter.
    jitter_amount (tuple): The range of uniform distribution to add jitter.
    columns (list): List of column names to include in the pairplot.
    plot_kws (dict): Keyword arguments for the scatter plot.
    height (float): Height of each subplot in the pairplot.

    Returns:
    PairGrid: The resulting pairplot as a seaborn PairGrid object.
    """
    # Apply jitter to the specified column
    df = df.copy()  # Work on a copy to avoid modifying the original DataFrame
    if jitter_col is not None:
        for col in jitter_col:
            df[col] = df[col] + np.random.uniform(*jitter_amount, size=len(df))
    
    if columns is None:
        columns = [col for col in df.columns if col != jitter_col]
    
    if plot_kws is None:
        plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'}
    
    # Create the pairplot
    pair_grid = sns.pairplot(df[columns], kind='scatter', diag_kind='kde', plot_kws=plot_kws, height=height)

    # Customize the lower triangle
    for i, j in zip(*np.tril_indices_from(pair_grid.axes, -1)):
        ax = pair_grid.axes[i, j]
        ax.clear()  # Clear the existing scatter plot
        sns.kdeplot(x=df[pair_grid.x_vars[j]], y=df[pair_grid.y_vars[i]], ax=ax, color='blue', fill=True)

    plt.show()
