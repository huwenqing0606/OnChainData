
############# BTC Value (USD) vs. Day with Moving Average ###########
#####################################################################

# Script to graph the closing price of Bitcoin per day with moving average
# Bitcoin data was retrieved from Yahoo finance from dates 11-11-19 to 11-09-20
# File was convered to tsv file and named BTC-USD.tsv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, r2_score
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from itertools import combinations

Bitcoin_Value = pd.read_csv('/Data/BTC-USD.tsv', sep='\t')

df = pd.DataFrame(Bitcoin_Value)

# Convert 'date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Ensure the DataFrame is sorted by date
df.sort_values('Date', inplace=True)

# Calculate the 20-day moving average of 'Adj Close'
df['Moving Avg BTC USD'] = df['Adj Close'].rolling(window=20).mean()

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Adj Close'], label='Bitcoin Value (USD)', color='lightblue')
plt.plot(df['Date'], df['Moving Avg BTC USD'], label='20-Day Moving Average', color='orange', linestyle='-')

# Formatting the plot
plt.title('BTC Value (USD) vs. Time')
plt.xlabel('Date')
plt.ylabel('BTC Value (USD)')
plt.legend()

# Format the x-axis to display dates clearly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Adjust for desired tick frequency
plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

# Add a red vertical line at May 11, 2020
plt.axvline(pd.Timestamp('2020-05-11'), color='red', alpha=0.3, linestyle='-', lw=2)

# Show the plot
plt.show()


##Script to generate Graphs of Summed Variables per Day with 20-day moving average #######
######################################################################################

#reward_usd
#transaction_count
#difficulty
#fee_total_usd
#For all blocks verified within one day (11-11-2019 to 11-09-2019), the variable is summed and grouped by date.

# Function to load data, process, and plot with moving average
def plot_with_moving_avg(file_path, metric, agg_func='sum', window_size=20, plot_title='', y_label=''):
    # Load data
    df = pd.read_csv(file_path, sep='\t')
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date

    # Group by 'date' and calculate the specified aggregation function
    grouped = df.groupby('date').agg({metric: agg_func})

    # Calculate the moving average
    grouped['moving_avg'] = grouped[metric].rolling(window=window_size).mean()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(grouped.index, grouped[metric], color='lightblue', label=f'Total Daily {y_label}')
    plt.plot(grouped.index, grouped['moving_avg'], color='orange', label='20-Day Moving Average')
    plt.title(plot_title)
    plt.xlabel('Date')
    plt.ylabel(y_label)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
    plt.gcf().autofmt_xdate()
    plt.axvline(pd.Timestamp('2020-05-11'), color='red', alpha=0.3, linestyle='-', lw=2)
    plt.legend()
    plt.show()

# File path
file_path = '/Data/Bitcoin_Combined_20191111_20201109.tsv'

# Plotting different metrics
plot_with_moving_avg(file_path, 'reward_usd', agg_func='sum', plot_title='Daily Summation of Rewards vs. Time', y_label='Rewards (USD)')
plot_with_moving_avg(file_path, 'transaction_count', agg_func='sum', plot_title='Total Daily Transactions vs. Time', y_label='Transaction Count')
plot_with_moving_avg(file_path, 'difficulty', agg_func='sum', plot_title='Total Daily Difficulty vs. Time', y_label='Difficulty')
plot_with_moving_avg(file_path, 'fee_total_usd', agg_func='sum', plot_title='Total Daily Fee vs. Time', y_label='Total Daily Fee (USD)')

##Script to generate Graphs of Averaged Variables per Day with 20-day moving average #######
########################################################################################

#reward_usd
#transaction_count
#difficulty
#fee_total_usd
#For all blocks verified within one day (11-11-2019 to 11-09-2019), the variable is averaged by dividing the 
#number of blocks verified on that day.

def plot_metric(file_path, metric_name, agg_func='mean', window_size=20, plot_title='', y_label=''):
    # Load data and preprocess
    df = pd.read_csv(file_path, sep='\t')
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date

    # Group by date and calculate the specified aggregation function for the metric
    grouped = df.groupby('date').agg({metric_name: agg_func})

    # Calculate the moving average
    grouped['moving_avg'] = grouped[metric_name].rolling(window=window_size).mean()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(grouped.index, grouped[metric_name], color='lightblue', label=f'Average Daily {y_label}')
    plt.plot(grouped.index, grouped['moving_avg'], color='orange', label='20-Day Moving Average')

    # Formatting the plot
    plt.title(plot_title)
    plt.xlabel('Date')
    plt.ylabel(y_label)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
    plt.gcf().autofmt_xdate()
    plt.axvline(pd.Timestamp('2020-05-11'), color='red', alpha=0.3, linestyle='-', lw=2)
    plt.legend()
    plt.show()

# Define the file path
file_path = '/Data/Bitcoin_Combined_20191111_20201109.tsv'

# Plotting different metrics using the function
plot_metric(file_path, 'reward_usd', 'mean', 20, 'Average Daily Rewards vs. Time', 'Rewards (USD)')
plot_metric(file_path, 'transaction_count', 'mean', 20, 'Average Daily Transactions vs. Time', 'Transaction Count')
plot_metric(file_path, 'difficulty', 'mean', 20, 'Average Daily Difficulty vs. Time', 'Difficulty')
plot_metric(file_path, 'fee_total_usd', 'mean', 20, 'Average Daily Fee vs. Time', 'Daily Fee (USD)')

##Script to generate Graphs of Averaged Variables per Transaction each Day with 20-day moving average #######
#######################################################################################################

#reward_usd
#transaction_count
#difficulty
#fee_total_usd
#For all blocks verified within one day (11-11-2019 to 11-09-2019), the variable is summed by date and dividing by
#the number of transactions on that day, hence averaged variable per transaction.

def plot_metric_with_moving_avg(dataframe, metric_name, color, label, window_size=20):
    # Group by 'date' and calculate the sum of the metric and 'transaction_count', then compute the average per transaction
    grouped = dataframe.groupby('date').agg({metric_name: 'sum', 'transaction_count': 'sum'})
    grouped[f'avg_{metric_name}_per_transaction'] = grouped[metric_name] / grouped['transaction_count']
    
    # Calculate the moving average of the metric per transaction
    grouped[f'moving_avg_{metric_name}'] = grouped[f'avg_{metric_name}_per_transaction'].rolling(window=window_size).mean()
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(grouped.index, grouped[f'avg_{metric_name}_per_transaction'], color=color, label=label)
    plt.plot(grouped.index, grouped[f'moving_avg_{metric_name}'], color='orange', label=f'20-Day Moving Average')
    
    # Adding title and labels
    plt.title(f'{label} vs. Time')
    plt.xlabel('Date')
    plt.ylabel(label)
    plt.legend()
    
    # Format the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Adjust the interval as needed
    plt.gcf().autofmt_xdate()  # Rotate the dates for better readability
    
    # Add a red vertical line at May 11, 2020
    plt.axvline(pd.Timestamp('2020-05-11'), color='red', alpha=0.3, linestyle='-', lw=2)
    
    plt.show()

# Load your data
Bitcoin_Blocks = pd.read_csv('/Data/Bitcoin_Combined_20191111_20201109.tsv', sep='\t')
Bitcoin_Blocks['time'] = pd.to_datetime(Bitcoin_Blocks['time'])
Bitcoin_Blocks['date'] = Bitcoin_Blocks['time'].dt.date

# Use the function to plot different metrics
plot_metric_with_moving_avg(Bitcoin_Blocks, 'reward_usd', 'lightblue', 'Average Reward per Transaction')
plot_metric_with_moving_avg(Bitcoin_Blocks, 'difficulty', 'lightblue', 'Average Difficulty per Transaction')
plot_metric_with_moving_avg(Bitcoin_Blocks, 'fee_total_usd', 'lightblue', 'Average Fee per Transaction')


############ Add Value and Moving averages to BTC-USD Data Frame ##########
##########################################################################

# BTC-USD Dataframe was obtained form Yahoo Finance to accuire closing price 
# of Bitcoin for each day (11-11-19 to 11-09-20)

# Load the Bitcoin Blocks data
Bitcoin_Blocks = pd.read_csv('/Data/Bitcoin_Combined_20191111_20201109.tsv', sep='\t')
Bitcoin_Blocks['time'] = pd.to_datetime(Bitcoin_Blocks['time'])
Bitcoin_Blocks['date'] = Bitcoin_Blocks['time'].dt.date

# Calculate necessary metrics and their moving averages
metrics = ['reward_usd', 'transaction_count', 'fee_total_usd', 'difficulty']
aggregations = {metric: ['sum', 'mean'] for metric in metrics}
grouped = Bitcoin_Blocks.groupby('date').agg(aggregations)

# Flatten the multi-level columns
grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

# Calculate average data per transaction
for metric in metrics:
    grouped[f'{metric}_per_transaction'] = grouped[f'{metric}_sum'] / grouped['transaction_count_sum']

# Define window size for moving averages and calculate them
window_size = 20
for column in grouped.columns:
    if 'sum' in column or 'mean' in column or 'per_transaction' in column:
        grouped[f'moving_avg_{column}'] = grouped[column].rolling(window=window_size).mean()

# Reset index to make 'date' a column for merging
grouped.reset_index(inplace=True)

# Load the BTC-USD data
btc_usd = pd.read_csv('/Data/BTC-USD.tsv', sep='\t', parse_dates=['Date'])
btc_usd['date'] = btc_usd['Date'].dt.date

# Merge the BTC-USD data with the calculated metrics based on the 'date' column
merged_data = btc_usd.merge(grouped, on='date', how='left')

# Select only relevant columns to include in the final dataset (modify this list based on your needs)
columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
columns_to_keep += [col for col in grouped.columns if col != 'date']

final_data = merged_data[columns_to_keep]

# Save the merged data back to a TSV or CSV file
final_data.to_csv('/Users/pauloconnor/Desktop/py.scripts/Bitcoin tsv/BTC-USD_2.1.tsv', sep='\t', index=False)

############ Add 20-day Moving Average of Bitcoin Value to BTC-USD Data Frame #####
############################################################################

# Load the BTC-USD combined data
btc_usd_combined = pd.read_csv('/Data/BTC-USD_2.1.tsv', sep='\t', parse_dates=['Date'])
btc_usd_combined['date'] = btc_usd_combined['Date'].dt.date

# Calculate the 20-day moving average of 'Adj Close'
btc_usd_combined['moving_avg_BTC USD'] = btc_usd_combined['Adj Close'].rolling(window=20).mean()

# Save the updated DataFrame back to a TSV file
btc_usd_combined.to_csv('/Data/BTC-USD_2.2.tsv', sep='\t', index=False)

########### Remove moving_avg_transaction_count_per_transaction#####
####################################################################

#This is meaningless vector as it always equate to 1, therefore it was removed from the data frame

# Load your DataFrame
df = pd.read_csv('/Data/BTC-USD_2.2.tsv', sep='\t')

# Remove the 'moving_avg_transaction_count_per_transaction' column
df = df.drop(columns=['moving_avg_transaction_count_per_transaction'])

# Save the updated DataFrame back to a TSV file
df.to_csv('/Data/BTC-USD_2.2.tsv', sep='\t', index=False)


##################### Correlation Matrix of Moving Average #########
####################################################################

# Load your dataset
df = pd.read_csv('/Data/BTC-USD_2.2.tsv', sep='\t')

# Filter columns that contain 'moving_avg' in their names to focus on moving averages
moving_avg_columns = [col for col in df.columns if 'moving_avg' in col]

# Compute the correlation matrix for these moving average columns
corr = df[moving_avg_columns].corr()

# Define custom titles for the rows and columns in your correlation matrix
custom_titles = {
    'moving_avg_reward_usd_sum': 'Total Reward/Day USD',
    'moving_avg_reward_usd_mean': 'Avg Reward/Day USD ',
    'moving_avg_transaction_count_sum': 'Total Trans Count/Day',
    'moving_avg_transaction_count_mean': 'Avg Trans Count/Day',
    'moving_avg_fee_total_usd_sum': 'Total Fee/Day',
    'moving_avg_fee_total_usd_mean': 'Avg Fee/Day',
    'moving_avg_difficulty_sum': 'Total Difficulty/Day',
    'moving_avg_difficulty_mean': 'Avg Difficulty/Day',
    'moving_avg_reward_usd_per_transaction': 'Avg Reward/Trans',
    'moving_avg_fee_total_usd_per_transaction': 'Avg Fee/Trans',
    'moving_avg_difficulty_per_transaction': 'Avg Difficult/Trans',
    'moving_avg_BTC USD': 'Bitcoin USD/Day'
    
    # Add more custom titles as needed
}

# Replace column names in the correlation matrix with custom titles
corr.rename(columns=custom_titles, index=custom_titles, inplace=True)

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio, formatting annotations to two decimal places
sns.heatmap(corr, cmap=cmap, vmax=1, center=0, annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=corr.columns, yticklabels=corr.columns)

# Add a title to the heatmap
plt.title('Correlation Matrix of Bitcoin Data (Moving Averages)')

# Optionally, rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()


##################### Correlation Matrix of Block Values ###################
############################################################################

# Load your dataset
df = pd.read_csv('/Data/BTC-USD_2.2.tsv', sep='\t')

# Define the columns to be used in the correlation matrix and their intended titles
columns_with_custom_titles = {
    'reward_usd_sum': 'Total Reward/Day USD',
    'reward_usd_mean': 'Avg Reward/Day USD',
    'transaction_count_sum': 'Total Trans Count/Day',
    'transaction_count_mean': 'Avg Trans Count/Day',
    'fee_total_usd_sum': 'Total Fee/Day',
    'fee_total_usd_mean': 'Avg Fee/Day',
    'difficulty_sum': 'Total Difficulty/Day',
    'difficulty_mean': 'Avg Difficulty/Day',
    'reward_usd_per_transaction': 'Avg Reward/Trans',
    'fee_total_usd_per_transaction': 'Avg Fee/Trans',
    'difficulty_per_transaction': 'Avg Difficult/Trans',
    'Adj Close': 'Bitcoin USD/Day'
}

# Select only the specified columns for the correlation matrix
selected_columns = list(columns_with_custom_titles.keys())
corr = df[selected_columns].corr()

# Rename the columns and rows with the intended custom titles
corr.rename(columns=columns_with_custom_titles, index=columns_with_custom_titles, inplace=True)

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio, formatting annotations to two decimal places
sns.heatmap(corr, cmap=cmap, vmax=1, center=0, annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=corr.columns, yticklabels=corr.columns)

# Add a title to the heatmap
plt.title('Correlation Matrix of Bitcoin Data')

# Optionally, rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()


####### Partial least Squares all Combinations of Predictor Variables ###########
#################################################################################

# This script creates a data frame that estimates root mean sqaure error for all
# possible combinations of predictor variables (minus reward_usd) and components
# The dependent variable is the 20-moving average of Bitcoin

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Custom scorer for cross_val_score
rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Load your dataset
df = pd.read_csv('/Data/BTC-USD_2.2.tsv', sep='\t')  # Update the path accordingly

# Exclude the first 19 rows
df = df.iloc[19:]

# Define predictor and response variables
columns = [
    'moving_avg_transaction_count_sum', 'moving_avg_transaction_count_mean',
    'moving_avg_fee_total_usd_sum', 'moving_avg_fee_total_usd_mean',
    'moving_avg_difficulty_sum', 'moving_avg_difficulty_mean',
    'moving_avg_fee_total_usd_per_transaction',
    'moving_avg_difficulty_per_transaction'
]

y = df['moving_avg_BTC USD']

# Cross-validation configuration
kf = KFold(n_splits=5, shuffle=True, random_state=42)
max_components = 8

# Initialize variables to find the best combination
min_rmse = np.inf
best_combination = None
best_n_components = 0

# Results list for saving
final_results = []

# Iterate over all non-empty combinations of predictor variables
for r in range(1, len(columns) + 1):
    for subset in combinations(columns, r):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[list(subset)])
        subset_results = []

        for n_components in range(1, min(len(subset), max_components) + 1):
            pls = PLSRegression(n_components=n_components)
            scores = cross_val_score(pls, X_scaled, y, scoring=rmse_scorer, cv=kf)
            avg_rmse = -scores.mean()

            # Save results
            subset_results.append(avg_rmse)

            # Update best combination if this is the lowest RMSE so far
            if avg_rmse < min_rmse:
                min_rmse = avg_rmse
                best_combination = subset
                best_n_components = n_components

        # Append the results for the current subset to the final results list
        final_results.append({"Variables": subset, "RMSE": subset_results})

# Convert the final results to a DataFrame for display and save
final_results_df = pd.DataFrame(final_results)
#final_results_df.to_csv('/Data/RMSE.tsv', sep='\t', index=False)  # Update the path accordingly

# Print the best configuration
print(f"Best RMSE: {min_rmse}")
print(f"Best combination of predictors: {best_combination}")
print(f"Best number of components: {best_n_components}")
print("Results saved to '/Data/RMSE.tsv'")

########### Comparison of RMSE for Each Component using all Predictors #############
#################################################################################

# This script calculates the RMSE up to 8 components using all predictors
#except for rewards_usd

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Custom scorer for cross_val_score
rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Load your dataset
df = pd.read_csv('/Data/BTC-USD_2.2.tsv', sep='\t')  # Update the path accordingly

# Exclude the first 19 rows
df = df.iloc[19:]

# Define predictor and response variables
columns = [
    'moving_avg_transaction_count_sum', 'moving_avg_transaction_count_mean',
    'moving_avg_fee_total_usd_sum', 'moving_avg_fee_total_usd_mean',
    'moving_avg_difficulty_sum', 'moving_avg_difficulty_mean',
    'moving_avg_fee_total_usd_per_transaction',
    'moving_avg_difficulty_per_transaction'
]

X = df[columns]
y = df['moving_avg_BTC USD']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validation configuration
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Testing 1 through 8 components
max_components = 8
results = []

for n_components in range(1, max_components + 1):
    pls = PLSRegression(n_components=n_components)
    
    # Perform cross-validation and calculate RMSE
    scores = cross_val_score(pls, X_scaled, y, scoring=rmse_scorer, cv=kf)
    
    # Append the average RMSE and number of components to the results
    results.append({"Components": n_components, "RMSE": -scores.mean()})  # Scores are negated to make them positive

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Print the DataFrame
print(results_df)

##################### Variance Explained ################
########################################################

# This script graphs the %varianc of the ouput relative to
# the number of components to use, helping to select the optimal 
# number of components to use

# Load dataset
df = pd.read_csv('/Data/BTC-USD_2.2.tsv', sep='\t')

# Exclude the first 19 rows to account for the moving average calculation starting from the 20th data point
df = df.iloc[19:]

# Select the columns for PLS regression, predictors of rewards were removed due to halving
columns = [
    'moving_avg_transaction_count_sum', 'moving_avg_transaction_count_mean',
    'moving_avg_fee_total_usd_sum', 'moving_avg_fee_total_usd_mean',
    'moving_avg_difficulty_sum', 'moving_avg_difficulty_mean',
    'moving_avg_fee_total_usd_per_transaction',
    'moving_avg_difficulty_per_transaction'
]

X = df[columns]  # Predictor variables
y = df['moving_avg_BTC USD']  # Target variable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose a maximum number of components to test
max_components = min(X_scaled.shape[1], 15)
variance_explained_y = []

for n_components in range(1, max_components + 1):
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, y)
    y_pred = pls.predict(X_scaled)
    
    # Calculate the R^2 score as the percentage of variance explained in Y
    variance_explained = r2_score(y, y_pred)
    variance_explained_y.append(variance_explained)

# Plotting the variance explained in Y
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_components + 1), np.array(variance_explained_y) * 100, marker='o')  # Multiply by 100 to get percentages
plt.xlabel('Number of PLS Components')
plt.ylabel('Percentage of Variance Explained in Y')
plt.title('Variance Explained in Y by PLS Components')
plt.grid(True)
plt.show()

######## Partial Least Sqaures of Moving Average BTC (USD) ############
######################################################################

# Load dataset
df = pd.read_csv('/Data/BTC-USD_2.2.tsv', sep='\t')

# Exclude the first 19 rows to account for the moving average calculation starting from the 20th data point
df = df.iloc[19:]

# Select the columns for PLS regression, predictors of reward were removed due to halving
columns = [
    'moving_avg_transaction_count_sum', 'moving_avg_transaction_count_mean',
    'moving_avg_fee_total_usd_sum', 'moving_avg_fee_total_usd_mean',
    'moving_avg_difficulty_sum', 'moving_avg_difficulty_mean',
    'moving_avg_fee_total_usd_per_transaction',
    'moving_avg_difficulty_per_transaction'
]

X = df[columns]  # Predictor variables
y = df['moving_avg_BTC USD']  # Target variable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and fit the PLS model
pls = PLSRegression(n_components=7)
pls.fit(X_train, y_train)

# Extract coefficients and construct the linear equation
coefficients = pls.coef_.flatten()
equation_terms = [f"({coef:.3f} * {name})" for coef, name in zip(coefficients, columns)]
equation = " + ".join(equation_terms)
print("Linear Equation:")
print(f"y = {equation}")

# Extract weights for the first component
component_weights = {}

# Iterate over the desired range of components (1 through 7 in this case)
for i in range(7):  # i will go from 0 to 6, corresponding to components 1 to 7
    component_weights[f'Component {i+1}'] = pls.x_weights_[:, i]

# Now you can print or process the weights as needed
for component, weights in component_weights.items():
    print(f"Weights for {component}: {weights}")

# Coefficients for the components in predicting the response
print("PLS Regression coefficients:", pls.coef_)

# Predictions
y_pred_train = pls.predict(X_train)
y_pred_test = pls.predict(X_test)

# Calculate and print the RMSE for both training and test sets
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f'Training RMSE: {rmse_train:.2f}')
print(f'Test RMSE: {rmse_test:.2f}')

# Calculate and print additional metrics
print("\nTraining Data Metrics:")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_train, y_pred_train):.2f}")
print(f"Coefficient of Determination (R²): {r2_score(y_train, y_pred_train):.2f}")

print("\nTest Data Metrics:")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_test):.2f}")
print(f"Coefficient of Determination (R²): {r2_score(y_test, y_pred_test):.2f}")

############## K-fold Cross Validation ################################
######################################################################

#K-fold cross validation to test predictive capacity of the final PLS model

# Load your dataset
df = pd.read_csv('/Users/pauloconnor/Desktop/py.scripts/Bitcoin tsv/BTC-USD_2.2.tsv', sep='\t')

# Exclude the first 19 rows to account for the moving average calculation starting from the 20th data point
df = df.iloc[19:]

# Select the columns for PLS regression
columns = [
    'moving_avg_transaction_count_sum', 'moving_avg_transaction_count_mean',
    'moving_avg_fee_total_usd_sum', 'moving_avg_fee_total_usd_mean',
    'moving_avg_difficulty_sum', 'moving_avg_difficulty_mean',
    'moving_avg_fee_total_usd_per_transaction',
    'moving_avg_difficulty_per_transaction'
]

X = df[columns]  # Predictor variables
y = df['moving_avg_BTC USD']  # Target variable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the PLS model
pls = PLSRegression(n_components=7)

# Fit the PLS model
pls.fit(X_scaled, y)

# Extract coefficients and construct the linear equation
coefficients = pls.coef_.flatten()
equation_terms = [f"({coef:.3f} * {name})" for coef, name in zip(coefficients, columns)]
equation = " + ".join(equation_terms)
print("Linear Equation:")
print(f"y = {equation}")

################# Evaluation of Error ####################
#########################################################

# Number of components used in the PLS model
n_components = pls.n_components

# Display weights for each component
for i in range(n_components):
    component_weights = pls.x_weights_[:, i]
    print(f"Weights for component {i+1}: {component_weights}")

    # Constructing a string for each component's weights for better readability
    weight_str = " + ".join([f"{weight:.3f}*X{index+1}" for index, weight in enumerate(component_weights)])
    print(f"Component {i+1}: {weight_str}\n")

# Perform K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_cv_pred = cross_val_predict(pls, X_scaled, y, cv=kf)

# Calculate the cross-validated RMSE
cv_rmse = np.sqrt(mean_squared_error(y, y_cv_pred))

# Calculate the cross-validated MAE
cv_mae = mean_absolute_error(y, y_cv_pred)

# Calculate the cross-validated R-squared value
cv_r2 = r2_score(y, y_cv_pred)

# Print the metrics
print(f'Cross-validated RMSE: {cv_rmse:.2f}')
print(f'Cross-validated MAE: {cv_mae:.2f}')
print(f'Cross-validated R-squared: {cv_r2:.2f}')

################### Plot PLS Regression vs BTC Price #################
####################################################################

y_pred = pls.predict(X_scaled)

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(df.index, y, label='Actual Moving Average BTC USD', color='lightblue', linewidth=2)
plt.plot(df.index, y_pred, label='PLS Predicted Moving Average USD', color='salmon', linestyle='-', linewidth=2)

plt.xlabel('Index')
plt.ylabel('Moving Average BTC USD')
plt.title('Actual vs. Predicted Moving Average BTC USD')
plt.legend()
plt.show()

######## Plot of PLS Regression Linear Equation #####################
####################################################################

y_pred_flattened = y_pred.flatten()

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred_flattened, color='lightblue', edgecolor='k', alpha=0.6)

# Plot a line 
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='salmon', linestyle='-', linewidth=2)

# Labeling the plot
plt.xlabel('Actual Moving Average BTC USD')
plt.ylabel('PLS Predicted Moving Average BTC USD')
plt.title('Scatter Plot of Actual vs. PLS Predicted Moving Average BTC USD')

# Show the plot
plt.show()
