import pandas as pd
import numpy as np
import os
import sys
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import matplotlib.pyplot as plt
from utils.featureBuilder import FeatureBuilder
#-----Add parent directory to python path--------
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
#------Configuration-------
crypto_symbol = 'BTC'
max_encoder_length = 60 #How many past days model sees
max_prediction_length = 1
output_dir = os.path.join(parent_dir,'trained_models')
batch_size = 64 # How many sequences to process in parallel
n_epochs = 10 # Number of training epochs (start small)
n_lags = 5 
#----- Path for saving checkpoints/models------
output_dir = os.path.join(parent_dir, "trained_models")
os.makedirs(output_dir,exist_ok=True)
model_name = f"{crypto_symbol}_tft_model" #Name for checkpoint


print("--- Configuration ---")
print(f"Crypto Symbol: {crypto_symbol}")
print(f"Encoder Length: {max_encoder_length}")
print(f"Prediction Length: {max_prediction_length}")
print(f"Output Directory: {output_dir}")
print("---------------------\n")


#---------Load and Preprocess Data------
print("Loading and Preprocessing Data")
try: 
    data_path = os.path.join(script_dir,"sample_data/BTC-USD_data.csv")
    data_raw = pd.read_csv(data_path,skiprows=[1])
except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}")
    sys.exit(1)

data_raw.rename(columns={'Price': 'date'},inplace=True)
data_raw['date'] = pd.to_datetime(data_raw['date'], format='%Y-%m-%d', errors='coerce')
data_raw.set_index('date', inplace=True)

cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in cols_to_convert:
    if col in data_raw.columns:
        if data_raw[col].dtype == 'object':
            data_raw[col] = pd.to_numeric(data_raw[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
        else:
            data_raw[col] = pd.to_numeric(data_raw[col], errors='coerce')
    else:
        print(f"Warning: Column '{col}' not found in the data.")
data_raw.dropna(subset=['Close','Volume'],inplace=True)

#----2. Feature Engineering------
print("====Feature Engineerong=====")
data_raw['Log Returns'] = np.log(data_raw["Close"] / data_raw['Close'].shift(1))
data_raw.replace([np.inf, -np.inf],np.nan,inplace=True)
data_raw.dropna(subset=["Log Returns"],inplace=True)

#Use feature builder
#Note that we create features first then add time_idx and group
#group needed by pytorch-forecasting

fb = FeatureBuilder(data_raw,target_col='Log Returns',n_lags= n_lags)
#We get all features created by the users feature builder
#We wont use the X,y split directly here but work with the full df from the builder

data_featured = fb.add_lag_features().add_rolling_features().add_technical_indicators().clean().df

#---- Prepare DataFrame for TimeSeriesDataSet------
print("--- Preparing DataFrame for TimeSeriesDataSet ---")
# torch forecasting needs specific columns:
# 1. A time index: A continuous integer sequence representing time steps.
# 2. A target column: What we want to predict ('Log Returns').
# 3. Group IDs: To distinguish different time series (here, just 'BTC').
# 4. Features: All other input columns.

#Reset time index if date is the index so we can add time_idx easily 
if isinstance(data_featured.index, pd.DatetimeIndex):
    data_featured = data_featured.reset_index()
    
#Add time index
data_featured['time_idx'] = range(len(data_featured))

#Add group ID
data_featured['group'] = crypto_symbol

# Add simple time features why we need these for transofrmers 
# Ensure 'date' column exists after reset_index
if 'date' in data_featured.columns:
    data_featured['month'] = data_featured['date'].dt.month.astype(str).astype("category") # Treat as category
    data_featured['day_of_week'] = data_featured['date'].dt.dayofweek.astype(str).astype("category")
else:
    print("Warning: 'date' column not found after index reset. Cannot add time features.")
    data_featured['month'] = 0 # Add dummy if needed later
    data_featured['day_of_week'] = 0

data_featured['Log Returns'] = data_featured['Log Returns'].astype(np.float32)


# Identify feature columns (adjust based on FeatureBuilder output)
# We need to tell TimeSeriesDataSet which features are known/unknown, static/dynamic, real/categorical
# For simplicity now, let's treat most features from FeatureBuilder as 'unknown reals'
# These are features whose future values we don't know at prediction time
#Why we need to say those things 

feature_cols = [col for col in fb.df.columns if col.startswith('lag_') or
                col.startswith('ma_') or col.startswith('std_') or
                col.startswith('z_score') or
                col in ['RSI_14', 'MACD', 'MACD_signal', 'EMA_12', 'EMA_26',
                        'BB_upper', 'BB_lower', 'OBV', 'Close', 'Volume']] # Added Close/Volume if they exist and needed

# Make sure all potential feature columns actually exist in the final df
feature_cols = [col for col in feature_cols if col in data_featured.columns]

print(f"Columns in final DataFrame: {data_featured.columns.tolist()}")
print(f"Feature columns identified: {feature_cols}")
print(data_featured.head())
print(data_featured.info())


# --- Stop Point 1: Data is prepared ---
print("\n--- Data loading and preparation complete. ---")
# --- ADD THIS SECTION TO HANDLE POTENTIAL REMAINING NaNs ---
print("\n--- Final Cleaning Before TimeSeriesDataSet ---")
# List all columns that TimeSeriesDataSet will actually use
# Make sure this list is accurate based on the parameters you pass to TimeSeriesDataSet below
cols_to_check = (
    ['Log Returns', 'time_idx', 'group'] + # Target, time, group
    time_varying_known_categoricals + # e.g., ['month', 'day_of_week']
    ['time_idx'] + # time_varying_known_reals
    feature_cols + # time_varying_unknown_reals (features from FeatureBuilder)
    ['Log Returns'] # Target also included in unknown_reals
)
# Remove duplicates
cols_to_check = sorted(list(set(cols_to_check)))

print(f"Checking columns for NaNs: {cols_to_check}")
initial_rows = len(data_featured)
# Drop rows where *any* of these essential columns have NaN
data_featured.dropna(subset=cols_to_check, inplace=True)
final_rows = len(data_featured)
print(f"Dropped {initial_rows - final_rows} rows with NaNs in required columns.")

# IMPORTANT: Recreate time_idx after dropping rows to ensure it's contiguous
data_featured.sort_values('date', inplace=True) # Ensure data is sorted by date first
data_featured['time_idx'] = range(len(data_featured))
print(f"Recreated time_idx. New range: {data_featured['time_idx'].min()} to {data_featured['time_idx'].max()}")
# --- END OF ADDED SECTION ---

# --- 4. Create TimeSeriesDataSet ---
print("\n--- Creating TimeSeriesDataSet ---")

# Define the training cutoff point (e.g., use last prediction_length + encoder_length steps for validation)
# This ensures the validation set is truly "future" data relative to the training set
validation_cutoff = data_featured["time_idx"].max() - max_prediction_length
training_cutoff = validation_cutoff - max_encoder_length # Start validation encoder here
# A simpler split: use ~80% for training
# training_cutoff = data_featured["time_idx"].quantile(0.8)


print(f"Data time_idx range: {data_featured['time_idx'].min()} to {data_featured['time_idx'].max()}")
print(f"Training cutoff (time_idx <= {training_cutoff})")
# print(f"Validation starts (time_idx > {training_cutoff})") # Use this if using quantile split


# We need to tell the dataset about the different types of features
# time_varying_unknown_reals: Features whose future values are unknown at prediction time (most of ours)
# time_varying_known_reals: Features whose future values ARE known (e.g., time_idx, maybe day/month if predicting far ahead)
# static_categoricals: Features that don't change over time for a group (e.g., the 'group' column itself)
# target_normalizer: How to scale the target variable ('Log Returns'). GroupNormalizer is common.

# Identify categorical time features we added
time_varying_known_categoricals = [col for col in ['month', 'day_of_week'] if col in data_featured.columns]

training_dataset = TimeSeriesDataSet(
    data_featured[lambda x: x.time_idx <= training_cutoff], # Data for training
    time_idx="time_idx",
    target="Log Returns",
    group_ids=["group"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    # static_categoricals=["group"], # Use if you have features constant per group
    time_varying_known_categoricals=time_varying_known_categoricals, # month, day_of_week
    time_varying_known_reals=["time_idx"], # time_idx is known in the future
    time_varying_unknown_reals=feature_cols + ["Log Returns"], # Past values of features and target
    # We include Log Returns also as unknown_reals because the model can use past target values as input features.
    target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"), # Normalizes target by group
    add_relative_time_idx=True, # Adds relative time index automatically
    add_target_scales=True,     # Adds target scaling information
    add_encoder_length=True,    # Adds encoder length information
)

# Create validation dataset: uses data after training cutoff for prediction target,
# but needs encoder data before the cutoff as input
validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,
    data_featured,
    predict=True, # Set to predict mode
    stop_randomization=True # Important for validation
)

# Convert datasets to dataloaders for training
# num_workers=0 helps avoid potential issues on some systems, adjust if needed
train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0) # Larger batch size for validation is fine

print("--- TimeSeriesDataSet and DataLoaders created ---")


# --- 5. Define Model & Trainer ---
print("\n--- Defining Model and Trainer ---")

# Configure PyTorch Lightning Trainer
# Set default_root_dir to save logs/checkpoints in a specific place
pl.seed_everything(42) # Set seed for reproducibility

# Define callbacks
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
# Checkpoint callback to save the best model based on validation loss
checkpoint_callback = ModelCheckpoint(
    dirpath=output_dir,
    filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.4f}}",
    save_top_k=1, # Save only the best model
    verbose=True,
    monitor="val_loss",
    mode="min",
)

trainer = pl.Trainer(
    max_epochs=n_epochs,
    # accelerator="gpu" if torch.cuda.is_available() else "cpu", # Use GPU if available
    # devices=1, # Number of GPUs/CPUs to use
    gradient_clip_val=0.1, # Helps prevent exploding gradients
    limit_train_batches=50,  # Limit batches per epoch for faster initial run (remove for full training)
    limit_val_batches=20,   # Limit validation batches (remove for full training)
    callbacks=[early_stop_callback, checkpoint_callback],
    logger=pl.loggers.TensorBoardLogger(save_dir=output_dir, name=model_name + "_logs"), # Log to TensorBoard
    # enable_progress_bar=True, # Set False if running in non-interactive environment
)

# Define the Temporal Fusion Transformer model
# Hyperparameters are examples, tuning is needed for optimal performance
tft = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate=0.03,
    hidden_size=32,         # Size of embeddings & network layers (rule of thumb: 16-64)
    attention_head_size=2,  # Number of attention heads
    dropout=0.1,            # Dropout rate
    hidden_continuous_size=16, # Size for continuous feature embeddings
    output_size=7,          # Number of output quantiles (for QuantileLoss, 7 is standard)
    loss=QuantileLoss(),    # Loss function for quantile regression (good for uncertainty)
    # loss=MAE(), # Could use MAE for point forecasts
    # reduce_on_plateau_patience=4, # Optional LR scheduler config
)

print(f"--- Model Structure ---")
print(tft.hparams)
print("---------------------\n")

# --- 6. Train Model ---
print(f"--- Starting Training for {n_epochs} epochs ---")
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
print("--- Training Finished ---")

# --- 7. Load Best Model and Generate Predictions ---
print("\n--- Loading Best Model from Checkpoint ---")
# Load the best model checkpoint saved by the Checkpoint callback
best_model_path = checkpoint_callback.best_model_path
print(f"Best model path: {best_model_path}")

if best_model_path and os.path.exists(best_model_path):
    try:
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        print("Best model loaded successfully.")

        print("\n--- Generating Predictions on Validation Set ---")
        # Use the validation dataloader we created earlier
        # Make predictions (returns predicted values)
        predictions = best_tft.predict(val_dataloader, return_index=True, return_decoder_lengths=True)
        # Note: predict returns the point forecast (median for QuantileLoss).
        # For quantiles, use mode="quantiles"

        print(f"Predictions generated for {len(predictions.index)} validation time points.")

        # --- Plotting Actual vs Predicted using pytorch-forecasting ---
        # This method plots the first batch/sequence from the dataloader by default.
        # It's useful for a quick visual check.
        print("\n--- Plotting Example Prediction (First Sequence in Validation Loader) ---")
        # Get raw predictions for plotting quantiles if using QuantileLoss
        raw_predictions, x_val = best_tft.predict(val_dataloader, mode="raw", return_x=True)

        # Plotting the first example from the validation set
        fig, ax = plt.subplots(figsize=(10, 6)) # Create a figure and axes
        best_tft.plot_prediction(x_val, raw_predictions, idx=0, add_loss_to_title=True, ax=ax)
        ax.set_title(f"{crypto_symbol} TFT: Actual vs Predicted (Validation Example Seq 0)")
        plt.tight_layout()
        plt.show()

        # --- Optional: Calculate overall validation metrics ---
        # You can iterate through the dataloader to get all actuals and predictions
        # if you want overall metrics like MAE/MSE for the point forecast.
        actuals_list = []
        predicted_list = []
        with torch.no_grad(): # Ensure no gradients are calculated
            best_tft.eval() # Set model to evaluation mode
            for x, y in iter(val_dataloader):
                # y[0] contains the target sequence
                actuals_list.append(y[0])
                # Predict returns the median prediction by default with QuantileLoss
                preds = best_tft(x)["prediction"]
                predicted_list.append(preds)

        if actuals_list:
            actuals_all = torch.cat(actuals_list, dim=0).squeeze().numpy()
            predicted_all = torch.cat(predicted_list, dim=0).squeeze().numpy()

            # Ensure shapes match (might need adjustment based on exact output)
            min_len = min(len(actuals_all), len(predicted_all))
            actuals_all = actuals_all[:min_len]
            predicted_all = predicted_all[:min_len]

            val_mse = mean_squared_error(actuals_all, predicted_all)
            val_mae = mean_absolute_error(actuals_all, predicted_all)
            print("\n--- Overall Validation Set Metrics (Point Forecast) ---")
            print(f" Validation MSE: {val_mse:.6f}")
            print(f" Validation MAE: {val_mae:.6f}")

            # --- Optional: Simple Line Plot (like XGBoost) ---
            # Need the corresponding dates for the validation predictions
            # The 'time_idx' for validation predictions starts after 'training_cutoff'
            val_start_idx = training_cutoff + 1
            val_end_idx = validation_dataset.data["data"]["time_idx"].max() # Use the dataset created earlier

            # Get the date index corresponding to the validation period time_idx
            # We need to be careful here as the actuals/predicted might not cover the *entire* validation range if batches were limited
            # Let's use the index from the predictions DataFrame if available from predict() with return_index=True
            if 'time_idx' in predictions.index.names:
                 val_indices = predictions.index.get_level_values('time_idx')
                 # Find corresponding dates in the original dataframe
                 val_dates = data_featured[data_featured['time_idx'].isin(val_indices)].index

                 # Adjust lengths if necessary
                 plot_len = min(len(val_dates), len(actuals_all), len(predicted_all))
                 val_dates = val_dates[-plot_len:] # Plot most recent part if lengths differ
                 actuals_plot = actuals_all[-plot_len:]
                 predicted_plot = predicted_all[-plot_len:]

                 plt.figure(figsize=(12, 6))
                 plt.plot(val_dates, actuals_plot, label='Actual Log Return', alpha=0.8)
                 plt.plot(val_dates, predicted_plot, label='Predicted Log Return (TFT)', alpha=0.8, linestyle='--')
                 plt.title(f"{crypto_symbol} TFT: Actual vs Predicted Log Returns (Validation Set)")
                 plt.xlabel("Date")
                 plt.ylabel("Log Return")
                 plt.legend()
                 plt.tight_layout()
                 plt.show()
            else:
                 print("Could not extract time_idx from prediction index for plotting.")


    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {best_model_path}")
    except Exception as e:
        print(f"An error occurred while loading model or predicting: {e}")
else:
    print("Could not find the best model checkpoint. Skipping prediction.")


print("\nTransformer trainer script finished.") # Move this line to the very end