# hybrid_m5_baseline_v1.py
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "./datasets"
HISTORY = 180
HORIZON = 28
STRIDE = 28
VAL_SPLIT_DAYS = 200
BATCH_SIZE = 1024
EPOCHS = 100
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# LOAD RAW CSVS
# -----------------------------
print("Loading CSV files...")
sales = pd.read_csv(os.path.join(DATA_DIR, "sales_train_validation.csv"))
calendar = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
prices = pd.read_csv(os.path.join(DATA_DIR, "sell_prices.csv"))

# all day columns named d_*
d_cols = [c for c in sales.columns if c.startswith("d_")]
sales_values = sales[d_cols].values.astype(np.float32)   # shape (n_series, n_days)
n_series, n_days = sales_values.shape
print(f"sales_values: {sales_values.shape}")

# -----------------------------
# ENCODE IDS & SIMPLE WEIGHTS
# -----------------------------
print("Encoding IDs and computing simple series weights...")
from sklearn.preprocessing import LabelEncoder
item_encoder = LabelEncoder()
store_encoder = LabelEncoder()

sales["item_id_enc"] = item_encoder.fit_transform(sales["item_id"])
sales["store_id_enc"] = store_encoder.fit_transform(sales["store_id"])

item_ids = sales["item_id_enc"].values.astype(np.int32)
store_ids = sales["store_id_enc"].values.astype(np.int32)

n_items = int(item_ids.max() + 1)
n_stores = int(store_ids.max() + 1)

# Simple proxy for WRMSSE importance: total historical volume (normalized)
series_weight = sales_values.sum(axis=1) + 1e-6
series_weight = series_weight / series_weight.mean()
series_weight = series_weight.astype(np.float32)

# -----------------------------
# TRAIN / VALIDATION TIME CUT
# -----------------------------
train_cutoff = n_days - VAL_SPLIT_DAYS
print(f"n_days={n_days}, train_cutoff index={train_cutoff}")

# Precompute how many train windows per series to derive steps_per_epoch
starts = np.arange(0, n_days - (HISTORY + HORIZON) + 1, STRIDE)
end_positions = starts + HISTORY
train_mask = end_positions < train_cutoff
num_train_windows_per_series = int(train_mask.sum())
train_sample_count = n_series * num_train_windows_per_series
print("Estimated train windows per series:", num_train_windows_per_series)
print("Estimated total train windows:", train_sample_count)
steps_per_epoch = max(1, train_sample_count // BATCH_SIZE)
val_sample_count = n_series  # one validation window per series (we build it deterministically)
val_steps = max(1, math.ceil(val_sample_count / BATCH_SIZE))

# -----------------------------
# TF.DATA: train dataset (on-the-fly windows per series)
# -----------------------------
print("Building tf.data training pipeline...")

# Prepare tensors for tf.data (kept on CPU / host memory)
sales_tensor = tf.constant(sales_values)         # shape (n_series, n_days)
item_tensor = tf.constant(item_ids)
store_tensor = tf.constant(store_ids)
weight_tensor = tf.constant(series_weight)

base_ds = tf.data.Dataset.from_tensor_slices((sales_tensor, item_tensor, store_tensor, weight_tensor))

def series_to_windows(series, item, store, weight):
    """
    For one series tensor, create a Dataset of sliding windows (past,HORIZON) 
    where the end index (t) < train_cutoff (training windows only).
    """
    # frames of length HISTORY+HORIZON, stepping by STRIDE
    frames = tf.signal.frame(series, frame_length=HISTORY + HORIZON, frame_step=STRIDE, axis=0)
    # number of frames:
    n_frames = tf.shape(frames)[0]  # frames correspond to start indices 0, STRIDE, 2*STRIDE, ...
    # start indices for each frame
    starts_local = tf.cast(tf.range(n_frames) * STRIDE, tf.int32)
    ends_local = starts_local + HISTORY  # end index t (we predict t..t+HORIZON)
    # mask windows with ends_local < train_cutoff
    mask = tf.less(ends_local, tf.constant(train_cutoff, dtype=tf.int32))
    selected = tf.boolean_mask(frames, mask)
    num_selected = tf.shape(selected)[0]

    # if none selected, return empty Dataset
    def empty_ds():
        return tf.data.Dataset.from_tensor_slices((
            tf.zeros([0, HISTORY], dtype=tf.float32),
            tf.zeros([0], dtype=tf.int32),
            tf.zeros([0], dtype=tf.int32),
            tf.zeros([0, HORIZON], dtype=tf.float32),
            tf.zeros([0], dtype=tf.float32),
        )).take(0)

    def filled_ds():
        past = selected[:, :HISTORY]                       # (num_selected, HISTORY)
        future = selected[:, HISTORY:]                     # (num_selected, HORIZON)
        items = tf.repeat(item, num_selected)              # (num_selected,)
        stores = tf.repeat(store, num_selected)
        weights = tf.repeat(weight, num_selected)
        return tf.data.Dataset.from_tensor_slices((past, items, stores, future, weights))

    return tf.cond(tf.equal(num_selected, 0), empty_ds, filled_ds)

# Flat map over series -> windows
train_windows_ds = base_ds.flat_map(series_to_windows)

# Map to model input shapes: ((past[...,1], item(1,), store(1,)), future, weight)
def to_model_input(past, item, store, future, weight):
    past = tf.expand_dims(past, -1)                 # (HISTORY,1)
    item = tf.expand_dims(tf.cast(item, tf.int32), axis=-1)   # (1,)
    store = tf.expand_dims(tf.cast(store, tf.int32), axis=-1)
    future = tf.cast(future, tf.float32)
    return ({"sales_hist": past, "item_id": item, "store_id": store}, future, weight)

train_ds = (
    train_windows_ds
    .map(to_model_input, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(10000, seed=SEED)
    .batch(BATCH_SIZE)
    .repeat()
    .prefetch(tf.data.AUTOTUNE)
)

# -----------------------------
# TF.DATA: validation dataset (one window per series at train_cutoff)
# -----------------------------
print("Building tf.data validation pipeline...")

# Build validation windows deterministically: past = series[train_cutoff-HISTORY:train_cutoff]
past_val = sales_values[:, train_cutoff - HISTORY: train_cutoff].astype(np.float32)   # (n_series, HISTORY)
future_val = sales_values[:, train_cutoff: train_cutoff + HORIZON].astype(np.float32) # (n_series, HORIZON)
item_val = item_ids.astype(np.int32)
store_val = store_ids.astype(np.int32)
weight_val = series_weight.astype(np.float32)

val_ds = tf.data.Dataset.from_tensor_slices((past_val, item_val, store_val, future_val, weight_val)) \
    .map(lambda past, item, store, fut, w: (
        {"sales_hist": tf.expand_dims(past, -1),
         "item_id": tf.expand_dims(tf.cast(item, tf.int32), -1),
         "store_id": tf.expand_dims(tf.cast(store, tf.int32), -1)},
        fut,
        w
    )) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

# -----------------------------
# BUILD MODEL (multi-horizon)
# -----------------------------
print("Building Keras multi-horizon model...")

sales_input = keras.Input(shape=(HISTORY, 1), name="sales_hist")
item_input = keras.Input(shape=(1,), dtype=tf.int32, name="item_id")
store_input = keras.Input(shape=(1,), dtype=tf.int32, name="store_id")

item_emb = layers.Embedding(input_dim=n_items, output_dim=32)(item_input)
store_emb = layers.Embedding(input_dim=n_stores, output_dim=16)(store_input)
item_emb = layers.Flatten()(item_emb)
store_emb = layers.Flatten()(store_emb)

x = layers.Conv1D(64, 3, padding="causal", activation="relu")(sales_input)
x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
x = layers.GlobalAveragePooling1D()(x)

x = layers.Concatenate()([x, item_emb, store_emb])
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
out = layers.Dense(HORIZON, name="out")(x)

model = keras.Model(inputs=[sales_input, item_input, store_input], outputs=out)
model.compile(optimizer=keras.optimizers.AdamW(1e-3), loss="mse")
model.summary()

callback = [
    callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True,)
]


# -----------------------------
# TRAIN
# -----------------------------
print("Training NN (streaming windows)...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=max(1, steps_per_epoch),
    validation_data=val_ds,
    validation_steps=val_steps,
    callbacks=callback,
    verbose=1
)

# -----------------------------
# PREDICT ON VALIDATION CUT (one window per series)
# -----------------------------
print("Predicting on validation cut (one window per series)...")
nn_pred_val = model.predict(
    [past_val[..., None], item_val.reshape(-1,1), store_val.reshape(-1,1)],
    batch_size=1024,
    verbose=1
)  # (n_series, HORIZON)
y_val = future_val  # (n_series, HORIZON)

baseline_rmse = np.sqrt(mean_squared_error(y_val.ravel(), nn_pred_val.ravel()))
print(f"NN baseline RMSE on validation cut: {baseline_rmse:.6f}")

# -----------------------------
# BUILD GBDT RESIDUAL DATAFRAME (horizon-conditioned)
# -----------------------------
print("Building LightGBM dataset from validation residuals (horizon-conditioned)...")
residuals = (y_val - nn_pred_val).astype(np.float32)

rows = []
for i in range(n_series):
    item_i = int(item_val[i])
    store_i = int(store_val[i])
    w_i = float(weight_val[i])
    for h in range(HORIZON):
        rows.append((item_i, store_i, h, float(nn_pred_val[i,h]), float(residuals[i,h]), w_i))

gbdt_df = pd.DataFrame(rows, columns=["item_id","store_id","horizon","nn_pred","residual","weight"])

# split for GBDT train/val (we're using validation-window residuals for POC)
g_train, g_val = train_test_split(gbdt_df, test_size=0.2, random_state=SEED)

# -----------------------------
# TRAIN LIGHTGBM ON RESIDUALS
# -----------------------------
print("Training LightGBM residual model...")
features = ["item_id","store_id","horizon","nn_pred"]
dtrain = lgb.Dataset(g_train[features], label=g_train["residual"], weight=g_train["weight"],
                     categorical_feature=["item_id","store_id","horizon"])
dval = lgb.Dataset(g_val[features], label=g_val["residual"], weight=g_val["weight"],
                   categorical_feature=["item_id","store_id","horizon"])

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": SEED
}

gbdt = lgb.train(params, dtrain, valid_sets=[dtrain, dval], num_boost_round=800,
                 callbacks=[lgb.early_stopping(stopping_rounds=10)])

# -----------------------------
# HYBRID EVALUATION
# -----------------------------
print("Predicting residuals with LightGBM and evaluating hybrid...")
pred_residuals_flat = gbdt.predict(gbdt_df[features])
pred_residuals = pred_residuals_flat.reshape(n_series, HORIZON)

final_pred = nn_pred_val + pred_residuals
hybrid_rmse = np.sqrt(mean_squared_error(y_val.ravel(), final_pred.ravel()))

print(f"NN baseline RMSE:   {baseline_rmse:.6f}")
print(f"Hybrid   RMSE:      {hybrid_rmse:.6f}")
print(f"Improvement (RMSE): {baseline_rmse - hybrid_rmse:.6f}")

# -----------------------------
# DONE
# -----------------------------
print("Finished. If improvement > 0 we have a working POC.")
