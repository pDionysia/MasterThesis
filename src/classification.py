import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from aeon.classification.convolution_based import RocketClassifier
from tsai.all import Learner
from tsai.all import InceptionTime
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.models.OmniScaleCNN import OmniScaleCNN
from tsai.models.ResNet import ResNet
from tsai.models.XceptionTime import XceptionTime
from tsai.all import TSDataLoaders
from tsai.all import get_splits
from tsai.all import accuracy
import torch

def load_folder(folder_path, label):
    X_list = []
    y_list = []

    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, fname))
            df.columns = [c.strip() for c in df.columns]

            signal = df["Signal/nA"].values

            # reshape to (1, series_length) for Aeon
            X_list.append(signal.reshape(1, -1))
            y_list.append(label)

    return X_list, y_list


# Load each class
Xclear, yclear = load_folder(r"C:\Dev\MasterThesis\data\Si_wafer_clear_SRRs", label=0)
Xeth,   yeth   = load_folder(r"C:\Dev\MasterThesis\data\SRRs_cleaned_with_ethanol", label=1)
X10,    y10    = load_folder(r"C:\Dev\MasterThesis\data\SRRs_with_10ppb_Acetamiprid", label=2)
X100,   y100   = load_folder(r"C:\Dev\MasterThesis\data\SRRs_with_100ppb_Acetamiprid", label=3)
X1000,  y1000  = load_folder(r"C:\Dev\MasterThesis\data\SRRs_with_1000ppb_Acetamiprid", label=4)


# Combine them
X = np.array(Xclear + Xeth + X10 + X100 + X1000, dtype=object)
y = np.array(yclear + yeth + y10 + y100 + y1000)

# Define n_samples and series_length NOW
n_samples = len(X)
print("n_samples:", n_samples)

series_length = max(ts.shape[1] for ts in X)
print("series_length:", series_length)

# ------------------------------
# Pad to uniform length
# ------------------------------

X_padded = np.zeros((n_samples, 1, series_length))

for i, ts in enumerate(X):
    length = ts.shape[1]
    X_padded[i, 0, :length] = ts  # fill into padded array

print("X_padded shape:", X_padded.shape)
print("y shape:", len(y))


'''
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42, stratify=y
)

# run the baseline classifier
clf = RocketClassifier(n_kernels=10000)
clf.fit(X_train, y_train)
print("ROCKET Accuracy:", clf.score(X_test, y_test))

# run Inception time
# InceptionTime model
# X: (400, 1, 4001)
# y: (400,)
def main():
    X = X_padded.astype(np.float32)

    # If wrong orientation, swap axes
    if X.shape[1] == series_length:
        X = np.swapaxes(X, 1, 2)

    print("Final X shape:", X.shape)  # MUST be (400, 1, 4001)

    y_int = y.astype(int)

    splits = get_splits(y_int, valid_size=0.2, shuffle=True)

    # dataloaders
    dls = TSDataLoaders.from_numpy(X, y_int, splits=splits, bs=32, num_workers=0)

    # create my model
    model = InceptionTime(
        c_in = 1,   # num channels = 1
        c_out=dls.c      # num classes
    )

    # learner
    learn = Learner(
        dls,
        model,
        metrics=[accuracy]
    )

    # train
    learn.fit_one_cycle(20, 1e-3)

    # evaluate
    learn.show_results()
    preds, targs = learn.get_preds()
    print("Finished training.")

if __name__ == "__main__":
    main()
'''
# test InceptionTimePlus, OmniScaleCNN, ResNet, XceptionTime
# --------------------------------------------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------------------------------------------

def load_folder(folder_path, label):
    X_list = []
    y_list = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, fname))
            df.columns = [c.strip() for c in df.columns]
            signal = df["Signal/nA"].values
            X_list.append(signal.reshape(1, -1))
            y_list.append(label)
    return X_list, y_list


# Update these paths if needed
DATA_PATHS = {
    0: r"C:\Dev\MasterThesis\data\Si_wafer_clear_SRRs",
    1: r"C:\Dev\MasterThesis\data\SRRs_cleaned_with_ethanol",
    2: r"C:\Dev\MasterThesis\data\SRRs_with_10ppb_Acetamiprid",
    3: r"C:\Dev\MasterThesis\data\SRRs_with_100ppb_Acetamiprid",
    4: r"C:\Dev\MasterThesis\data\SRRs_with_1000ppb_Acetamiprid"
}

# Load all classes
X = []
y = []
for label, path in DATA_PATHS.items():
    Xi, yi = load_folder(path, label)
    X += Xi
    y += yi

X = np.array(X, dtype=object)   # list of (1, series_len)
y = np.array(y, dtype=int)

print("Samples:", len(X))


# --------------------------------------------------------------------------------------
# 2. PADDING TO UNIFORM LENGTH
# --------------------------------------------------------------------------------------

series_length = max(ts.shape[1] for ts in X)
print("Max series length:", series_length)

X_padded = np.zeros((len(X), 1, series_length))
for i, ts in enumerate(X):
    L = ts.shape[1]
    X_padded[i, 0, :L] = ts

print("Padded shape:", X_padded.shape)


# --------------------------------------------------------------------------------------
# 3. PREPARE DATA FOR TSAI (N,C,L → N,L,C)
# --------------------------------------------------------------------------------------

X_tsai = np.swapaxes(X_padded, 1, 2).astype(np.float32)  # -> (N, L, C)
y_tsai = y.copy()

splits = get_splits(y_tsai, valid_size=0.2, shuffle=True)

dls = TSDataLoaders.from_numpy(X_tsai, y_tsai, splits=splits,
                               bs=32, num_workers=0)

c_in = 1   # number of channels = 1
c_out = len(np.unique(y_tsai))


# --------------------------------------------------------------------------------------
# 4. FUNCTION TO TRAIN + EVALUATE A MODEL
# --------------------------------------------------------------------------------------

def train_model(model_cls, model_name, **kwargs):
    print("\n" + "=" * 70)
    print(f"Training {model_name}...")
    print("=" * 70)

    if model_name == "InceptionTimePlus":
        X_model = X_padded.astype(np.float32)     # (N,1,L)
    else:
        X_model = X_tsai                          # (N,L,1)

    splits = get_splits(y_tsai, valid_size=0.2, shuffle=True)

    dls = TSDataLoaders.from_numpy(
        X_model, y_tsai, splits=splits,
        bs=32, num_workers=0
    )

    model = model_cls(c_in=c_in, c_out=c_out, **kwargs)

    learn = Learner(dls, model, metrics=accuracy)
    learn.fit_one_cycle(20, 1e-3)

    valid_loss = learn.recorder.values[-1][1]
    valid_acc = learn.recorder.values[-1][2]

    print(f"\n{model_name} RESULTS:")
    print(f"Validation Loss = {valid_loss:.4f}")
    print(f"Validation Accuracy = {valid_acc:.4f}")

    preds, targs = learn.get_preds()

    return {
        "name": model_name,
        "loss": float(valid_loss),
        "acc": float(valid_acc)
    }


# --------------------------------------------------------------------------------------
# 5. RUN ALL FOUR MODELS
# --------------------------------------------------------------------------------------

results = []

results.append(train_model(InceptionTimePlus, "InceptionTimePlus"))
results.append(
    train_model(
        OmniScaleCNN,
        "OmniScaleCNN",
        seq_len=series_length
    )
)

results.append(train_model(ResNet, "ResNet"))
results.append(train_model(XceptionTime, "XceptionTime"))


# --------------------------------------------------------------------------------------
# 6. PRINT SUMMARY TABLE
# --------------------------------------------------------------------------------------

print("\n\n=================== FINAL COMPARISON ===================")
print("Model Name            | Accuracy | Loss")
print("-------------------------------------------------------")
for r in results:
    print(f"{r['name']:22} | {r['acc']:.4f}   | {r['loss']:.4f}")
print("=======================================================\n")

