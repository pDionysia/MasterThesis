import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from aeon.classification.convolution_based import RocketClassifier
from tsai.all import Learner
from tsai.all import InceptionTime
from tsai.all import TSDataLoaders
from tsai.all import get_splits
from tsai.all import accuracy


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
'''
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