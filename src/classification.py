import os
import numpy as np
import pandas as pd

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
Xclear, yclear = load_folder("data/Si_wafer_clear_SRRs", label=0)
Xeth, yeth = load_folder("data/Si_wafer_with_ethanol", label=1)
X10, y10 = load_folder("data/SRRs_with_10ppb_Acetamiprid", label=2)
X100, y100 = load_folder("data/SRRs_with_100ppb_Acetamiprid", label=3)
X1000, y1000 = load_folder("data/SRRs_with_1000ppb_Acetamiprid", label=4)

# Combine them
X = np.array(X10 + X100 + X1000 + Xeth + Xclear, dtype=object)
y = np.array(y10 + y100 + y1000 + yeth + yclear)