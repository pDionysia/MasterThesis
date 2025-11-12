import pandas as pd
import matplotlib.pyplot as plt
import os
import re
'''
# Experimental data part 1
# Root path to my data
root_folder = "data/first_experimental_data"

# Get all subfolders (15_10_2025, 17_10_2025, 21_10_2025)
subfolders = [
    os.path.join(root_folder, f)
    for f in os.listdir(root_folder)
    if os.path.isdir(os.path.join(root_folder, f))
]

for folder in subfolders:
    substance_name = os.path.basename(folder)
    print(f"\n📁 Processing folder: {substance_name}")

    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    groups = {}
    for file in csv_files:
        match = re.search(r"(reference|ethanol|[A-Za-z]+_\d+ppb)", file)
        key = match.group(1) if match else "unknown"
        groups.setdefault(key, []).append(file)

    for conc, files in groups.items():
        plt.figure(figsize=(10, 5))

        for file in sorted(files):
            path = os.path.join(folder, file)

            try:
                data = pd.read_csv(path)
                # Clean up column names
                data.columns = [c.strip() for c in data.columns]

                # Check available columns
                if "Time_abs/ps" not in data.columns or "Signal/nA" not in data.columns:
                    print(f"⚠️ Skipping {file} — columns found: {data.columns.tolist()}")
                    continue

                plt.plot(
                    data["Time_abs/ps"],
                    data["Signal/nA"],
                    linewidth=0.6,
                    alpha=0.7,
                    label=file.replace(".csv", "")
                )

            except Exception as e:
                print(f"Error reading {file}: {e}")

        plt.title(f"{substance_name} – {conc}")
        plt.xlabel("Time (ps)")
        plt.ylabel("Signal (nA)")
        plt.legend(fontsize=7, ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


'''
# I Choose my chemical folder
folder = "data/first_experimental_data/15_10_2025"

plt.figure(figsize=(12, 6))

# Concentrations for Slide 7
target_concs = [
    "ethanol",
    "10ppb",
    "100ppb",
    "1000ppb",
    "10000ppb",
    "100000ppb"
]

# Assign one strong color per concentration
colors = {
    "ethanol":     "blue",
    "10ppb":       "green",
    "100ppb":      "red",
    "1000ppb":     "purple",
    "10000ppb":    "orange",
    "100000ppb":   "black"
}

for conc in target_concs:
    files = [f for f in os.listdir(folder) if conc in f and f.endswith(".csv")]

    signals = []

    for file in sorted(files):
        path = os.path.join(folder, file)
        data = pd.read_csv(path)

        # Clean column names
        data.columns = [c.strip() for c in data.columns]

        time = data["Time_abs/ps"]
        signal = data["Signal/nA"]

        # SUPER THIN raw lines in strong color
        plt.plot(time, signal, linewidth=0.2, color=colors[conc])

        signals.append(signal)

    # Mean line (strong color, slightly thicker)
    if signals:
        mean_signal = pd.concat(signals, axis=1).mean(axis=1)
        plt.plot(time, mean_signal, linewidth=1.2, color=colors[conc], label=conc)

plt.title("Chemical Measurements on Si — Increasing Concentrations for Metalaxyl")
plt.xlabel("Time (ps)")
plt.ylabel("Signal (nA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Comparison between different chemicals

root = "data/first_experimental_data"

# Chemicals and their folders
folders = {
    "Ethanol":       "15_10_2025",  
    "Metalaxyl":     "15_10_2025",
    "Acetamiprid":   "17_10_2025",
    "Abamectin":     "21_10_2025"
}

# Pick one concentration per chemical for the comparison
representative = {
    "Ethanol": "ethanol",
    "Metalaxyl": "1000ppb",
    "Acetamiprid": "1000ppb",
    "Abamectin": "1000ppb"
}

plt.figure(figsize=(10, 5))

for chem, subfolder in folders.items():
    folder = os.path.join(root, subfolder)
    keyword = representative[chem]

    files = [f for f in os.listdir(folder) if keyword in f and f.endswith(".csv")]

    signals = []

    for file in sorted(files):
        path = os.path.join(folder, file)
        data = pd.read_csv(path)
        data.columns = [c.strip() for c in data.columns]

        time = data["Time_abs/ps"]
        signals.append(data["Signal/nA"])

    # Mean waveform for the chemical
    if signals:
        mean_signal = pd.concat(signals, axis=1).mean(axis=1)
        plt.plot(time, mean_signal, linewidth=2, label=f"{chem} ({keyword})")

plt.title("Comparison Between Chemicals")
plt.xlabel("Time (ps)")
plt.ylabel("Signal (nA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Experimental data part 2
# Path to one measurement file
file_path = "data/Si_wafer_clear_SRRs/x_direction_sample_1_1.csv"

# Load the data
df = pd.read_csv(file_path)

# Quick look at the first rows
print(df.head())

# Check columns and data types
print(df.info())


'''
Διαδρομή φακέλου split-ring resonators (SRRs), μετρήσεις οι οποίες αντιστοιχούν
 σε κάθε μία δομή που έχει το Si wafer πριν βάλουμε οποιοδήποτε χημικό πάνω:

'''
folder = "data/Si_wafer_clear_SRRs"

# Λίστα όλων των CSV αρχείων
files = [f for f in os.listdir(folder) if f.endswith(".csv")]

plt.figure(figsize=(12, 6))

for file in sorted(files):
    path = os.path.join(folder, file)
    
    
    data = pd.read_csv(path)
    
    # οι δύο πρώτες στήλες είναι time και signal
    time = data.iloc[:, 0]
    signal = data.iloc[:, 1]
    
    plt.plot(time, signal, alpha=0.6, linewidth=0.3, label=file.replace('.csv', ''))

plt.title("THz Time Domain Signals (Si wafer clear SRRs)")
plt.xlabel("Time (ps)")
plt.ylabel("Signal (nA)")
plt.legend(fontsize=6, loc='upper right', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

'''
Διαδρομή φακέλου "Si_wafer_with_ethanol" όπου έχουμε εναποθέσει 600μL αιθανόλης. 

'''
folder2 = "data/Si_wafer_with_ethanol"

# Λίστα όλων των CSV αρχείων
files2 = [f for f in os.listdir(folder2) if f.endswith(".csv")]

plt.figure(figsize=(12, 6))

for file in sorted(files2):
    path2 = os.path.join(folder2, file)
    
    
    data2 = pd.read_csv(path2)
    
    # oι δύο πρώτες στήλες είναι time και signal
    time = data2.iloc[:, 0]
    signal = data2.iloc[:, 1]
    
    plt.plot(time, signal, alpha=0.6, linewidth=0.3, label=file.replace('.csv', ''))

plt.title("THz Time Domain Signals (Si wafer with Ethanol)")
plt.xlabel("Time (ps)")
plt.ylabel("Signal (nA)")
plt.legend(fontsize=6, loc='upper right', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

