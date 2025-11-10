import pandas as pd
import matplotlib.pyplot as plt
import os

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
    
    plt.plot(time, signal, alpha=0.6, label=file.replace('.csv', ''))

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
    
    plt.plot(time, signal, alpha=0.6, label=file.replace('.csv', ''))

plt.title("THz Time Domain Signals (Si wafer with Ethanol)")
plt.xlabel("Time (ps)")
plt.ylabel("Signal (nA)")
plt.legend(fontsize=6, loc='upper right', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

