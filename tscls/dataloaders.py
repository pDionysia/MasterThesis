import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from tsai.all import check_data


@dataclass
class TimeSeriesLoader:
    """
    Loads, processes, and pads time series data from CSV files for usage
    with libraries like Aeon or sktime.
    """
    data_paths: dict[int, Path | list[Path]]
    target_column: str = "Signal/nA"
    file_pattern: str = "*.csv"
    
    # Internal state storage
    _X_padded: np.ndarray | None = field(default=None, init=False)
    _y: np.ndarray | None = field(default=None, init=False)

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Orchestrates loading, reshaping, and padding of signals.
        Returns (X_padded, y).
        """
        raw_signals: list[np.ndarray] = []
        labels: list[int] = []

        print(f"Loading data from {len(self.data_paths)} categories...")
        
        # 1. Load all data into a standard Python list
        for label, paths_input in self.data_paths.items():
            
            if isinstance(paths_input, list):
                folder_list = paths_input
            else:
                folder_list = [paths_input]
            
            for folder_path in folder_list:
                signals, lbls = self._load_folder(Path(folder_path), label)
                raw_signals.extend(signals)
                labels.extend(lbls)

        if not raw_signals:
            raise ValueError("No data found in any of the provided paths.")

        # 2. Get dimensions directly from the list (No intermediate object array)
        n_samples = len(raw_signals)
        max_length = max(ts.shape[1] for ts in raw_signals)
        
        print(f"Processing complete: {n_samples} samples found.")
        print(f"Max series length: {max_length}")

        # 3. Pre-allocate the final array
        # Shape: (n_samples, 1, max_length)
        self._X_padded = np.zeros((n_samples, 1, max_length), dtype=float)
        self._y = np.array(labels, dtype=int)

        # 4. Fill the array (Ragged -> Dense)
        for i, ts in enumerate(raw_signals):
            length = ts.shape[1]
            self._X_padded[i, 0, :length] = ts

        return self._X_padded, self._y

    def _load_folder(
        self, folder_path: Path,
        label: int
    ) -> tuple[list[np.ndarray], list[int]]:
        """Helper to process a single folder."""
        signals: list[np.ndarray] = []
        lbls: list[int] = []
        
        if not folder_path.exists():
            print(f"Warning: Path not found: {folder_path}")
            return [], []

        files: list[Path] = sorted(folder_path.glob(self.file_pattern))
        
        for fname in files:
            try:
                df = pd.read_csv(fname)
                df.columns = [c.strip() for c in df.columns]
                
                if self.target_column not in df.columns:
                    raise KeyError(
                        f"Column '{self.target_column}' not found in {fname.name}"
                    ) from None

                signal = df[self.target_column].values
                signals.append(signal.reshape(1, -1))
                lbls.append(label)
            except Exception as e:
                print(f"Error reading {fname.name}: {e}")

        return signals, lbls
    
    def report(self) -> None:
        """
        Prints a statistical report and runs tsai.check_data.
        """
        if self._X_padded is None or self._y is None:
            print("Data not loaded yet. Please run .load() first.")
            return

        print("\n" + "="*40)
        print("DATASET REPORT")
        print("="*40)
        
        # 1. Basic Dimensions
        n_samples, n_channels, length = self._X_padded.shape
        print(f"Total Samples:    {n_samples}")
        print(f"Series Length:    {length}")
        print(f"Channels:         {n_channels}")
        print("-" * 40)

        # 2. Class Distribution
        unique, counts = np.unique(self._y, return_counts=True)
        total = sum(counts)
        print("Class Distribution:")
        print(f"{'Label':<10} {'Count':<10} {'Percentage':<10}")
        for label, count in zip(unique, counts):
            percent = (count / total) * 100
            print(f"{label:<10} {count:<10} {percent:.1f}%")
        
        ratio = max(counts) / min(counts)
        if ratio > 2.0:
            print(f"\n[!] WARNING: High Class Imbalance (Ratio {ratio:.1f}:1)")

        print("-" * 40)
        
        # 3. TSAI Check Data Integration
        print("Running tsai.check_data()...\n")
        try:
            # We cast to float32 specifically for tsai check because it's picky
            check_data(self._X_padded.astype(np.float32), self._y.astype(int))
        except Exception as e:
            print(f"[!] tsai check failed: {e}")
            print("    (Ensure tsai is installed and imported correctly)")

        print("="*40 + "\n")

    def __len__(self) -> int:
        """Returns the number of samples."""
        if self._y is None:
            return 0
        return len(self._y)

    @property
    def X(self) -> np.ndarray:
        if self._X_padded is None:
            raise ValueError("Data not loaded. Call .load() first.")
        return self._X_padded

    @property
    def y(self) -> np.ndarray:
        if self._y is None:
            raise ValueError("Data not loaded. Call .load() first.")
        return self._y