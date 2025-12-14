"""
Provides classes and tools for loading, processing, and analyzing time series data.

Main features include loading from directories, sequence padding, label handling,
spectral (FFT) analysis, and caching. The key class, `TimeSeriesLoader`, manages
loading multi-class time series datasets into numpy arrays, offering pandas
integration and optional FFT computation. Compatible with downstream libraries
such as tsai and PyTorch.
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tsai.all import check_data


@dataclass
class TimeSeriesLoader:
    """
    Loads and processes multi-class time series datasets from disk.

    This class supports sequence padding, label handling, optional FFT
    (frequency spectrum) computation, and efficient caching of parsed arrays.
    Data is organized in folders, each corresponding to a class label.

    Attributes
    ----------
    data_paths : dict[int, Path or list[Path]]
        Mapping from class labels to one or more folders containing .csv files
        for that class.
    target_column : str, default="Signal/nA"
        Name of the column to extract as the signal from each .csv file.
    file_pattern : str, default="*.csv"
        Glob pattern for selecting time series files within each folder.
    X : np.ndarray
        Returns the loaded, padded data array.
    y : np.ndarray
        Returns the loaded label array.

    Methods
    -------
    load(force_reload=False)
        Loads all data, applies sequence padding and returns (X, y) arrays.
    report()
        Prints dataset statistics and performs a tsai.check_data check.
    plot_statistics(fft=False, ...)
        Plots mean and standard deviation bands for each class.
    plot_comparison(fft=False, ...)
        Overlays class means on a single plot.
    plot_samples(fft=False, n_samples=3, ...)
        Plots random raw or FFT-transformed traces per class.
    """

    data_paths: dict[int, Path | list[Path]]
    target_column: str = "Signal/nA"
    file_pattern: str = "*.csv"

    # Internal state storage
    _X_padded: np.ndarray | None = field(default=None, init=False)
    _X_fft: np.ndarray | None = field(default=None, init=False)
    _y: np.ndarray | None = field(default=None, init=False)

    def load(self, force_reload: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Load and process multi-class time series datasets from disk.

        Load all classes' time series, pad them to the maximum length, and
        compile the joined `X`, `y` arrays.

        Parameters
        ----------
        force_reload : bool, default False
            If True, force reload data even if already cached.

        Returns
        -------
        X : np.ndarray
            Array of shape (n_samples, 1, max_seq_len), samples are zero-padded.
        y : np.ndarray
            Array of integer labels for each sample.
        """
        # 1. Check Cache
        if not force_reload and self._X_padded is not None and self._y is not None:
            print("Returning cached data (use force_reload=True to refresh).")
            return self._X_padded, self._y

        # Reset caches
        self._X_padded = None
        self._X_fft = None

        # 2. Start Loading
        raw_signals: list[np.ndarray] = []
        labels: list[int] = []

        print(f"Loading data from {len(self.data_paths)} categories...")

        for label, paths_input in self.data_paths.items():
            folder_list = (
                paths_input if isinstance(paths_input, list) else [paths_input]
            )

            for folder_path in folder_list:
                signals, lbls = self._load_folder(Path(folder_path), label)
                raw_signals.extend(signals)
                labels.extend(lbls)

        if not raw_signals:
            raise ValueError("No data found in any of the provided paths.")

        n_samples = len(raw_signals)
        max_length = max(ts.shape[1] for ts in raw_signals)

        print(f"Processing complete: {n_samples} samples found.")
        print(f"Max series length: {max_length}")

        self._X_padded = np.zeros((n_samples, 1, max_length), dtype=float)
        self._y = np.array(labels, dtype=int)

        for i, ts in enumerate(raw_signals):
            length = ts.shape[1]
            self._X_padded[i, 0, :length] = ts

        return self._X_padded, self._y

    def _get_data_for_plot(self, fft: bool) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Return the main data for plotting, in either raw or FFT space. Caches the FFT.

        Parameters
        ----------
        fft : bool
            If True, returns frequency domain view via rFFT.

        Returns
        -------
        data : np.ndarray
            Array with shape (n_samples, 1, seq_len or freq_bins).
        axis : np.ndarray
            Array of time or frequency values for plotting.
        label : str
            Axis label, either 'Time Step' or 'Frequency'.
        """
        if self._X_padded is None:
            raise ValueError("Load data first.")

        if not fft:
            # Return Time Series
            t = np.arange(self._X_padded.shape[2])
            return self._X_padded, t, "Time Step"

        # Return FFT (Compute & Cache if needed)
        if self._X_fft is None:
            print("Computing FFT cache...")
            self._X_fft = np.abs(np.fft.rfft(self._X_padded, axis=2))

        freqs = np.fft.rfftfreq(self._X_padded.shape[2])
        return self._X_fft, freqs, "Frequency"

    def _load_folder(
        self, folder_path: Path, label: int
    ) -> tuple[list[np.ndarray], list[int]]:
        """
        Load all files in a folder for a specific label.

        Parameters
        ----------
        folder_path : Path
            Path to the folder with signal .csv files.
        label : int
            Integer class label for all files in this folder.

        Returns
        -------
        signals : list of np.ndarray
            Each entry is a (1, seq_len) array (before padding).
        lbls : list of int
            Label repeated for all files found.
        """
        if not folder_path.exists():
            print(f"Warning: Path not found: {folder_path}")
            return [], []

        signals, lbls = [], []
        files = sorted(folder_path.glob(self.file_pattern))

        def _raise_missing_column_error(fname: Path) -> None:
            """
            Raise a KeyError if the required target column is missing in a CSV file.

            Parameters
            ----------
            fname : Path
                Path object representing the CSV file being checked.

            Raises
            ------
            KeyError
                If `self.target_column` is not present in the file's columns.
            """
            raise KeyError(f"Column '{self.target_column}' not found in {fname.name}")

        for fname in files:
            try:
                df = pd.read_csv(fname)
                df.columns = [c.strip() for c in df.columns]

                if self.target_column not in df.columns:
                    _raise_missing_column_error(fname)

                signal = df[self.target_column].values
                signals.append(signal.reshape(1, -1))
                lbls.append(label)
            except (
                KeyError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                OSError,
            ) as e:
                print(f"Error reading {fname.name}: {e}")

        return signals, lbls

    def report(self) -> None:
        """Print a summary of the loaded dataset."""
        if self._X_padded is None or self._y is None:
            print("Data not loaded yet. Please run .load() first.")
            return

        print("\n" + "=" * 40)
        print("DATASET REPORT")
        print("=" * 40)

        n_samples, n_channels, length = self._X_padded.shape
        print(f"Total Samples:    {n_samples}")
        print(f"Series Length:    {length}")
        print(f"Channels:         {n_channels}")
        print("-" * 40)

        unique, counts = np.unique(self._y, return_counts=True)
        total = sum(counts)
        print("Class Distribution:")
        print(f"{'Label':<10} {'Count':<10} {'Percentage':<10}")
        for label, count in zip(unique, counts, strict=False):
            percent = (count / total) * 100
            print(f"{label:<10} {count:<10} {percent:.1f}%")

        print("-" * 40)
        print("Running tsai.check_data()...\n")
        try:
            check_data(self._X_padded.astype(np.float32), self._y.astype(int))
        except (ValueError, TypeError, AttributeError) as e:
            print(f"[!] tsai check failed: {e}")
        print("=" * 40 + "\n")

    def plot_statistics(
        self,
        fft: bool = False,
        figsize: tuple[int, int] = (15, 10),
        sharey: bool = True,
    ) -> None:
        """
        Plot mean and standard deviation bands for each class as separate subplots.

        Parameters
        ----------
        fft : bool, default False
            If True, plots the FFT spectrum; else time series statistics.
        figsize : tuple of int, default (15, 10)
            Overall figure size (width, height).
        sharey : bool, default True
            Share the y-axis across subplots.
        """
        X_data, x_axis, x_label = self._get_data_for_plot(fft)

        unique_labels = np.unique(self._y)
        n_classes = len(unique_labels)
        cols = 3
        rows = int(np.ceil(n_classes / cols))

        fig, axes = plt.subplots(
            rows, cols, figsize=figsize, sharey=sharey, sharex=True
        )
        axes = axes.flatten()

        for i, label in enumerate(unique_labels):
            ax = axes[i]

            class_mask = self._y == label
            X_class = X_data[class_mask, 0, :]

            mu = X_class.mean(axis=0)
            sigma = X_class.std(axis=0)

            color = "red" if fft else "blue"

            ax.plot(x_axis, mu, color=color, label="Mean")
            ax.fill_between(x_axis, mu - sigma, mu + sigma, color=color, alpha=0.3)
            ax.fill_between(
                x_axis, mu - 2 * sigma, mu + 2 * sigma, color=color, alpha=0.1
            )

            ax.set_title(f"Class {label}")
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        title = (
            f"{'FFT Spectrum' if fft else 'Signal Statistics'} ({self.target_column})"
        )
        plt.suptitle(title, fontsize=16)
        plt.xlabel(x_label)
        plt.tight_layout()
        plt.show()

    def plot_comparison(
        self, fft: bool = False, figsize: tuple[int, int] = (12, 6)
    ) -> None:
        """
        Overlay the mean trace of each class on a single plot.

        Parameters
        ----------
        fft : bool, default False
            If True, plots the FFT spectrum means; else time domain means.
        figsize : tuple of int, default (12, 6)
            Figure size.
        """
        X_data, x_axis, x_label = self._get_data_for_plot(fft)

        plt.figure(figsize=figsize)

        unique_labels = np.unique(self._y)
        for label in unique_labels:
            class_mask = self._y == label
            mu = X_data[class_mask, 0, :].mean(axis=0)
            plt.plot(x_axis, mu, label=f"Class {label}")

        plt.title(f"Comparison of Class Means ({'Frequency' if fft else 'Time'})")
        plt.xlabel(x_axis)
        plt.ylabel("Magnitude" if fft else self.target_column)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_samples(
        self,
        fft: bool = False,
        n_samples: int = 3,
        figsize: tuple[int, int] = (15, 10),
        sharey: bool = True,
    ) -> None:
        """
        Plot multiple random samples per class, in time or frequency domain.

        Parameters
        ----------
        fft : bool, default False
            If True, show FFT curves; else show time-domain traces.
        n_samples : int, default 3
            Number of random samples to plot per class.
        figsize : tuple of int, default (15, 10)
            Grid figure size.
        sharey : bool, default True
            Whether y-axis is shared across plots.
        """
        X_data, x_axis, x_label = self._get_data_for_plot(fft)

        unique_labels = np.unique(self._y)
        cols = 3
        rows = int(np.ceil(len(unique_labels) / cols))

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=figsize,
            sharey=sharey,
            sharex=True,
        )
        axes = axes.flatten()

        for i, label in enumerate(unique_labels):
            ax = axes[i]
            class_mask = self._y == label
            X_class = X_data[class_mask, 0, :]

            indices = np.random.choice(
                len(X_class), size=min(n_samples, len(X_class)), replace=False
            )

            for idx in indices:
                ax.plot(x_axis, X_class[idx], alpha=0.7)

            ax.set_title(f"Class {label} - Random Samples")
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        title = f"{'FFT' if fft else 'Raw'} Samples ({self.target_column})"
        plt.suptitle(title, fontsize=16)
        plt.xlabel(x_label)
        plt.tight_layout()
        plt.show()

    @property
    def X(self) -> np.ndarray:  # noqa: N802
        """
        Return the loaded, padded time series array.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 1, seq_len). Raises ValueError if not loaded.
        """
        if self._X_padded is None:
            raise ValueError("Data not loaded. Call .load() first.")
        return self._X_padded

    @property
    def y(self) -> np.ndarray:
        """
        Return the class labels for each loaded sample.

        Returns
        -------
        np.ndarray
            Integer label array. Raises ValueError if not loaded.
        """
        if self._y is None:
            raise ValueError("Data not loaded. Call .load() first.")
        return self._y
