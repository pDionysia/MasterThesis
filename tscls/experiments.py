"""
Base classes and structures for running and comparing time series experiments.

Provides base classes and structures for running and comparing time series
classification model experiments using the tsai library and scikit-learn.
Standardizes experiment workflow, results, and validation splitting to
enable reproducible research and fair model comparisons.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss
from tsai.all import Learner, TSDataLoaders, accuracy, get_splits


@dataclass
class ExperimentResult:
    """
    Container for results of a model training run.

    Stores the model name, accuracy, loss, and the trained model object
    (e.g. a tsai Learner or aeon/sklearn Estimator).

    Attributes
    ----------
    name : str
        Identifier for the model or experiment.
    accuracy : float
        Classification accuracy on the validation or test set.
    loss : float
        Loss value (e.g., log loss) on the validation or test set.
    model : Any
        Reference to the trained model instance (tsai Learner, sklearn estimator, etc.).
    """

    name: str
    accuracy: float
    loss: float
    model: Any  # Can be a tsai Learner or an aeon/sklearn Estimator


@dataclass(kw_only=True)
class Experiment:
    """
    Base experiment class for fair model comparison with reproducible validation splits.

    Handles random shuffling, stratified splitting of input data, and provides a
    tabular summary report of model results. Intended to be subclassed for
    experiment logic for different model libraries (e.g., tsai, aeon). All splits
    are created in a reproducible, stratified manner to ensure consistency
    across models.

    Attributes
    ----------
    X : np.ndarray
        Array of features or time series data to be split and passed to models.
    y : np.ndarray
        Target class labels corresponding to each sample in X.
    valid_size : float, default=0.2
        Fraction of data to use for validation set.
    shuffle : bool, default=True
        Whether to randomly shuffle samples before splitting.
    random_state : int, default=42
        Seed for reproducibility of the data splitting.
    splits : tuple
        Tuple containing train and validation indices produced by `get_splits`.

    Methods
    -------
    print_summary(results)
        Print a formatted comparison table for a list of ExperimentResult objects.
    """

    X: np.ndarray
    y: np.ndarray
    valid_size: float = 0.2
    shuffle: bool = True
    random_state: int = 42
    splits: tuple = field(init=False)

    def __post_init__(self):
        """
        Initialize the experiment by generating reproducible, stratified splits.

        Splits the input dataset into training and validation indices using
        stratified sampling. Ensures splits are reproducible and balanced per
        class, controlled by the random seed, validation size, and shuffle
        parameters.

        Returns
        -------
        None
            The splits are stored in the `self.splits` attribute as a tuple of
            index arrays.
        """
        self.splits = get_splits(
            self.y,
            valid_size=self.valid_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
            stratify=True,
            show_plot=False,
        )

    def print_summary(self, results: list[ExperimentResult]) -> None:
        """
        Print a comparison table for a list of results.

        Parameters
        ----------
        results : list[ExperimentResult]
            List of ExperimentResult objects to display in the comparison table.
        """
        print("\n\n" + "=" * 55)
        print(f"{'FINAL COMPARISON':^55}")
        print("=" * 55)
        print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Loss':<10}")
        print("-" * 55)

        sorted_results = sorted(results, key=lambda r: r.accuracy, reverse=True)

        for r in sorted_results:
            loss_str = f"{r.loss:.4f}" if not np.isnan(r.loss) else "N/A"
            print(f"{r.name:<25} | {r.accuracy:.4f}     | {loss_str:<10}")
        print("=" * 55 + "\n")


@dataclass(kw_only=True)
class AeonExperiment(Experiment):
    """
    Experiment manager for training and evaluating Aeon, Sktime, or Scikit-learn models.

    This dataclass provides convenient utilities for training classical
    (non-deep learning) models on time series datasets. It uses the
    train/validation split provided by
    the base `Experiment` class and supports any model compatible with the sklearn API.

    Attributes
    ----------
    X : np.ndarray
        Time series data, typically of shape (n_samples, ...), ready for classical ML.
    y : np.ndarray
        Target labels for each sample.
    valid_size : float, default=0.2
        Proportion of the dataset to include in the validation split.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    random_state : int, default=42
        Random seed for reproducibility.
    splits : tuple of np.ndarray (init=False)
        Tuple of training and validation indices, set by the parent Experiment.

    Methods
    -------
    train(model_cls, model_name, **kwargs)
        Fit the specified model using training data, compute accuracy and log-loss
        (when available) on the validation set, and return the fitted model and metrics.
    """

    def train(
        self, model_cls: type[Any], model_name: str, **kwargs
    ) -> ExperimentResult:
        """
        Train and evaluate a classical ML model using the given data split.

        This method fits the specified model class on the training set, evaluates
        accuracy and log-loss (if available) on the validation set, and returns
        an ExperimentResult containing metrics and the fitted model.

        Parameters
        ----------
        model_cls : Type[Any]
            The scikit-learn, Aeon, or Sktime-compatible model class to instantiate.
        model_name : str
            A descriptive name for the model, used for reporting.
        **kwargs
            Additional keyword arguments passed to the model's constructor.

        Returns
        -------
        ExperimentResult
            Object containing the name, accuracy, loss, and fitted model.

        Notes
        -----
        If the model class supports probability predictions via `predict_proba`,
        log-loss will also be computed; otherwise, it will be NaN.
        """
        print(f"\nTraining {model_name} (Aeon)...")
        print("-" * 40)

        # 1. Prepare Data (Convert indices to Train/Test sets)
        train_idx, test_idx = self.splits[0], self.splits[1]

        X_train, y_train = self.X[train_idx], self.y[train_idx]
        X_test, y_test = self.X[test_idx], self.y[test_idx]

        # 2. Instantiate Model
        model = model_cls(**kwargs)

        # 3. Fit
        model.fit(X_train, y_train)

        # 4. Evaluate
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        loss = float("nan")
        try:
            # Only calculate loss if the model supports probabilities
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)
                loss = log_loss(y_test, probs)
        except (ValueError, AttributeError) as e:
            # Ignore loss calculation errors (common with some metrics/models)
            print(f"Warning: Could not calculate log loss: {e}")

        print(f"Done. Acc: {acc:.4f}")

        return ExperimentResult(name=model_name, accuracy=acc, loss=loss, model=model)


@dataclass(kw_only=True)
class TsaiExperiment(Experiment):
    """
    Experiment manager for training and evaluating Tsai/FastAI/PyTorch models.

    Attributes
    ----------
    batch_size : int, default=32
        Batch size used by the TSDataLoaders during training and validation.
    num_workers : int, default=0
        Number of worker processes to use for data loading.
    device : torch.device, default=None
        Device on which to train the model. Defaults to CUDA if available,
        otherwise CPU.
    _c_in : int
        Number of input channels in the feature data. Set during __post_init__.
    _c_out : int
        Number of output classes for classification. Set during __post_init__.
    _seq_len : int
        Sequence length of the input data. Set during __post_init__.

    Methods
    -------
    train(model_cls, model_name, epochs=20, lr=1e-3, **kwargs) -> ExperimentResult
        Trains the provided model class using Tsai's TSDataLoaders and Learner,
        then returns an ExperimentResult containing accuracy, loss, and the
        fitted model.

    Notes
    -----
    This dataclass expects the input data (`self.X`, `self.y`) to be compatible
    with PyTorch (float32, shape: [batch, channels, length]). Cross-validation
    splits should be provided via `self.splits`. The model class must follow the
    Tsai/FastAI API.
    """

    batch_size: int = 32
    num_workers: int = 0
    device: torch.device = field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Internal fields calculated in post_init
    _c_in: int = field(init=False)
    _c_out: int = field(init=False)
    _seq_len: int = field(init=False)

    def __post_init__(self):
        """
        Post-initialization for TsaiExperiment.

        This method calls the parent __post_init__ to generate cross-validation splits,
        ensures the feature data `self.X` is of dtype float32 for PyTorch compatibility,
        and extracts important shape-related properties:
        the number of input channels (`_c_in`), the sequence length (`_seq_len`),
        and the number of output classes (`_c_out`).

        Notes
        -----
        - Assumes `self.X` has shape (batch, channels, length).
        - Assumes `self.y` is a label array compatible with np.unique.
        """
        # 1. Call Parent to generate self.splits
        super().__post_init__()

        # 2. Enforce Float32 for PyTorch
        self.X = self.X.astype(np.float32)

        # 3. Extract Data Properties
        # Assumes X shape: (Batch, Channels, Length)
        self._c_in = self.X.shape[1]
        self._seq_len = self.X.shape[2]
        self._c_out = len(np.unique(self.y))

    def train(
        self,
        model_cls: type[Any],
        model_name: str,
        epochs: int = 20,
        lr: float = 1e-3,
        **kwargs,
    ) -> ExperimentResult:
        """
        Train a model using the provided model class and experiment settings.

        Uses Tsai's TSDataLoaders and Learner to train a time series
        classification model. The method prepares data loaders, instantiates
        the model, performs training for a fixed number of epochs, and then
        evaluates performance metrics.

        Parameters
        ----------
        model_cls : Type[Any]
            The model class to instantiate. Must follow the Tsai/FastAI model API, and
            accept c_in and c_out as input arguments.
        model_name : str
            Name for this experiment/model (for reporting).
        epochs : int, default=20
            Number of training epochs to perform.
        lr : float, default=1e-3
            Initial learning rate for training.
        **kwargs
            Additional keyword arguments to pass to the model's constructor.
            If 'seq_len' is not given, it is automatically added.

        Returns
        -------
        ExperimentResult
            An ExperimentResult object containing the model name, final accuracy, loss,
            and the trained learner object.

        Notes
        -----
        Assumes self.X and self.y are prepared and PyTorch compatible, and self.splits
        are defined for cross-validation.
        """
        print(f"\nTraining {model_name} (Tsai)...")
        print("-" * 40)

        # 1. Create DataLoaders
        dls = TSDataLoaders.from_numpy(
            self.X,
            self.y,
            splits=self.splits,
            bs=self.batch_size,
            num_workers=self.num_workers,
            device=self.device,
        )

        # 2. Inject dimensions into kwargs if not provided
        if "seq_len" not in kwargs:
            kwargs["seq_len"] = self._seq_len

        # 3. Instantiate Model
        try:
            model = model_cls(c_in=self._c_in, c_out=self._c_out, **kwargs)
        except TypeError:
            # Fallback for models that don't accept seq_len
            kwargs.pop("seq_len", None)
            model = model_cls(c_in=self._c_in, c_out=self._c_out, **kwargs)

        # 4. Train
        learn = Learner(dls, model, metrics=accuracy)
        learn.fit_one_cycle(epochs, lr)

        # 5. Evaluate
        val_results = learn.validate()
        val_loss = val_results[0]
        val_acc = val_results[1]

        print(f"Done. Acc: {val_acc:.4f} | Loss: {val_loss:.4f}")

        return ExperimentResult(
            name=model_name, accuracy=val_acc, loss=val_loss, model=learn
        )
