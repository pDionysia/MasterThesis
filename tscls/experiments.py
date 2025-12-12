import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Type
from sklearn.metrics import accuracy_score, log_loss
from tsai.all import Learner, TSDataLoaders, accuracy, get_splits
import torch

@dataclass
class ExperimentResult:
    """Standardized output for any model training run."""
    name: str
    accuracy: float
    loss: float
    model: Any  # Can be a tsai Learner or an aeon/sklearn Estimator


@dataclass(kw_only=True)
class Experiment:
    """
    Base class that handles data splitting and reporting.
    Ensures all derived experiments use the exact same validation splits.
    """
    X: np.ndarray
    y: np.ndarray
    valid_size: float = 0.2
    shuffle: bool = True
    random_state: int = 42
    splits: tuple = field(init=False)

    def __post_init__(self):
        self.splits = get_splits(
            self.y,
            valid_size=self.valid_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
            stratify=True,
            show_plot=False,
        )

    def print_summary(self, results: list[ExperimentResult]) -> None:
        """Prints a comparison table for a list of results."""
        print("\n\n" + "="*55)
        print(f"{'FINAL COMPARISON':^55}")
        print("="*55)
        print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Loss':<10}")
        print("-" * 55)
        
        sorted_results = sorted(results, key=lambda r: r.accuracy, reverse=True)
        
        for r in sorted_results:
            loss_str = f"{r.loss:.4f}" if not np.isnan(r.loss) else "N/A"
            print(f"{r.name:<25} | {r.accuracy:.4f}     | {loss_str:<10}")
        print("="*55 + "\n")


@dataclass(kw_only=True)
class AeonExperiment(Experiment):
    """
    Manages training for Aeon/Sktime/Sklearn models.
    """
    
    def train(self, model_cls: Type[Any], model_name: str, **kwargs) -> ExperimentResult:
        print(f"\nTraining {model_name} (Aeon)...")
        print("-" * 40)

        # 1. Prepare Data (Convert indices to Train/Test sets)
        train_idx, test_idx = self.splits[0], self.splits[1]
        
        X_train, y_train = self.X[train_idx], self.y[train_idx]
        X_test,  y_test  = self.X[test_idx],  self.y[test_idx]

        # 2. Instantiate Model
        model = model_cls(**kwargs)

        # 3. Fit
        model.fit(X_train, y_train)

        # 4. Evaluate
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        loss = float('nan')
        try:
            # Only calculate loss if the model supports probabilities
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)
                loss = log_loss(y_test, probs)
        except Exception:
            pass # Ignore loss calculation errors (common with some metrics/models)

        print(f"Done. Acc: {acc:.4f}")

        return ExperimentResult(
            name=model_name,
            accuracy=acc,
            loss=loss,
            model=model
        )


@dataclass(kw_only=True)
class TsaiExperiment(Experiment):
    """
    Manages training for Tsai/FastAI/PyTorch models.
    """
    batch_size: int = 32
    num_workers: int = 0
    device: torch.device = field(
        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Internal fields calculated in post_init
    _c_in: int = field(init=False)
    _c_out: int = field(init=False)
    _seq_len: int = field(init=False)

    def __post_init__(self):
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
        model_cls: Type[Any], 
        model_name: str, 
        epochs: int = 20, 
        lr: float = 1e-3, 
        **kwargs
    ) -> ExperimentResult:
        
        print(f"\nTraining {model_name} (Tsai)...")
        print("-" * 40)

        # 1. Create DataLoaders
        dls = TSDataLoaders.from_numpy(
            self.X, 
            self.y, 
            splits=self.splits, 
            bs=self.batch_size, 
            num_workers=self.num_workers,
            device=self.device
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
            name=model_name,
            accuracy=val_acc,
            loss=val_loss,
            model=learn
        )