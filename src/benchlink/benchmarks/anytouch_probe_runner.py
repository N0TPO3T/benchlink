#!/usr/bin/env python3
"""
AnyTouch Probe → BenchmarkRunner.

AnyTouch probe evaluation is not a simulation environment, but a tactile representation quality benchmark.
4 subtasks:
  - TAG:  tactile material classification (12 classes)
  - OF1:  object classification — same sensor (10 classes)
  - OF2:  object classification — cross-sensor (10 classes)
  - Feel: tactile perception classification (6 classes)

Evaluation flow:
  Freeze tactile encoder → extract features for all samples per subtask
  → train linear classification head (LogisticRegression) → report accuracy

Usage:
    runner = AnyTouchProbeRunner()
    runner.setup({"data_root": "/path/to/anytouch_benchmark"})
    results = runner.evaluate(model)  # model must have encode()
    # → {"tag": 0.85, "of1": 0.72, "of2": 0.68, "feel": 0.91}
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings

import imageio.v3 as iio
import numpy as np

from benchlink.base import BenchmarkRunner
from benchlink.schema import CanonicalObs


class AnyTouchProbeRunner(BenchmarkRunner):
    """AnyTouch probe evaluation — freeze encoder → linear probe → accuracy.

    Note: evaluate() expects the model to be a TactileAdapter (with encode() method).
    If a ModelAdapter (only act()) is passed, pair it with a TactileAdapter for feature extraction.
    """

    def __init__(self):
        super().__init__()
        self.data_root: Optional[Path] = None
        self.img_size: int = 224
        self._datasets: Dict[str, List[Tuple[np.ndarray, int]]] = {}
        self._label_names: Dict[str, List[str]] = {}

    def setup(self, config: dict) -> None:
        """Load probe evaluation dataset.

        Required config fields:
            data_root:  AnyTouch benchmark dataset root directory
        Optional config fields:
            img_size:   input image size (default 224)
            subsets:    evaluation subset list, default all ["tag", "of1", "of2", "feel"]
            seed:       random seed (default 42)
        """
        self.data_root = Path(config["data_root"])
        if not self.data_root.exists():
            raise FileNotFoundError(f"AnyTouch probe data not found: {self.data_root}")

        self.img_size = config.get("img_size", 224)
        subsets = config.get("subsets", ["tag", "of1", "of2", "feel"])

        # ── Load probe dataset ──
        for subset in subsets:
            images, labels = self._load_subset(subset)
            self._datasets[subset] = list(zip(images, labels))
            print(f"[AnyTouchProbe] Loaded '{subset}': {len(images)} samples")

        self.config = config

    def evaluate(
        self,
        model,
        n_episodes: int = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run probe evaluation.

        Args:
            model: must implement encode(tactile_img) → feature method
                   can be a TactileAdapter or a ModelAdapter with encode()

        Returns:
            dict: {subset_name: accuracy, ..., "mean": float}
        """
        # ── Verify model has encode method ──
        if not hasattr(model, "encode") or not callable(model.encode):
            raise TypeError(
                "AnyTouchProbeRunner requires a model with encode(tactile_img) method. "
                "Use a TactileAdapter (e.g., AnyTouchAdapter) or a ModelAdapter that also "
                "implements encode()."
            )

        results = {}
        all_accuracies = []

        for subset_name, samples in self._datasets.items():
            # ── Extract features ──
            features = []
            labels = []
            for tactile_img, label in samples:
                feat = model.encode(tactile_img)
                features.append(feat)
                labels.append(label)

            feat_arr = np.array(features)  # (N, D)
            label_arr = np.array(labels)   # (N,)

            # ── Train linear probe ──
            accuracy = self._train_linear_probe(feat_arr, label_arr)
            results[subset_name] = accuracy
            all_accuracies.append(accuracy)

            print(f"[AnyTouchProbe] {subset_name}: accuracy = {accuracy:.4f}")

        # ── Average accuracy ──
        if all_accuracies:
            results["mean"] = float(np.mean(all_accuracies))

        return results

    # ── Internal methods ──

    def _load_subset(self, subset: str) -> Tuple[List[np.ndarray], List[int]]:
        """Load a single probe subset.

        Expected dataset structure:
            {data_root}/{subset}/images/   — tactile images
            {data_root}/{subset}/labels.txt — each line: image_name label_id
        """
        subset_dir = self.data_root / subset
        images = []
        labels = []

        # Read label index
        label_file = subset_dir / "labels.txt"
        if not label_file.exists():
            # Fallback: read subdirectory structure under images/
            return self._load_from_subdirs(subset_dir)

        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                img_name, label_id = parts[0], int(parts[1])
                img_path = subset_dir / "images" / img_name
                if img_path.exists():
                    img = iio.imread(str(img_path))
                    images.append(img)
                    labels.append(label_id)

        return images, labels

    def _load_from_subdirs(self, subset_dir: Path) -> Tuple[List[np.ndarray], List[int]]:
        """Load from subdirectory structure: {subset_dir}/{class_name}/*.png"""

        images = []
        labels = []
        class_dirs = sorted([d for d in subset_dir.iterdir() if d.is_dir()])

        for label_id, class_dir in enumerate(class_dirs):
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    img = iio.imread(str(img_file))
                    images.append(img)
                    labels.append(label_id)

        return images, labels

    @staticmethod
    def _train_linear_probe(
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.3,
    ) -> float:
        """Train a Linear Probe (LogisticRegression) and return accuracy.

        Freeze the encoder (no training), only train the classification head.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import accuracy_score

        # Stratified sampling
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Logistic Regression with standardization
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                random_state=42,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        return float(accuracy_score(y_test, y_pred))

    def _to_canonical(self, raw_obs: Any) -> CanonicalObs:
        """Not used in probe mode — placeholder for ABC compliance."""
        raise NotImplementedError("AnyTouchProbeRunner does not use _to_canonical()")

    def _from_canonical(self, action: np.ndarray) -> Any:
        """Not used in probe mode — placeholder for ABC compliance."""
        raise NotImplementedError("AnyTouchProbeRunner does not use _from_canonical()")
