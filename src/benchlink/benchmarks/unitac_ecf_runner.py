#!/usr/bin/env python3
"""
UniTacECFRunner — multi-dataset tactile evaluation executor.

Evaluation sets:
  - TAG:     tactile attribute classification (12 material attributes)
  - OF1:     optical flow classification (3 contact modes)
  - OF2:     optical flow classification (5 contact modes)
  - Feel:    tactile emotion classification (4 emotions)
  - VitacLab: robot tactile scene classification (6 tasks)
  - FOTA:    tactile object classification (10 objects)

Each subset: image + label → model.encode() → linear probe / KNN → accuracy

Usage:
    runner = UniTacECFRunner()
    runner.setup({"data_root": "/path/to/unitac_ecf_data"})
    results = runner.evaluate(model)
    # → {"tag": 0.85, "of1": 0.72, "feel": 0.91, "vitaclab": 0.65, "mean": 0.78}
"""

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from benchlink.base import BenchmarkRunner, TactileAdapter


class UniTacECFRunner(BenchmarkRunner):
    """UniTacECF multi-dataset tactile evaluation executor."""

    # Default subset list: (name, n_classes, description)
    DEFAULT_SUBSETS = {
        "tag":     {"n_classes": 12, "desc": "tactile attribute classification"},
        "of1":     {"n_classes": 3,  "desc": "optical flow classification (3-class)"},
        "of2":     {"n_classes": 5,  "desc": "optical flow classification (5-class)"},
        "feel":    {"n_classes": 4,  "desc": "tactile emotion classification"},
        "vitaclab": {"n_classes": 6, "desc": "robot tactile scene classification"},
        "fota":    {"n_classes": 10, "desc": "tactile object classification"},
    }

    def __init__(self):
        super().__init__()
        self.data_root: Optional[Path] = None
        self._datasets: dict = {}       # subset_name → [(img, label), ...]
        self._subsets: dict = {}        # subset config
        self._test_ratio: float = 0.3

    def setup(self, config: dict) -> None:
        """Initialize evaluation configuration.

        Required config fields:
            data_root:  UniTacECF dataset root directory
        Optional config fields:
            subsets:    enabled subset list (default all)
            test_ratio: test set ratio (default 0.3)
            seed:       random seed (default 42)
        """
        self.data_root = Path(config.get("data_root", ""))
        if not self.data_root or not self.data_root.exists():
            raise FileNotFoundError(
                f"UniTacECF data root not found: {self.data_root}. "
                f"Please set 'data_root' in config."
            )

        self._test_ratio = config.get("test_ratio", 0.3)
        seed = config.get("seed", 42)
        np.random.seed(seed)

        # Enabled subsets
        enabled = config.get("subsets", list(self.DEFAULT_SUBSETS.keys()))
        self._subsets = {k: v for k, v in self.DEFAULT_SUBSETS.items() if k in enabled}

        # Load data
        self._datasets = {}
        for name in self._subsets:
            data_path = self.data_root / name
            if data_path.exists():
                self._datasets[name] = self._load_subset(data_path)
                print(f"[UniTacECFRunner] Loaded '{name}': {len(self._datasets[name])} samples")
            else:
                print(f"[UniTacECFRunner] Warning: subset '{name}' not found at {data_path}, skipping")

        self.config = config

    def evaluate(
        self,
        model: TactileAdapter,
        n_episodes: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run tactile evaluation on all subsets.

        Args:
            model: loaded TactileAdapter

        Returns:
            dict: {subset_name: accuracy, "mean": float, "n_subsets": int}
        """
        results = {}

        for name, subset_cfg in self._subsets.items():
            data = self._datasets.get(name, [])
            if len(data) < 10:
                results[name] = -1.0
                continue

            accuracy = self._evaluate_subset(model, data, subset_cfg["n_classes"])
            results[name] = accuracy
            print(f"[UniTacECFRunner] {name}: accuracy = {accuracy:.4f}")

        # Compute average
        valid = [v for v in results.values() if v >= 0]
        results["mean"] = float(np.mean(valid)) if valid else -1.0
        results["n_subsets"] = len(valid)

        return results

    # ── Internal methods ──

    def _load_subset(self, data_path: Path) -> list:
        """Load a single subset.

        Supports two formats:
          1. images/ + labels.txt (each line: image_filename label_id)
          2. .npz ({"images": (N,H,W,3), "labels": (N,)})
        """
        # Try .npz
        npz_files = list(data_path.glob("*.npz"))
        if npz_files:
            data = np.load(npz_files[0])
            images = data["images"]
            labels = data["labels"]
            return [(images[i], int(labels[i])) for i in range(len(images))]

        # Try images/ + labels.txt
        img_dir = data_path / "images"
        labels_file = data_path / "labels.txt"
        if img_dir.exists() and labels_file.exists():
            samples = []
            for line in labels_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_file = img_dir / parts[0]
                    label = int(parts[1])
                    if img_file.exists():
                        import cv2
                        img = cv2.imread(str(img_file))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        samples.append((img, label))
            return samples

        # Try subdirectories under images/ named by label
        subdirs = sorted([d for d in img_dir.iterdir() if d.is_dir()]) if img_dir.exists() else []
        if subdirs:
            samples = []
            for label, subdir in enumerate(subdirs):
                for img_file in sorted(subdir.glob("*.*"))[:200]:  # max 200 per class
                    import cv2
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        samples.append((img, label))
            return samples

        return []

    def _evaluate_subset(
        self,
        model: TactileAdapter,
        data: list,
        n_classes: int,
    ) -> float:
        """Linear probe evaluation: train a logistic regression classifier on encoded features."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import train_test_split

        # Encode all samples
        images, labels = zip(*data)
        labels = np.array(labels)

        # Encode in batches to prevent OOM
        features = []
        batch_size = 64
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_feats = [model.encode(img) for img in batch]
            features.extend(batch_feats)
        features = np.array(features)

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=self._test_ratio, random_state=42,
            stratify=labels,
        )

        # Train linear classifier
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=1000, multi_class="auto", C=1.0,
        ))
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)

        return accuracy

    def _to_canonical(self, raw_obs: Any) -> None:
        raise NotImplementedError("UniTacECFRunner does not use _to_canonical()")

    def _from_canonical(self, action: np.ndarray) -> None:
        raise NotImplementedError("UniTacECFRunner does not use _from_canonical()")

    def close(self) -> None:
        """Clean up."""
        self._datasets.clear()
