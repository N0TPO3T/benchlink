#!/usr/bin/env python3
"""
UniTacECFRunner — 多数据集触觉评测执行器。

评测集合:
  - TAG:     触觉属性分类（12 类材料属性）
  - OF1:     光学流分类（3 类接触模式）
  - OF2:     光学流分类（5 类接触模式）
  - Feel:    触觉情感分类（4 类情感）
  - VitacLab: 机器人触觉场景分类（6 类任务）
  - FOTA:    触觉物体分类（10 类物体）

每个子数据集: 图像 + 标签 → model.encode() → 线性 probe / KNN → accuracy

用法:
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
    """UniTacECF 多数据集触觉评测执行器。"""

    # 默认子数据集列表： (name, n_classes, description)
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
        """初始化评测配置。

        config 必需字段:
            data_root:  UniTacECF 数据集根目录
        config 可选字段:
            subsets:    启用的子数据集列表 (默认所有)
            test_ratio: 测试集比例 (默认 0.3)
            seed:       随机种子 (默认 42)
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

        # 启用子数据集
        enabled = config.get("subsets", list(self.DEFAULT_SUBSETS.keys()))
        self._subsets = {k: v for k, v in self.DEFAULT_SUBSETS.items() if k in enabled}

        # 加载数据
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
        """运行所有子数据集的触觉评测。

        Args:
            model: 已加载的 TactileAdapter

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

        # 计算平均
        valid = [v for v in results.values() if v >= 0]
        results["mean"] = float(np.mean(valid)) if valid else -1.0
        results["n_subsets"] = len(valid)

        return results

    # ── 内部方法 ──

    def _load_subset(self, data_path: Path) -> list:
        """加载单个子数据集。

        支持两种格式:
          1. images/ + labels.txt (每行: image_filename label_id)
          2. .npz ({"images": (N,H,W,3), "labels": (N,)})
        """
        # 尝试 .npz
        npz_files = list(data_path.glob("*.npz"))
        if npz_files:
            data = np.load(npz_files[0])
            images = data["images"]
            labels = data["labels"]
            return [(images[i], int(labels[i])) for i in range(len(images))]

        # 尝试 images/ + labels.txt
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

        # 尝试 images/ 下每个子目录以标签命名
        subdirs = sorted([d for d in img_dir.iterdir() if d.is_dir()]) if img_dir.exists() else []
        if subdirs:
            samples = []
            for label, subdir in enumerate(subdirs):
                for img_file in sorted(subdir.glob("*.*"))[:200]:  # 每类最多 200 张
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
        """线性 probe 评测: 用编码特征训练逻辑回归分类器。"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import train_test_split

        # 编码所有样本
        images, labels = zip(*data)
        labels = np.array(labels)

        # 分批编码以防 OOM
        features = []
        batch_size = 64
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_feats = [model.encode(img) for img in batch]
            features.extend(batch_feats)
        features = np.array(features)

        # 划分训练/测试
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=self._test_ratio, random_state=42,
            stratify=labels,
        )

        # 训练线性分类器
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
        """清理。"""
        self._datasets.clear()
