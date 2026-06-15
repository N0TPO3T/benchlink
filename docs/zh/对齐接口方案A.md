# 对齐接口方案A

## 背景

项目目录下有 13 个 baseline（Motus、FastWAM、OpenPi、RDT、RDP、DP、DreamZero、VLA-Touch、AnyTouch、Sparsh、T3、UniTac_ECF、TouchWAM）和 8 个 benchmark（LIBERO、RoboTwin 2.0、AnyTouch、ManiFeel、UniVTAC、UniTac_ECF、DROID Sim、ManiSkill）。

目标是让任意 baseline 能跑任意 benchmark，而不是每个模型只跑自己的评测。

---

## 一、设计理念

### 三层结构

```
           标准观察字典 (obs dict)
                 ↕
  BenchmarkRunner  ←→  ModelAdapter
  (每个benchmark一个)   (每个baseline一个)
```

- **标准观察字典**：Runner 和 Adapter 之间传数据的固定契约
- **BenchmarkRunner**：负责跟仿真环境交互，做格式转换
- **ModelAdapter**：负责加载模型、做推理、做格式转换

### 数据流（以 FastWAM 跑 LIBERO 为例）

```
Runner 驱动整个循环:

  ① env.step() → LIBERO 原生观察
     (agentview_image, robot0_joint_pos, ...)

  ② Runner._to_canonical(原生观察) → 标准观察字典
     {
         "rgb_static":   agentview_image,
         "rgb_gripper":  eye_in_hand_image,
         "proprio":      robot0_joint_pos,
         "language":     "open the middle drawer",
         "tactile_img":  None,     ← LIBERO 没有触觉
         "tactile_feat": None,
     }

  ③ model.act(标准观察字典) → 标准动作
     FastWAMAdapter.act():
         - 取 rgb_static 和 language
         - 做 FastWAM 特有的预处理 (resize, normalize)
         - 调用 FastWAM 模型推理
         - 把输出映射到标准动作 space
         → 返回 ndarray shape=(7,)

  ④ Runner._from_canonical(标准动作) → LIBERO 原生动作
     标准动作是末端 delta pose, LIBERO 吃 joint position
     做一次转换 → joint target

  ⑤ env.step(原生动作) → 下一帧观察
```

整个过程就是一个翻译链：**benchmark原生 → 标准 → 模型原生 → 标准 → benchmark原生**。

---

## 二、标准观察字典

### 定义

Runner 和 Adapter 之间的数据契约。所有字段要么有值，要么 None。

```python
obs = {
    # === 视觉 ===
    "rgb_static":   np.ndarray | None,   # shape=(H, W, 3), 固定视角 RGB
    "rgb_gripper":  np.ndarray | None,   # shape=(H, W, 3), 腕部相机 RGB
    "depth":        np.ndarray | None,   # shape=(H, W), 深度图

    # === 机器人状态 ===
    "proprio":      np.ndarray | None,   # shape=(7,), 关节位置或末端位姿

    # === 触觉 ===
    "tactile_img":  np.ndarray | None,   # shape=(H, W, 3), 触觉图像
    "tactile_feat": np.ndarray | None,   # shape=(64,), 预提取的触觉特征

    # === 语言 ===
    "language":     str | None,           # 任务描述文本
    "lang_embed":   np.ndarray | None,   # shape=(512,), 预计算指令嵌入
}
```

### 标准动作

```python
action: np.ndarray  # shape=(7,)
# [dx, dy, dz, droll, dpitch, dyaw, gripper]
# 前 6 维：末端执行器的位姿增量（delta pose）
# 第 7 维：夹爪开合 [0, 1]
```

### 使用规则

- Runner 在 `_to_canonical()` 中把环境原生的字段填进来，没有的字段填 None
- Adapter 在 `act()` 中只取自己需要的字段，不依赖的字段不取
- 如果模型需要的字段值全是 None（例如 RDP 需要 tactile_img 但 benchmark 是 LIBERO 没有触觉），由 Adapter 决定怎么处理（报错/ fallback/ zero padding）

---

## 三、BenchmarkRunner

### 抽象基类

```python
class BenchmarkRunner:
    """每个 benchmark 一个 Runner，负责封装仿真环境"""

    def setup(self, config: dict):
        """
        初始化 benchmark 环境。
        参数 config 包含环境配置（任务名、数据路径等）。
        """

    def evaluate(self, model: ModelAdapter, n_episodes: int = 10) -> dict:
        """
        跑 N 个 episode，返回评测指标。
        返回值格式：{"metric_name": float, ...}
        """

    def _to_canonical(self, raw_obs: dict) -> dict:
        """
        benchmark 原生观察 → 标准观察字典。
        每个 benchmark 的格式不同，这个方法做映射。
        """

    def _from_canonical(self, action: np.ndarray) -> np.ndarray:
        """
        标准动作 → benchmark 原生动作。
        每个 benchmark 的动作空间不同，这个方法做转换。
        """
```

### 具体示例：LiberoRunner

```python
class LiberoRunner(BenchmarkRunner):
    def setup(self, config):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.env = OffScreenRenderEnv(
            task_embedding_path=get_libero_path("task_emb") + "/" + config["task_name"],
            task_name=config["task_name"],
        )
        self.task_name = config["task_name"]

    def evaluate(self, model, n_episodes=10):
        successes = []

        for ep in range(n_episodes):
            raw_obs, _ = self.env.reset()
            model.reset()
            done = False

            while not done:
                # 转成标准观察字典
                canonical_obs = self._to_canonical(raw_obs)

                # 模型推理
                action = model.act(canonical_obs)

                # 转回环境原生动作
                env_action = self._from_canonical(action)

                # 环境 step
                raw_obs, reward, done, trunc, info = self.env.step(env_action)

            successes.append(info.get("success", 0))

        return {"success_rate": np.mean(successes)}

    def _to_canonical(self, raw_obs):
        return {
            "rgb_static":   raw_obs["agentview_image"],
            "rgb_gripper":  raw_obs["robot0_eye_in_hand_image"],
            "depth":        None,
            "proprio":      raw_obs["robot0_joint_pos"],
            "tactile_img":  None,           # LIBERO 无触觉
            "tactile_feat": None,
            "language":     self.task_name,
            "lang_embed":   self.env.get_task_embedding(),
        }

    def _from_canonical(self, action):
        # 标准动作: delta_pose [dx, dy, dz, droll, dpitch, dyaw, gripper]
        # LIBERO 吃关节位置
        current_joints = self.env._env.get_robot0_joint_positions()
        # 简单实现：delta_pose → 逆运动学 → 关节目标
        joint_target = self._delta_pose_to_joint(current_joints, action[:6])
        joint_target[-1] = action[6]  # gripper
        return joint_target
```

### 具体示例：AnyTouchRunner

```python
class AnyTouchRunner(BenchmarkRunner):
    """
    AnyTouch benchmark 是触觉 probe 评测，不是仿真环境。
    所以 evaluate 逻辑不同：用 probe 数据集的图像，调 model.encode()，训练分类头。
    """

    def setup(self, config):
        # 加载 AnyTouch probe 数据集 (TAG, OF1, OF2, Feel)
        self.datasets = {
            "tag":  AnyTouchDataset("tag"),
            "of1":  AnyTouchDataset("of1"),
            "of2":  AnyTouchDataset("of2"),
            "feel": AnyTouchDataset("feel"),
        }

    def evaluate(self, model, n_episodes=None):
        results = {}
        for name, dataset in self.datasets.items():
            # 对每个子任务：用模型提取所有样本的特征
            features = []
            labels = []
            for tactile_img, label in dataset:
                feat = model.encode(tactile_img)  # 触觉编码器接口
                features.append(feat)
                labels.append(label)

            # 训练线性分类头（冻住编码器）
            accuracy = self._train_linear_probe(features, labels)
            results[name] = accuracy

        return results
```

---

## 四、ModelAdapter

### 抽象基类

```python
class ModelAdapter:
    """每个 baseline 一个 Adapter，负责封装模型推理"""

    def load(self, checkpoint: str, config: dict):
        """
        加载模型权重。
        checkpoint: 权重路径
        config: 配置（设备、推理参数等）
        """

    def act(self, obs: dict) -> np.ndarray:
        """
        接收标准观察字典，返回标准动作。
        内部做：取字段 → 模型预处理 → 模型推理 → 映射到标准动作空间
        """

    def reset(self):
        """
        重置模型内部状态。
        对 Transformer/RNN 类模型需要清空 history buffer。
        """
```

### 触觉编码器基类

```python
class TactileAdapter:
    """触觉编码器模型使用这个接口（不是 act，是 encode）"""

    def load(self, checkpoint: str, config: dict):
        pass

    def encode(self, tactile_img: np.ndarray) -> np.ndarray:
        """触觉图像 → 特征向量"""
```

### 具体示例：FastWAMAdapter

```python
class FastWAMAdapter(ModelAdapter):
    def load(self, checkpoint, config):
        import torch
        from fastwam import FastWAMModel

        self.device = config.get("device", "cuda")
        self.model = FastWAMModel.load_from_checkpoint(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # 预处理管线（FastWAM 特有的）
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def act(self, obs):
        # 1. 从标准观察字典取需要的字段
        rgb = obs["rgb_static"]             # (H, W, 3)
        instr = obs["language"]             # str

        # 2. 模型特有的预处理
        img_tensor = self.transform(rgb)     # → (3, 224, 224)
        img_tensor = img_tensor.unsqueeze(0) # → (1, 3, 224, 224)

        # 3. 模型推理
        import torch
        with torch.no_grad():
            # FastWAM 输出 shape=(50, 7) action chunk
            raw_action = self.model.infer(
                image=img_tensor.to(self.device),
                instruction=[instr],
            )

        # 4. 映射到标准动作空间
        #    FastWAM 输出关节位置绝对值，标准动作要 delta_pose
        first_step = raw_action[0].cpu().numpy()  # (7,)
        current_joints = obs["proprio"]
        delta = first_step[:6] - current_joints[:6]
        gripper = first_step[6]

        return np.concatenate([delta, [gripper]])

    def reset(self):
        self.model.reset_history()
```

### 具体示例：AnyTouchAdapter

```python
class AnyTouchAdapter(TactileAdapter):
    def load(self, checkpoint, config):
        from anytouch import AnyTouchEncoder
        self.model = AnyTouchEncoder(checkpoint)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(...),
        ])

    def encode(self, tactile_img):
        # 标准观察字典里 tactile_img 就是 np.ndarray (H, W, 3)
        # 直接喂给 AnyTouch
        img = self.transform(tactile_img).unsqueeze(0)
        with torch.no_grad():
            feat = self.model(img)
        return feat.squeeze(0).numpy()
```

---

## 五、CLI 入口

```python
# cli.py
import argparse
from registry import get_runner, get_adapter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)        # 模型名称
    parser.add_argument("--benchmark", required=True)    # benchmark 名称
    parser.add_argument("--checkpoint", required=True)   # 权重路径
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    # 从注册中心取对应的类
    adapter_cls = get_adapter(args.model)        # "fastwam" → FastWAMAdapter
    runner_cls = get_runner(args.benchmark)      # "libero" → LiberoRunner

    cfg = load_config(args.config)

    # 初始化
    model = adapter_cls()
    model.load(args.checkpoint, cfg)

    runner = runner_cls()
    runner.setup(cfg)

    # 跑评测
    result = runner.evaluate(model, n_episodes=args.episodes)
    print(result)

if __name__ == "__main__":
    main()
```

### 使用示例

```bash
# FastWAM 跑 LIBERO
python cli.py --model fastwam --benchmark libero --checkpoint /ckpt/fastwam.pt

# FastWAM 跑 RoboTwin（同一个模型，换 benchmark）
python cli.py --model fastwam --benchmark robotwin --checkpoint /ckpt/fastwam.pt

# Motus 跑 LIBERO（换模型，同 benchmark）
python cli.py --model motus --benchmark libero --checkpoint /ckpt/motus.ckpt

# AnyTouch 跑 AnyTouch probe
python cli.py --model anytouch --benchmark anytouch_probe --checkpoint /ckpt/anytouch.pth

# Sparsh 跑 AnyTouch probe（不同触觉模型，同 benchmark）
python cli.py --model sparsh --benchmark anytouch_probe --checkpoint /ckpt/sparsh.pth
```

---

## 六、首批实现计划（P0）

### 目标

打通两条线，验证接口设计：

| 线 | Runner | Adapter | 验证什么 |
|----|--------|---------|---------|
| 动作线 | LiberoRunner | FastWAMAdapter | model.act(obs) → action 流程 |
| 触觉线 | AnyTouchRunner | AnyTouchAdapter | model.encode(tactile) → feature 流程 |

### 文件列表

```
eval_framework/
├── schema.py              # 标准观察字典定义
├── base.py                # 基类：ModelAdapter, TactileAdapter, BenchmarkRunner
├── registry.py            # 注册中心：get_adapter(), get_runner()
├── cli.py                 # 命令行入口
├── models/
│   ├── __init__.py
│   └── fastwam_adapter.py # FastWAM 封装
├── benchmarks/
│   ├── __init__.py
│   ├── libero_runner.py   # LIBERO 评测
│   └── anytouch_runner.py # AnyTouch probe 评测
└── configs/
    └── default.yaml       # 默认配置
```

### 工作量预估

| 文件 | 行数 | 内容 |
|------|------|------|
| `schema.py` | 30 | 定义 CanonicalObs 的 key 和格式说明 |
| `base.py` | 40 | 三个基类 + 注释文档 |
| `registry.py` | 30 | 注册 dict + get 函数 |
| `cli.py` | 60 | 参数解析 + 调用流程 |
| `libero_runner.py` | 150 | LIBERO 环境初始化 + evaluate + 格式转换 |
| `anytouch_runner.py` | 120 | AnyTouch probe 数据集加载 + linear probe 评测 |
| `fastwam_adapter.py` | 150 | FastWAM 模型加载 + 预处理 + 推理 + 动作映射 |
| `anytouch_adapter.py` | 80 | AnyTouch 编码器加载 + 预处理 + encode |
| 合计 | ~660 | 首批 P0 代码量 |

---

## 七、如何扩展

### 加一个新模型（如 Motus）

只需要写一个 `motus_adapter.py`，不需要动任何 Runner：

```python
class MotusAdapter(ModelAdapter):
    def load(self, checkpoint, config):
        # Motus 在 Docker 里，通过 API 调用
        ...

    def act(self, obs):
        # 从标准观察字典取图像和指令
        # Motus 吃图像序列（需要 history）
        # Motus 输出 action chunk (50, 7)
        # 取第一帧映射到标准动作
        ...

    def reset(self):
        self.history.clear()
```

然后直接跑：`cli.py --model motus --benchmark libero ...`

### 加一个新 benchmark（如 ManiFeel）

只需要写一个 `manifeel_runner.py`，不需要动任何 Adapter：

```python
class ManiFeelRunner(BenchmarkRunner):
    def setup(self, config):
        # 配置 IsaacGym + TacSL 环境

    def evaluate(self, model, n_episodes):
        # env 出触觉图像 → 填到 tactile_img 字段
        # model 有 tactile 就用，没有就 fallback

    def _to_canonical(self, raw_obs):
        return {
            "rgb_static":   raw_obs["rgb"],
            "tactile_img":  raw_obs["tactile"],   # ManiFeel 特有的
            "proprio":      raw_obs["ee_pose"],
            "language":     raw_obs["task_desc"],
            ...
        }
```

---

## 八、与其他方案的关系

| 方案 | 区别 |
|------|------|
| **方案A（当前）** | 每个模型写一个 Adapter，每个 benchmark 一个 Runner，通过标准字典对接 |
| 后续可能优化 | 共享动作空间转换逻辑（多个 Adapter 可能写重复的 delta_pose→joint 转换） |
| 后续可能优化 | 运行时校验（Adapter 需要的字段 benchmark 没有时提前报错） |

当前方案先做 P0，等遇到重复劳动再优化。
