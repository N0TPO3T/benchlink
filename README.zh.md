# BenchLink

**统一评测框架：通过标准观察字典解耦机器人模型与评测基准。**

BenchLink 提供了一套标准化接口，让**任意机器人模型**（动作策略、触觉编码器）可以对接**任意评测基准**（仿真环境、离线数据集、线性探针）。核心思想：定义一个固定的 `CanonicalObs` 数据契约，把每个模型包装成 **Adapter**，每个基准包装成 **Runner**，通过注册中心连接两者。

## 快速开始

```bash
pip install benchlink

# 查看已注册组件
benchlink --list
```

### 零依赖 Demo

```bash
PYTHONPATH=. python examples/dummy_adapter.py
PYTHONPATH=. python examples/dummy_runner.py
```

### 真实评测

```bash
# 动作模型 -> 仿真基准
benchlink --model fastwam --benchmark libero \
  --checkpoint /path/to/weights.ckpt \
  --config configs/libero.yaml --episodes 10

# 触觉编码器 -> 线性探针
benchlink --model anytouch --benchmark anytouch_probe \
  --checkpoint /path/to/encoder.pt \
  --config configs/anytouch.yaml
```

## 架构

```
+---------------+   CanonicalObs    +---------------+
| Benchmark     | ----------------> | ModelAdapter  |
| Runner        |   rgb_static,     |   .act()      |
|               |   proprio,        |               |
|  .evaluate()  |   language, ...   |   -> action   |
|               | <---------------- |   (7,)        |
+---------------+      action       +---------------+
```

标准动作空间: `(7,)` → `[dx, dy, dz, droll, dpitch, dyaw, gripper]`

## 已注册组件

- **8 个动作模型**: rdp, fastwam, dp, openpi, motus, dreamzero, rdt, vla_touch
- **5 个触觉编码器**: anytouch, sparsh, t3, vla_touch, unitac_ecf
- **8 个评测基准**: libero, maniskill, droid_sim, manifeel_sim, manifeel, robotwin, anytouch_probe, unitac_ecf

## 扩展

新增一个模型：实现 `ModelAdapter` 的三个方法（`load`、`act`、`reset`），调用 `register_model()` 即可。
新增一个基准：实现 `BenchmarkRunner` 的四个方法（`setup`、`evaluate`、`_to_canonical`、`_from_canonical`），调用 `register_benchmark()` 即可。

## 文档

- 中文设计文档: [docs/zh/](docs/zh/)
- 英文文档: [docs/](docs/)

## 许可证

Apache 2.0 — 详见 [LICENSE](LICENSE)。
