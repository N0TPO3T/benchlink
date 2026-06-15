# 对齐接口方案A — 简洁版

## 目标

让所有 baseline 都能跑任意一个 benchmark。

---

## 设计

```
每个 benchmark 一个 Runner（封装环境）
每个 baseline  一个 Adapter（封装模型）
中间通过标准观察字典对接
```

---

## 三个组件

### ① 标准观察字典

Runner 和 Adapter 之间的数据契约，固定格式：

```python
obs = {
    "rgb_static":   ndarray | None,   # 固定视角图像
    "rgb_gripper":  ndarray | None,   # 腕部图像
    "proprio":      ndarray | None,   # 机器人状态
    "tactile_img":  ndarray | None,   # 触觉图像
    "tactile_feat": ndarray | None,   # 预提取触觉特征
    "language":     str | None,       # 任务指令
}
action = ndarray                      # shape=(7,), [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

Runner 负责把环境原生格式转成这个字典，Adapter 从字典取自己需要的字段。

---

### ② BenchmarkRunner

每个 benchmark 一个，负责跟仿真环境交互。

```python
class LiberoRunner:
    # 做的事情：
    #   1. 初始化 LIBERO 环境
    #   2. 循环：env.step() → 转标准观察字典 → model.act() → 转原生动作 → env.step()
    #   3. 统计 success_rate
```

---

### ③ ModelAdapter

每个 baseline 一个，负责封装模型推理。

```python
class FastWAMAdapter:
    # 做的事情：
    #   1. 加载 checkpoint
    #   2. act(标准观察字典) → 取需要的字段 → 模型预处理 → 推理 → 映射到标准动作
```

---

## 一句话

**写一个 Adapter，这个模型就能跑所有已经有 Runner 的 benchmark。**
