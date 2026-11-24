# 开源组件与许可证

| 组件 | 版本 | 用途 | 许可证 | 位置 |
|---|---|---|---|---|
| Python | 3.12（镜像 `python:3.12-slim-bookworm`） | 运行时 | PSF License 2.0 | Docker 基础镜像 |
| Debian bookworm（slim） | 12 | 镜像操作系统 | 各软件包自身许可证（DFSG） | Docker 基础镜像 |
| NumPy | 1.26.4 | 官方训练器与 Harness 的数值计算 | BSD-3-Clause（内含 OpenBLAS，BSD-3-Clause） | `pip install numpy==1.26.4` |
| PyYAML | 6.0.3 | 读取离线配置与经验库 | MIT | `pip install PyYAML==6.0.3` |
| contest_harness SDK | 1.0.4 | 平台协议（describe / train / predict） | 主办方提供，随赛题发布，未声明开源许可证；未修改 | `third_party/contest-harness-sdk-1.0.4/` |
| 官方 CPU 基线训练器 `baselines/` | cpu-b-rul-20261110-v1 | 工业模型训练与预测；Harness 在其 4 个插槽中换函数体 | 主办方提供，随赛题发布，未声明开源许可证 | `vendor/cpu-baseline/baselines/` |
| SwanLab（可选） | ≥ 0.10 | 离线实验指标同步，镜像内未安装 | Apache-2.0 | `pip install swanlab`（可选依赖） |

大模型为 Qwen3.8-27B，比赛中只通过平台宿主网关调用（`rnd.model`），镜像内没有模型权重、客户端或密钥。
