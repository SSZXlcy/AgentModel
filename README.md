# Agent Harness：工业模型受控自进化系统

围绕官方 CPU 训练器（`cpu-*-20261110-v1/baselines/`）构建的两层自进化系统。完整设计、实测依据和验收结果见 **[设计方案.md](设计方案.md)**。

- **模型训练层**：每来一批数据，决定用哪些数据、练多少轮、从哪个版本开始练；候选必须超过波动范围才能试用，下一批确认后才转正。
- **监督器**：每批做五项健康检查，决定**延续**当前策略、**复查**（一次轻量判断），还是**更新**策略。
- **三种训练节奏**：积极跟随（答案及时、输入在变）、设备完成时更新（答案要等设备跑完）、节省（输入被训练数据覆盖、或更新无收益），外加谨慎的标准节奏。由经验库里带条件的经验根据数据特征选配，运行中特征变化时自动切换（设计方案 §13）。
- **统一 pipeline（第四版，设计方案 §14）**：开局按结构、数据流格式和输入新颖度选节奏；开局实验有预算并计入成本，数据与训练代码可组合生效；节奏切换要有累计证据和最短停留期；失守时先退回开局验证过的模型；27B 只在规则不够用时出场，每次运行 ≤ 3 万 Token。任务配置只需给出数据位置。
- **策略迭代**：需要更新时，agent 先用通用工具自己探索数据和运行记录，再给出分析方案；每个修正都要在已发生的数据上验证，通过才生效，生效后有观察期，不行就撤销。每轮由 agent 总结经验，写进 journal、经验库和报告。

工业模型始终由官方训练程序产出。插槽保持默认内容时，训练结果与官方逐字节相同。系统里不写死任何任务或字段名，数据结构由工具自己发现。

## 演示视频与文档

| 内容 | 链接 |
|---|---|
| 5 分钟演示视频（中文字幕已压入画面） | [demo/智训引擎演示.mp4](demo/智训引擎演示.mp4)，字幕文件 [demo/智训引擎演示.srt](demo/智训引擎演示.srt)，分镜 [demo/分镜与字幕.md](demo/分镜与字幕.md) |
| 项目说明书 | [docs/pdf/项目说明书.pdf](docs/pdf/项目说明书.pdf) |
| 系统运行说明 | [docs/pdf/系统运行说明.pdf](docs/pdf/系统运行说明.pdf)（源文件 [docs/系统运行说明.md](docs/系统运行说明.md)） |
| 基础模型使用与实验记录 | [docs/实验记录.md](docs/实验记录.md)，云端记录 [SwanLab](https://swanlab.cn/@SSZXlcy/AgentModel-zhixun-harness) |

## 环境

官方训练器锁定 NumPy 1.26.4，请使用 `.venv-official`：

```powershell
uv venv .venv-official --python 3.11
uv pip install --python .venv-official/Scripts/python.exe "numpy==1.26.4" "PyYAML>=6.0"
$env:PYTHONPATH = "src"
```

## 运行

```powershell
# 在模拟数据流上跑一遍：训练层 + 监督器，结束时生成 report.md
.venv-official\Scripts\python.exe -m agent_harness run --task B
.venv-official\Scripts\python.exe -m agent_harness run --task B --label-protocol on_group_complete   # 设备跑完才给答案
.venv-official\Scripts\python.exe -m agent_harness run --task B --no-supervise                       # 只有训练层

# 中断后续跑
.venv-official\Scripts\python.exe -m agent_harness run --task B --resume runs\B\<时间戳>

# 对照：不更新 / 无脑更新 / 只有训练层 / 完整系统（加 --llm 再多一组）；compare.json 含 gain_per_cost
.venv-official\Scripts\python.exe -m agent_harness compare --task B --seeds 0 1 2
# 任意任务配置（更长的公开模拟流；快照格式先运行 scripts\make_public_snapshot.py）
.venv-official\Scripts\python.exe -m agent_harness compare --task-config configs\tasks\A_long.yaml
.venv-official\Scripts\python.exe -m agent_harness compare --task-config configs\tasks\C_snapshot.yaml
# 留一任务验证（通用性）
.venv-official\Scripts\python.exe scripts\leave_one_task_out.py

# 报告、查看
.venv-official\Scripts\python.exe -m agent_harness report runs\B\<时间戳>
.venv-official\Scripts\python.exe -m agent_harness inspect runs\B\<时间戳>

# 离线策略迭代（代码插槽 / 规则参数）
.venv-official\Scripts\python.exe -m agent_harness evolve-code --task B --rounds 3 --seeds 0 1 2
.venv-official\Scripts\python.exe -m agent_harness evolve-policy --task A --seeds 0 1 2

# 测试
.venv-official\Scripts\python.exe -m unittest discover -s tests -v
```

### 大模型（Qwen3.8-27B）

`--llm` 启用大模型（OpenAI 兼容接口，默认地址 `https://maas.qianwenaiapi.com/compatible-mode/v1`，默认模型 `qwen3.8-27b`）。密钥**由你自己**放在用户目录下的私有文件里，不进项目、不进命令行、不进任何记录：

```powershell
New-Item -ItemType Directory -Force "$HOME\.agent_harness" | Out-Null
Set-Content -NoNewline "$HOME\.agent_harness\llm_key" "<你的密钥>"
.venv-official\Scripts\python.exe -m agent_harness llm-check      # 只发一句与项目无关的话，确认能连通
```

也可以用环境变量 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_ID`。大模型不可用、回答不是合法 JSON、越权或超预算时，系统自动改用确定性的兜底分析器，不会中断运行。

`--swanlab` 把指标同步到 SwanLab（密钥放在 `SWANLAB_API_KEY`）。SwanLab 失败不影响运行，本地 `runs/` 才是可恢复的事实记录。

**不要把任何密钥写进代码、配置或记录。**

## 平台提交（contest_harness SDK 1.0.4）

平台每个检查点轮次运行一次 `python3 /app/harness.py train`。适配层是 `src/agent_harness/contest.py`，入口模块是 `submission/contest_submission.py`。每一轮：

1. 识别数据结构，选节奏；
2. 首轮在训练集上按设备留出，做开局实验；
3. 从父检查点续训官方对照和已验证策略两个候选；
4. 平台测量，同时在本地做配对测量；
5. 27B 在安全集合内选择，越界时由规则选择；
6. 保存状态。

详见 [docs/系统运行说明.md](docs/系统运行说明.md)。

```powershell
docker build -f docker/Dockerfile -t zhixun-harness:1.0 .
docker run --rm zhixun-harness:1.0 describe
# 用 JSONL 宿主替身驱动 A/B/C 各两轮，并逐行预测核对（加 --docker 在镜像内、无网络运行）
.venv-official\Scripts\python.exe scripts\sdk_smoke.py --sdk third_party\contest-harness-sdk-1.0.4 --docker zhixun-harness:1.0
```

## 数据

只使用官方公开数据：官方检查点作为起点，公开验证集按原始顺序切成若干批，模拟数据流。`data/raw/` 中可能与隐藏测试流重合的原始数据，不用于训练或调整参数（见设计方案 §5.4）。

## 目录

```text
configs/tasks/{A,B,C}.yaml     任务配置（数据包、每批大小；设备字段由数据自动识别）
configs/tasks/{A_long,C_long,C_snapshot}.yaml  更长的公开模拟流 / C 官方测试格式（只用公开数据）
configs/policies/default.yaml  训练层规则参数 π
configs/experiences/seed.yaml  经验库的初始经验（条件 → 算子调用）
docs/修改说明书.md              算子手册（由 `agent_harness manual` 生成）
src/agent_harness/             系统代码（模块说明见设计方案 §8）
tests/test_harness.py          52 项验收测试
artifacts/<任务>/               代码版本、晋升记录、修改尝试记录、重放结果
artifacts/memory/library/      跨运行、跨任务共享的经验笔记（含 27B 总结出的带条件经验）
artifacts/memory/experiences/  每条经验在观察期的确认/撤销记录（决定其可信度）
runs/<任务>/<时间戳>/           manifest / events / transitions（含 strategy_update）/ tree / summary /
                               journal.jsonl / strategies/ / reports/ / report.md
legacy/agent_harness_v0/       旧的 PyTorch 版本（已归档，不再使用）
```
