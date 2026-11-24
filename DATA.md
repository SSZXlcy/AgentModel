# 本地数据输入与预处理安排

原始数据统一放在 `data/raw/`，派生缓存放在 `data/processed/`，训练产物放在 `runs/`。三套数据分别创建独立Pipeline，不进行联合采样或联合训练。

```text
data/
├── raw/
│   ├── gas/
│   │   ├── batch1.dat
│   │   └── ... batch10.dat
│   ├── xjtu-sy/
│   │   ├── 35Hz12kN/Bearing1_1/1.csv
│   │   └── ...（保留原轴承目录层级）
│   └── cmapss/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
└── processed/
    └── bearing_features.npz
```

## Gas

`configs/tasks/gas_real.yaml` 默认递归读取 `data/raw/gas` 下的 `batch*.dat`，兼容官方压缩包的 `Dataset/` 目录。适配器逐行解析类别和128维特征，支持官方的空格分隔格式及分号分隔格式，类别从1—6转换为0—5，文件名决定时间批次。Pipeline用初始批次训练，并从初始训练数据拟合标准化器；后续批次按顺序进入漂移监测。原始数据不需要转成图片或时序窗口。

```powershell
.\.venv\Scripts\python.exe -m agent_harness demo --task gas --config configs/tasks/gas_real.yaml
```

## XJTU-SY

`configs/tasks/bearing_real.yaml` 递归读取 `data/raw/xjtu-sy/**/*.csv`，父目录标识轴承实体，文件数字编号决定退化顺序。每个文件应为两列振动信号。每个通道计算16个时域和频域特征，双通道形成32维向量；RUL按该轴承剩余采集次数乘采样间隔构造。首次提取后缓存到 `data/processed/bearing_features.npz`，原始数据变化时应删除旧缓存再生成。

划分以完整轴承为单位：前60%实体用于初始训练，接下来20%作为历史保留评价，最后20%形成独立时间流。一个轴承的相邻CSV不会同时进入初始训练和历史保留集。当前版本实现的是特征MLP，不直接把65,536个采样点展开到MLP。

```powershell
.\.venv\Scripts\python.exe -m agent_harness demo --task bearing --config configs/tasks/bearing_real.yaml
```

## C-MAPSS

`configs/tasks/cmapss_real.yaml` 默认读取 `train_FD001.txt`。适配器按发动机编号和周期排序，以30周期滑动窗口展开24个工况/传感器字段，形成720维输入；训练RUL由失效周期减当前周期得到，并默认截断到125。窗口只使用当前及过去周期。不同FD子集应使用不同配置文件和独立运行目录。

当前真实训练入口使用完整训练序列。官方截断测试文件与 `RUL_FD*.txt` 应保留为最终评价数据，不参与策略训练或阈值调节。FD002/FD004的多工况条件化仍是需要增强的领域预处理，不能简单把全部周期标准化后解释为已经解决工况差异。

```powershell
.\.venv\Scripts\python.exe -m agent_harness demo --task cmapss --config configs/tasks/cmapss_real.yaml
```

## 标签流与实验边界

公开数据实验由Pipeline按窗口评价，并在触发更新后从当前窗口选择有限标签用于候选训练。真实隐藏流应由评测SDK提供等价的标签预算接口，不能通过读取完整测试文件绕过标签限制。每次运行的配置、数据划分、标签消耗、模型版本和评价结果都写入 `runs/`，SwanLab只负责实时指标展示，本地JSONL和模型文件才是可恢复的事实记录。
