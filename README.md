# 多模态时间序列预测提交包

本仓库是wcx同学的AI基础课程 Project Two 的实验代码仓库，主题为多模态时间序列预测。项目围绕三个问题展开：

- 补充文本模态是否能提升时间序列预测。
- 哪类文本模态更有效：随机文本 VS LLM 自生文本 VS 外源报告文本。
- 如何通过更稳健的融合方式让文本模态真正有效。

代码支持在 8 个 Time-MMD 风格领域数据集上运行单变量长期预测实验，预测主变量为 `OT`。数值 backbone 包含 `DLinear` 和 `PatchTST`，实验路径包含纯数值 baseline、MM-TSFlib 风格文本融合、非线性/保守残差融合、VoT 风格频域融合、LLM 自生文本和随机文本控制组。

## 目录结构

```text
submit/
├── README.md                         # 本说明文件
├── requirements.txt                  # Python 依赖
├── run.py                            # 统一训练/测试入口
├── data/                             # 已整理的数据集与文本模态
│   ├── Algriculture/                 # 农业，保留原目录拼写
│   ├── Climate/                      # 气候
│   ├── Economy/                      # 经济
│   ├── Energy/                       # 能源
│   ├── Public_Health/                # 公共健康
│   ├── Security/                     # 安全
│   ├── SocialGood/                   # 社会公益
│   ├── Traffic/                      # 交通
│   └── llm-generated/                # 已缓存的 LLM 自生文本数据与审计 JSONL
├── data_provider/                    # 数据读取、时间切分、归一化和 DataLoader
├── exp/                              # 不同实验路径的训练、验证、测试逻辑
├── layers/                           # Transformer/PatchTST 等模型层
├── models/                           # DLinear、PatchTST
├── scripts/                          # 批量运行、文本数据生成和控制组构造脚本
├── utils/                            # 指标、早停、时间特征等工具函数
└── results/                          # 已打包的实验结果、汇总表和图
```

主要代码文件说明如下：

| 路径 | 作用 |
|---|---|
| `run.py` | 统一入口，解析实验参数，设置随机种子，按 `--experiment` 选择实验类。当前提交包会强制使用单变量预测：`features=S`、`enc_in=dec_in=c_out=1`。 |
| `data_provider/data_loader.py` | `Dataset_Custom` 数据集实现。按时间顺序切分为训练/验证/测试，使用训练段拟合 `StandardScaler`，并按样本 origin 对齐文本。 |
| `exp/exp_numeric_long_term.py` | 纯数值长期预测 baseline。 |
| `exp/exp_long_term_forecasting.py` | 原始 MM-TSFlib 风格文本融合实验。 |
| `exp/exp_nonlinear_fusion_forecasting.py` | 非线性门控、边界约束和 shrink 残差融合实验。 |
| `exp/exp_vot_frequency_fusion_forecasting.py` | VoT 启发的频域分解/频带融合实验。 |
| `exp/exp_llm_generated_forecasting.py` | 使用 `ECNU_LLM_Text` 的 LLM 自生文本实验。 |
| `exp/exp_random_text_forecasting.py` | 使用随机文本的负控制实验。 |
| `scripts/run_long_matrix.sh` | 批量运行完整或部分实验矩阵。 |
| `scripts/build_random_text_control_dataset.py` | 构造字符噪声随机文本控制组。 |
| `scripts/build_ecnu_llm_text_dataset.py` | 基于 OpenAI-compatible 接口或 `--mock` 重新生成自生文本，并保存 prompt/response 审计记录。 |

## 数据和实验设置

`data/` 中包含 8 个领域数据集：

| 数据集目录 | CSV 文件 | 频率/领域 |
|---|---|---|
| `Algriculture/` | `US_RetailBroilerComposite_Month.csv` | 月度农业价格 |
| `Climate/` | `US_precipitation_month.csv` | 月度气候降水 |
| `Economy/` | `US_TradeBalance_Month.csv` | 月度经济贸易 |
| `Energy/` | `US_GasolinePrice_Week.csv` | 周度能源价格 |
| `Public_Health/` | `US_FLURATIO_Week.csv` | 周度公共健康 |
| `Security/` | `US_FEMAGrant_Month.csv` | 月度安全/灾害拨款 |
| `SocialGood/` | `Unadj_UnemploymentRate_ALL_processed.csv` | 月度失业率 |
| `Traffic/` | `US_VMT_Month.csv` | 月度交通里程 |

每个原始 CSV 至少包含以下列：

- `date`：时间戳。
- `OT`：预测目标列，本提交包默认只预测这一列。
- `start_date`、`end_date`：该行对应的时间范围。
- `prior_history_avg`：历史可见统计先验，文本分支可作为辅助 prior 使用。
- `Final_Search_2`、`Final_Search_4`、`Final_Search_6`、`Final_Output`：历史检索文本及原始输出文本。

`data/llm-generated/` 包含已经缓存好的 LLM 自生文本 CSV，文件名形如：

```text
<Dataset>_H24_F12_ecnu_llm.csv
```

其中 `H24` 表示历史窗口 `seq_len=24`，`F12` 表示预测长度 `pred_len=12`。LLM 自生文本列包括：

- `ECNU_LLM_Fact`
- `ECNU_LLM_Pred`
- `ECNU_LLM_Text`

默认长期预测设置由 `scripts/run_long_matrix.sh` 给出：

| 设置项 | 默认值 |
|---|---|
| 历史窗口 `seq_len` | `24` |
| decoder label 长度 `label_len` | `12` |
| 预测长度 `pred_len` | `12,24,36,48` |
| 随机种子 | `2021` |
| 训练轮数 | `10` |
| early stopping patience | `5` |
| batch size | `32` |
| backbone | `DLinear`、`PatchTST` |
| 指标 | MSE、MAE、RMSE、MAPE、MSPE，同时保存标准化尺度和原尺度指标 |

数据切分方式：

- 按原始时间顺序切分，不随机打乱时间轴。
- 训练集 70%，验证集 10%，测试集 20%。
- `StandardScaler` 只在训练段拟合，再应用到全序列。
- 训练 DataLoader 内部会 shuffle 训练样本窗口；验证和测试不 shuffle。

文本对齐约束：

```text
X = 历史数值窗口
S = 历史可见文本或自生文本
Y = 未来预测窗口
origin = X 的最后一个时间点
max(time used by S) <= origin
```

批处理脚本中多模态实验默认使用：

```text
--text_origin_offset -1
--prior_mode origin_repeat
```

即文本索引对齐到预测 origin，prior 使用 origin 时刻可见值重复到预测长度，避免将未来窗口统计量泄漏给模型。

## 环境安装

建议使用 Python 3.10 或 3.11。先进入当前包目录，创建并激活虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS/WSL/Git Bash：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 中包含 `numpy`、`pandas`、`scikit-learn`、`torch`、`transformers`。如果需要 GPU，请按本机 CUDA 版本安装匹配的 PyTorch wheel；CPU 也可以跑小规模 smoke test，但完整矩阵会明显更慢。

## 快速运行

以下命令均假设当前目录为。

### 1. 纯数值 baseline 冒烟测试

```bash
python -u run.py \
  --task_name long_term_forecast \
  --experiment numeric \
  --is_training 1 \
  --root_path ./data/Energy \
  --data_path US_GasolinePrice_Week.csv \
  --model_id Energy_DLinear_numeric_smoke \
  --model DLinear \
  --data custom \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 12 \
  --des smoke \
  --seed 2021 \
  --train_epochs 1 \
  --patience 1 \
  --batch_size 32 \
  --num_workers 0 \
  --d_model 128 \
  --n_heads 4 \
  --e_layers 2 \
  --d_ff 256 \
  --dropout 0.1 \
  --learning_rate 0.0001 \
  --output_dir ./results/smoke_numeric_DLinear \
  --save_name ./results/smoke_numeric_DLinear_summary.txt
```

### 2. 使用历史检索文本的多模态实验

```bash
python -u run.py \
  --task_name long_term_forecast \
  --experiment mm_tsflib \
  --is_training 1 \
  --root_path ./data/Energy \
  --data_path US_GasolinePrice_Week.csv \
  --model_id Energy_DLinear_mm_tsflib_smoke \
  --model DLinear \
  --data custom \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 12 \
  --des smoke \
  --seed 2021 \
  --text_len 4 \
  --text_origin_offset -1 \
  --prior_mode origin_repeat \
  --prompt_weight 0.1 \
  --pool_type avg \
  --llm_model BERT \
  --huggingface_token NA \
  --use_fullmodel 0 \
  --train_epochs 1 \
  --patience 1 \
  --batch_size 32 \
  --num_workers 0 \
  --d_model 128 \
  --n_heads 4 \
  --e_layers 2 \
  --d_ff 256 \
  --dropout 0.1 \
  --learning_rate 0.0001 \
  --output_dir ./results/smoke_mm_tsflib_DLinear \
  --save_name ./results/smoke_mm_tsflib_DLinear_summary.txt
```

`mm_tsflib` 默认使用 `Final_Search_<text_len>` 列。上例中 `--text_len 4` 对应 `Final_Search_4`。

### 3. 使用 LLM 自生文本

```bash
python -u run.py \
  --task_name long_term_forecast \
  --experiment llm_generated \
  --is_training 1 \
  --root_path ./data/Energy \
  --data_path ../llm-generated/Energy_H24_F12_ecnu_llm.csv \
  --model_id Energy_DLinear_llm_generated_smoke \
  --model DLinear \
  --data custom \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 12 \
  --text_column ECNU_LLM_Text \
  --text_origin_offset -1 \
  --prior_mode origin_repeat \
  --des smoke \
  --seed 2021 \
  --text_len 4 \
  --prompt_weight 0.1 \
  --pool_type avg \
  --llm_model BERT \
  --huggingface_token NA \
  --use_fullmodel 0 \
  --train_epochs 1 \
  --patience 1 \
  --batch_size 32 \
  --num_workers 0 \
  --d_model 128 \
  --n_heads 4 \
  --e_layers 2 \
  --d_ff 256 \
  --dropout 0.1 \
  --learning_rate 0.0001 \
  --output_dir ./results/smoke_llm_generated_DLinear \
  --save_name ./results/smoke_llm_generated_DLinear_summary.txt
```

注意：LLM 自生文本文件与 `seq_len`、`pred_len` 绑定。上例使用 `H24_F12`，因此对应 `--seq_len 24 --pred_len 12`。

### 4. 使用随机文本控制组

先构造控制组 CSV：

```bash
python scripts/build_random_text_control_dataset.py \
  --input ./data/Energy/US_GasolinePrice_Week.csv \
  --output ./data/Energy/US_GasolinePrice_Week_random_text.csv \
  --source-text-column Final_Search_4 \
  --text-column Random_Text \
  --mode char_noise \
  --seed 20240624 \
  --overwrite
```

再运行随机文本实验：

```bash
python -u run.py \
  --task_name long_term_forecast \
  --experiment random_text \
  --is_training 1 \
  --root_path ./data/Energy \
  --data_path US_GasolinePrice_Week_random_text.csv \
  --model_id Energy_DLinear_random_text_smoke \
  --model DLinear \
  --data custom \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 12 \
  --text_column Random_Text \
  --text_origin_offset -1 \
  --prior_mode origin_repeat \
  --des smoke \
  --seed 2021 \
  --text_len 4 \
  --prompt_weight 0.1 \
  --pool_type avg \
  --llm_model BERT \
  --huggingface_token NA \
  --use_fullmodel 0 \
  --train_epochs 1 \
  --patience 1 \
  --batch_size 32 \
  --num_workers 0 \
  --d_model 128 \
  --n_heads 4 \
  --e_layers 2 \
  --d_ff 256 \
  --dropout 0.1 \
  --learning_rate 0.0001 \
  --output_dir ./results/smoke_random_text_DLinear \
  --save_name ./results/smoke_random_text_DLinear_summary.txt
```

随机文本控制组只保留字符长度等粗略属性，不包含真实语义，用于判断提升是否来自文本内容本身，而不是额外分支、参数或 prior。

## 批量运行实验矩阵

推荐在 Linux、WSL、AutoDL 或 Git Bash 中使用：

```bash
bash scripts/run_long_matrix.sh [experiment] [model] [dataset] [gpu] [dry_run]
```

位置参数：

| 参数 | 含义 | 示例 |
|---|---|---|
| `experiment` | 实验路径，支持 `all` 或逗号分隔列表 | `numeric`、`mm_tsflib`、`random_text` |
| `model` | backbone，支持 `all`、`DLinear`、`PatchTST` | `PatchTST` |
| `dataset` | 数据集，支持 `all` 或逗号分隔列表 | `Energy,Public_Health` |
| `gpu` | `CUDA_VISIBLE_DEVICES` | `0` |
| `dry_run` | `1` 只打印命令，不真正运行 | `1` |

示例：

```bash
# 只打印 Energy 上 PatchTST + mm_tsflib 的命令
bash scripts/run_long_matrix.sh mm_tsflib PatchTST Energy 0 1

# 跑 Energy 和 Public_Health 的纯数值 DLinear
bash scripts/run_long_matrix.sh numeric DLinear Energy,Public_Health 0

# 跑所有默认实验、两个模型、全部数据集
bash scripts/run_long_matrix.sh
```

常用环境变量：

```bash
PRED_LENS=12,24 SEEDS=2021 EPOCHS=3 PATIENCE=2 bash scripts/run_long_matrix.sh numeric DLinear Energy 0
```

PowerShell 中可以这样设置：

```powershell
$env:PRED_LENS="12,24"
$env:SEEDS="2021"
$env:EPOCHS="3"
$env:PATIENCE="2"
bash scripts/run_long_matrix.sh numeric DLinear Energy 0
```

脚本会自动跳过已经存在且 `metrics.json` 中 MSE/MAE 为有限值的结果。对 `random_text` 实验，若控制组 CSV 不存在，脚本会默认先调用 `build_random_text_control_dataset.py` 生成。

## 支持的实验路径

`run.py --experiment` 当前支持：

| 实验名 | 含义 |
|---|---|
| `numeric` | 纯数值 baseline。 |
| `mm_tsflib` | 原始 MM-TSFlib 风格文本融合。 |
| `mm_tsflib_nonlinear` | 非线性门控融合。 |
| `mm_tsflib_vot_freq` | VoT 启发的低/高频分解融合。 |
| `mm_tsflib_freq_residual` | 频域 residual-only 融合变体。 |
| `llm_generated` | 使用缓存的 `ECNU_LLM_Text` 自生文本。 |
| `random_text` | 随机字符文本控制组。 |

支持的 backbone：

```text
DLinear
PatchTST
```

## 重新生成 LLM 自生文本

提交包中已经包含 `data/llm-generated/` 下的缓存结果，一般不需要重新生成。如果需要复现文本生成过程，可以使用 OpenAI-compatible Chat Completions 接口。

环境变量：

```bash
export ECNU_LLM_BASE_URL="https://your-api-base.example/v1"
export ECNU_LLM_API_KEY="your-api-key"
export ECNU_LLM_MODEL="your-model-name"
```

本地 mock 冒烟测试不会调用外部接口：

```bash
python scripts/build_ecnu_llm_text_dataset.py \
  --input ./data/Energy/US_GasolinePrice_Week.csv \
  --output ./data/llm-generated/Energy_H24_F12_ecnu_llm_mock.csv \
  --audit-output ./data/llm-generated/audit/Energy_H24_F12_ecnu_llm_mock_audit.jsonl \
  --domain Energy \
  --target-col OT \
  --value-cols OT \
  --seq-len 24 \
  --pred-len 12 \
  --mock \
  --overwrite
```

真实接口生成时去掉 `--mock`，并确保已经配置 `ECNU_LLM_BASE_URL` 和 `ECNU_LLM_API_KEY`。脚本会保存：

- 新 CSV：包含 `ECNU_LLM_Fact`、`ECNU_LLM_Pred`、`ECNU_LLM_Text`。
- 审计 JSONL：包含 row index、prompt、response、模型名、生成参数和泄漏检查字段。

## 结果文件

单次运行会在 `--output_dir` 下创建目录：

```text
<output_dir>/long_term_forecast_<model_id>_.../
```

每个 run 目录通常包含：

| 文件 | 含义 |
|---|---|
| `metrics.json` | 主指标，包含 MSE/MAE/RMSE/MAPE/MSPE，以及原尺度 `*_orig` 指标。 |
| `metrics.npy` | NumPy 格式指标数组。 |
| `pred.npy`、`true.npy` | 标准化尺度预测和真实值。 |
| `pred_orig.npy`、`true_orig.npy` | 反归一化后的预测和真实值。 |
| 频域融合额外字段 | `metrics.json` 中记录频带权重、shrink 参数等融合状态。 |

如果传入 `--save_name`，脚本还会把每次测试的 MSE/MAE 追加写入对应 summary txt。

已打包结果位于：

```text
results/output_extracted_20260626/
```

重点查看：

| 路径 | 内容 |
|---|---|
| `results/output_extracted_20260626/analysis_long_term/LONG_TERM_RESULTS_ANALYSIS.md` | 长期预测矩阵自动分析摘要。 |
| `results/output_extracted_20260626/analysis_long_term/long_term_metrics_raw.csv` | 所有解析到的原始 run 指标。 |
| `results/output_extracted_20260626/analysis_long_term/long_term_metrics_mean.csv` | 按数据集、模型、horizon、实验聚合后的均值表。 |
| `results/output_extracted_20260626/analysis_long_term/numeric_vs_mm_tsflib_by_horizon.csv` | 纯数值 vs 真实历史文本。 |
| `results/output_extracted_20260626/analysis_long_term/numeric_vs_random_text_by_horizon.csv` | 纯数值 vs 随机文本控制组。 |
| `results/output_extracted_20260626/analysis_long_term/random_text_vs_mm_tsflib_by_horizon.csv` | 随机文本 vs 真实历史文本。 |
| `results/output_extracted_20260626/analysis_long_term/fig_*.png`、`fig_*.pdf` | 可放入报告或展示的结果图。 |

`LONG_TERM_RESULTS_ANALYSIS.md` 中记录的已解析 run 数为 2372，并给出了 `mm_tsflib` 相对 `numeric` 的 win rate、随机文本控制组说明、非线性融合控制和频域融合控制。小幅提升应视为探索性结果，因为当前打包矩阵主要是单 seed。

## 复现实验建议

为了最小化复现成本，可以按以下顺序运行：

1. 先跑一个纯数值 smoke test，确认环境、数据路径和输出目录正常。
2. 跑同一数据集、同一 backbone、同一 horizon 的 `numeric`、`mm_tsflib`、`random_text` 三组，先判断真实文本是否超过随机文本控制组。
3. 再比较 `mm_tsflib_nonlinear`、`mm_tsflib_vot_freq`、`*_shrink` 等融合变体，判断改进来自文本内容还是融合结构。
4. 如果时间允许，对关键组合增加多个 seed，例如 `SEEDS=2021,2022,2023`。

示例小矩阵：

```bash
PRED_LENS=12 SEEDS=2021 EPOCHS=3 PATIENCE=2 bash scripts/run_long_matrix.sh numeric,mm_tsflib,random_text DLinear Energy 0
```

若要使用 PowerShell：

```powershell
$env:PRED_LENS="12"
$env:SEEDS="2021"
$env:EPOCHS="3"
$env:PATIENCE="2"
bash scripts/run_long_matrix.sh numeric,mm_tsflib,random_text DLinear Energy 0
```

## 解释结果时的边界

报告结论时建议使用以下口径：

- 明确比较对象：相对 `numeric`、`random_text` 还是另一个 fusion 变体。
- 明确数据集、backbone、`pred_len` 和 seed。
- 同时看 MSE 和 MAE，必要时查看原尺度 `mse_orig`、`mae_orig`。
- 多模态提升必须和随机文本控制组对照，不能只说“加文本就提升”。
- 单 seed 小幅提升不能直接解释为稳定结论，应标注为探索性现象。
- 如果某个数据集或 horizon 下退化，应如实报告，不要只挑提升项。

更合适的表述示例：

```text
在 Energy、DLinear、seq_len=24、pred_len=12、seed=2021 的同一切分下，
mm_tsflib 相对 numeric 的 MSE 下降，同时优于 random_text 控制组；
因此在该设置下，真实历史文本内容比随机文本更有帮助。
该结论是否稳定仍需更多 seed 验证。
```

## 常见问题

**1. 为什么命令里写了 `--features S` 但 `run.py` 还会打印 overwrite？**

`run.py` 会强制设置 `features='S'`、`enc_in=dec_in=c_out=1`，保证提交包所有实验都按单变量 `OT` 预测执行。

**2. 为什么 `llm_generated` 的 `data_path` 是 `../llm-generated/...`？**

脚本保持 `root_path=./data/<Dataset>`，这样结果解析仍能从 `root_path` 识别原始数据集名称；实际 CSV 通过相对路径读取 `data/llm-generated/` 下的缓存文件。

**3. Windows 上不能运行 `.sh` 怎么办？**

可以使用 Git Bash、WSL 或 AutoDL/Linux 运行 `scripts/run_long_matrix.sh`。也可以参考上面的单次 `python -u run.py ...` 命令在 PowerShell 中逐个运行。

**4. 结果中出现 NaN 怎么处理？**

先查看对应 run 目录下的 `metrics.json`、终端日志和数据长度。`scripts/run_long_matrix.sh` 会跳过样本窗口不足的组合，并对已有非有限指标进行重跑。已打包分析中也列出了部分 invalid/NaN 指标，解释结果时应排除或单独说明。
