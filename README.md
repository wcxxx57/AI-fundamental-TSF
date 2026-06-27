# AI基础大作业：多模态时间序列预测

本目录是一个时间序列预测代码与实验结果汇总包。任务是在 8 个领域数据集上做单变量长期预测，目标列统一为 `OT`，数值 backbone 为 `DLinear` 和 `PatchTST`。

实验重点是比较文本模态对预测的影响：纯数值 baseline、MM-TSFlib 风格历史文本融合、非线性残差融合、VoT 风格频域融合、LLM 自生文本，以及随机文本负控制组。

## 目录结构

```text
submit/
├── README.md
├── requirements.txt
├── run.py
├── data/
│   ├── Algriculture/
│   ├── Climate/
│   ├── Economy/
│   ├── Energy/
│   ├── Public_Health/
│   ├── Security/
│   ├── SocialGood/
│   ├── Traffic/
│   └── llm-generated/
├── data_provider/
├── exp/
├── layers/
├── models/
├── scripts/
├── utils/
└── results/
    └── results_summary/
```

主要文件：

| 路径 | 作用 |
|---|---|
| `run.py` | 统一训练/测试入口，通过 `--experiment` 选择实验类型。代码会强制使用单变量设置：`features=S`、`enc_in=dec_in=c_out=1`。 |
| `scripts/run_long_matrix.sh` | 批量运行数据集、模型、预测长度和实验类型的矩阵。 |
| `scripts/build_random_text_control_dataset.py` | 构造随机文本控制组 CSV。 |
| `scripts/build_ecnu_llm_text_dataset.py` | 重新生成 LLM 自生文本数据，可接 OpenAI-compatible Chat Completions 接口，也支持 `--mock`。 |
| `data_provider/data_loader.py` | 数据读取、时间顺序切分、归一化、文本对齐和 prior 构造。 |
| `exp/exp_numeric_long_term.py` | 纯数值长期预测 baseline。 |
| `exp/exp_long_term_forecasting.py` | MM-TSFlib 风格文本融合。 |
| `exp/exp_nonlinear_fusion_forecasting.py` | 非线性残差门控融合。 |
| `exp/exp_vot_frequency_fusion_forecasting.py` | VoT 风格频域分解融合。 |
| `exp/exp_llm_generated_forecasting.py` | LLM 自生文本实验。 |
| `exp/exp_random_text_forecasting.py` | 随机文本控制组实验。 |

## 正式实验

`run.py --experiment` 保留以下入口：

| 实验名 | 含义 |
|---|---|
| `numeric` | 纯数值 baseline，不使用文本。 |
| `mm_tsflib` | 使用原始数据中的历史检索文本列，按 MM-TSFlib 风格融合数值预测和文本分支。 |
| `mm_tsflib_nonlinear` | 在数值预测和文本侧预测之间学习非线性残差门控。 |
| `mm_tsflib_vot_freq` | 借鉴 VoT 的频域思想，对数值侧和文本侧预测做低/中/高频分解后融合。 |
| `llm_generated` | 使用 `data/llm-generated/` 中缓存的 `ECNU_LLM_Text` 自生文本。 |
| `random_text` | 使用随机字符文本作为负控制组，用来区分文本语义和额外分支/参数带来的影响。 |


## 数据

原始数据位于 `data/<Dataset>/`，每个 CSV 至少包含：

- `date`：时间戳。
- `OT`：预测目标列。
- `start_date`、`end_date`：该行对应的时间范围。
- `prior_history_avg`：历史可见统计 prior。
- `Final_Search_2`、`Final_Search_4`、`Final_Search_6`、`Final_Output`：历史检索文本列。

数据集清单：

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

`data/llm-generated/` 中缓存了 `seq_len=24`、`pred_len=12` 的 LLM 自生文本文件，命名格式为：

```text
<Dataset>_H24_F12_ecnu_llm.csv
```

其中主要文本列为 `ECNU_LLM_Text`，审计记录位于 `data/llm-generated/audit/`。

## 实验设置

默认矩阵设置由 `scripts/run_long_matrix.sh` 给出：

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
| 指标 | MSE、MAE、RMSE、MAPE、MSPE，并保存标准化尺度和原尺度指标 |

数据按时间顺序切分，不打乱时间轴：训练集 70%，验证集 10%，测试集 20%。`StandardScaler` 只在训练段拟合，再应用到完整序列。

文本对齐遵循：

```text
X = 历史数值窗口
S = 历史可见文本或自生文本
Y = 未来预测窗口
origin = X 的最后一个时间点
max(time used by S) <= origin
```

批量脚本默认传入：

```text
--text_origin_offset -1
--prior_mode origin_repeat
```

即文本对齐到预测 origin，prior 使用 origin 时刻可见值重复到预测长度，避免使用未来窗口统计信息。

## 环境安装

建议使用 Python 3.10 或 3.11。

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux、macOS、WSL 或 Git Bash：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含 `numpy`、`pandas`、`scikit-learn`、`torch`、`transformers`。如果使用 GPU，请根据本机 CUDA 版本安装匹配的 PyTorch wheel。

## 运行方式

推荐使用批量脚本：

```bash
bash scripts/run_long_matrix.sh [experiment] [model] [dataset] [gpu] [dry_run]
```

位置参数：

| 参数 | 含义 | 示例 |
|---|---|---|
| `experiment` | `all` 或逗号分隔实验名 | `numeric,mm_tsflib,random_text` |
| `model` | `all`、`DLinear` 或 `PatchTST` | `DLinear` |
| `dataset` | `all` 或逗号分隔数据集名 | `Energy,Public_Health` |
| `gpu` | `CUDA_VISIBLE_DEVICES` | `0` |
| `dry_run` | `1` 只打印命令，不实际运行 | `1` |

示例：

```bash
# 只打印命令，检查 Energy 上 DLinear 的三组核心对比
bash scripts/run_long_matrix.sh numeric,mm_tsflib,random_text DLinear Energy 0 1

# 运行 Energy 和 Public_Health 的纯数值 baseline
bash scripts/run_long_matrix.sh numeric DLinear Energy,Public_Health 0

# 运行所有正式实验、两个 backbone 和全部数据集
bash scripts/run_long_matrix.sh
```

常用环境变量：

```bash
PRED_LENS=12 SEEDS=2021 EPOCHS=3 PATIENCE=2 bash scripts/run_long_matrix.sh numeric,mm_tsflib,random_text DLinear Energy 0
```

PowerShell 中可先设置环境变量，再调用 Git Bash/WSL 中的脚本：

```powershell
$env:PRED_LENS="12"
$env:SEEDS="2021"
$env:EPOCHS="3"
$env:PATIENCE="2"
bash scripts/run_long_matrix.sh numeric,mm_tsflib,random_text DLinear Energy 0
```

脚本会将单次运行产物写入 `results/raw_runs/`，将每组实验的 summary 追加写入 `results/results_summary/`。`results/raw_runs/` 已加入 `.gitignore`，提交包只保留汇总结果。

## 单次命令示例

如果不使用批量脚本，可以直接调用 `run.py`。下面示例运行 Energy 数据集上的 DLinear 纯数值预测：

```bash
python -u run.py \
  --task_name long_term_forecast \
  --experiment numeric \
  --is_training 1 \
  --root_path ./data/Energy \
  --data_path US_GasolinePrice_Week.csv \
  --model_id Energy_DLinear_numeric_s2021_sl24_pl12 \
  --model DLinear \
  --data custom \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 12 \
  --des numeric_example \
  --seed 2021 \
  --train_epochs 3 \
  --patience 2 \
  --batch_size 32 \
  --num_workers 0 \
  --d_model 128 \
  --n_heads 4 \
  --e_layers 2 \
  --d_ff 256 \
  --dropout 0.1 \
  --learning_rate 0.0001 \
  --output_dir ./results/raw_runs/numeric_DLinear_long_term \
  --save_name ./results/results_summary/numeric_DLinear_summary.txt
```

文本融合实验需要额外传入文本分支参数。批量脚本已经统一处理这些参数，手动运行时可参考脚本生成的命令。

## 随机文本控制组

`random_text` 实验需要随机文本 CSV。批量脚本默认在文件不存在时自动调用：

```bash
python scripts/build_random_text_control_dataset.py \
  --input ./data/Energy/US_GasolinePrice_Week.csv \
  --output ./data/Energy/US_GasolinePrice_Week_random_text.csv \
  --source-text-column Final_Search_4 \
  --text-column Random_Text \
  --mode char_noise \
  --seed 20240624
```

随机文本只保留字符长度等粗略属性，不携带真实语义，用来判断提升是否来自文本内容本身。

## LLM 自生文本

提交包已经包含 `data/llm-generated/` 下的缓存结果。通常不需要重新生成。

如需复现生成过程，设置 OpenAI-compatible 接口：

```bash
export ECNU_LLM_BASE_URL="https://your-api-base.example/v1"
export ECNU_LLM_API_KEY="your-api-key"
export ECNU_LLM_MODEL="your-model-name"
```

本地 mock 生成示例：

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

真实接口生成时去掉 `--mock`，并确保环境变量已经配置。

## 结果文件

正式汇总结果位于：

```text
results/results_summary/
```

当前包含以下 12 个 summary 文件：

```text
numeric_DLinear_summary.txt
numeric_PatchTST_summary.txt
mm_tsflib_DLinear_summary.txt
mm_tsflib_PatchTST_summary.txt
mm_tsflib_nonlinear_DLinear_summary.txt
mm_tsflib_nonlinear_PatchTST_summary.txt
mm_tsflib_vot_freq_DLinear_summary.txt
mm_tsflib_vot_freq_PatchTST_summary.txt
llm_generated_DLinear_summary.txt
llm_generated_PatchTST_summary.txt
random_text_DLinear_summary.txt
random_text_PatchTST_summary.txt
```

单次 run 目录通常包含：

| 文件 | 含义 |
|---|---|
| `metrics.json` | MSE、MAE、RMSE、MAPE、MSPE，以及原尺度指标。 |
| `metrics.npy` | NumPy 格式指标数组。 |
| `pred.npy`、`true.npy` | 标准化尺度预测值和真实值。 |
| `pred_orig.npy`、`true_orig.npy` | 反归一化后的预测值和真实值。 |
| `config.json`、`history.csv` | 实验参数和训练过程记录。 |
