# Lean4 Delethink 数据准备工具

将 Lean4 证明切分成块，构造用于训练"分块推理"模型的数据集。

## 📋 功能

- ✅ 下载 MiniF2F-Lean4 数据集
- ✅ 三种切分策略：按行、按语法、按token
- ✅ 自动构造训练样本（模拟 Delethink 推理流程）
- ✅ 数据统计和可视化
- ✅ 一键运行完整流程

## 🚀 快速开始

### 1. 安装依赖

```bash
cd code
pip install -r requirements.txt
```

### 2. 配置参数

编辑 `config.yaml` 调整配置：

```yaml
# 切分策略
chunking:
  strategy: "line_based"  # 可选: line_based, syntax_based, token_based

  line_based:
    target_chunks: 3      # 每个证明切成几块
    min_lines_per_chunk: 5
    max_lines_per_chunk: 50

# Delethink 参数
delethink:
  keep_head: 100          # 保留头部行数
  keep_tail: 20           # 保留尾部行数
```

### 3. 运行流程

#### 方式A：一键运行（推荐）

```bash
python run_pipeline.py --steps all
```

#### 方式B：分步运行

```bash
# 步骤1: 下载数据
python run_pipeline.py --steps download

# 步骤2: 构造训练数据
python run_pipeline.py --steps build

# 步骤3: 数据分析
python run_pipeline.py --steps analyze
```

#### 方式C：指定策略

```bash
# 使用按语法切分
python run_pipeline.py --strategy syntax_based

# 使用按token切分
python run_pipeline.py --strategy token_based
```

### 4. 输出文件

```
data/
├── raw/                          # 原始下载数据
│   ├── valid_raw.jsonl
│   ├── valid_filtered.jsonl
│   └── test_...
│
└── processed/                    # 训练数据
    ├── train.jsonl              # 训练集 ⭐
    ├── val.jsonl                # 验证集 ⭐
    ├── valid_training.jsonl
    ├── test_training.jsonl
    └── plots/                   # 统计图表
        ├── chunks_per_proof.png
        ├── chunk_lengths.png
        └── length_distributions.png
```

## 📊 数据格式

### 输入格式（原始数据）

```json
{
  "id": 0,
  "theorem": "theorem add_comm (a b : ℕ) : a + b = b + a := by",
  "proof": "  intro a b\n  induction a with\n  | zero => simp\n  | succ n ih => ...",
  "informal_statement": "Prove that addition is commutative",
  "informal_proof": "By induction on a..."
}
```

### 输出格式（训练样本）

```json
{
  "messages": [
    {
      "role": "user",
      "content": "You are a Lean4 theorem prover. Continue the proof...\n\nTheorem:\ntheorem add_comm...\n\nPrevious progress:\nintro a b\n...\n<continue>\n\nContinue the proof:"
    },
    {
      "role": "assistant",
      "content": "<proof>\ninduction a with\n| zero => simp\n</proof>"
    }
  ],
  "metadata": {
    "chunk_id": 1,
    "total_chunks": 3,
    "is_first_chunk": false,
    "is_last_chunk": false,
    "example_id": 0
  }
}
```

## 🔧 三种切分策略

### 1. 按行切分（line_based）- 默认

- 将证明均匀切成 N 块
- 简单可靠
- 适合快速验证

```yaml
chunking:
  strategy: "line_based"
  line_based:
    target_chunks: 3        # 切成3块
```

### 2. 按语法切分（syntax_based）

- 在 Lean4 关键词处切分（have, cases, induction等）
- 更符合语义边界
- 需要识别 Lean4 语法

```yaml
chunking:
  strategy: "syntax_based"
  syntax_based:
    target_chunk_size: 30
    split_keywords:
      - "have"
      - "cases"
      - "induction"
```

### 3. 按token切分（token_based）

- 固定每块 token 数
- 适合控制输入长度
- 可配置重叠

```yaml
chunking:
  strategy: "token_based"
  token_based:
    tokens_per_chunk: 2048
    overlap: 100
```

## 📈 数据统计示例

运行后会显示类似输出：

```
📊 Overall:
  Total samples: 1458
  Total proofs:  486

🔢 Chunks per proof:
  Mean: 3.0 ± 0.8
  Min:  1
  Max:  5

📏 Chunk lengths (lines):
  Mean: 12.5 ± 8.3
  Min:  2
  Max:  48

📝 Token counts (words):
  Avg input:  156 words
  Avg output: 45 words

🎯 Chunk positions:
  First chunks:  486 (33.3%)
  Middle chunks: 486 (33.3%)
  Last chunks:   486 (33.3%)
```

## 🧪 测试单个模块

```bash
# 测试数据下载
cd data_preparation
python download_data.py

# 测试切分逻辑
python chunk_proofs.py

# 测试样本构造
python build_training_data.py

# 测试数据分析
python analyze_data.py
```

## 🎯 下一步：模型训练

使用生成的 `train.jsonl` 和 `val.jsonl` 进行 SFT 训练：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# 加载数据
dataset = load_dataset('json', data_files={
    'train': 'data/processed/train.jsonl',
    'validation': 'data/processed/val.jsonl'
})

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "internlm/internlm2-math-plus-1_8b"
)
tokenizer = AutoTokenizer.from_pretrained(
    "internlm/internlm2-math-plus-1_8b"
)

# 训练
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    # ... 其他参数
)

trainer.train()
```

## ⚙️ 高级配置

### 过滤规则

```yaml
filtering:
  min_proof_length: 10      # 最短证明长度
  max_proof_length: 200     # 最长证明长度
  skip_empty_proofs: true   # 跳过空证明
  skip_sorry: true          # 跳过未完成证明（含sorry）
```

### 提示模板

编辑 `config.yaml` 中的 `prompt_template` 自定义输入格式：

```yaml
training:
  prompt_template: |
    You are a Lean4 theorem prover.

    Theorem: {theorem}
    Context: {context}

    Continue the proof:
```

### 特殊标记

```yaml
training:
  special_tokens:
    proof_start: "<proof>"
    proof_end: "</proof>"
    chunk_sep: "<chunk>"
    continue_tag: "<continue>"
```

## 📝 常见问题

### Q1: 数据下载失败？

**A:** 确保有网络连接，HuggingFace 数据集需要访问 `huggingface.co`。

### Q2: 切分结果不理想？

**A:** 尝试不同的切分策略：
- `line_based`：最稳定
- `syntax_based`：更智能但可能失败
- `token_based`：固定长度

### Q3: 内存不足？

**A:** 减少 `target_chunks` 或增加 `min_lines_per_chunk`。

### Q4: 想要更多训练数据？

**A:**
- 修改 `filtering` 放宽过滤条件
- 使用 test split：`dataset.splits: ["valid", "test"]`

## 📚 代码结构

```
code/
├── config.yaml                   # 配置文件
├── requirements.txt              # 依赖
├── run_pipeline.py              # 一键运行脚本
├── README.md                    # 本文档
│
├── data_preparation/            # 核心模块
│   ├── __init__.py
│   ├── download_data.py         # 数据下载
│   ├── chunk_proofs.py          # 证明切分
│   ├── build_training_data.py   # 样本构造
│   └── analyze_data.py          # 数据分析
│
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   └── processed/               # 训练数据
│
└── output/                      # 其他输出
```

## 🤝 贡献

欢迎提交 Issue 和 PR！

## 📄 许可

Apache 2.0 License

---

**祝训练顺利！🚀**
