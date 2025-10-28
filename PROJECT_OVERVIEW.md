# Lean4 Delethink 数据准备工具 - 项目概览

## 🎯 项目目标

将 Lean4 形式化证明切分成块（chunks），构造训练数据，用于训练能够"分块推理"的小型语言模型。

## 🧠 核心思想

**Delethink 分块推理模式：**

```
传统方法（LongCoT）:
  问题 → 一次性生成完整证明 (5000+ tokens)

Delethink 方法:
  问题 → 生成第1块 (2000 tokens)
       → 总结关键信息（头部+尾部）
       → 基于摘要生成第2块 (2000 tokens)
       → 总结关键信息
       → 基于摘要生成第3块 (1000 tokens)
       → 完成证明 ✓

优势：
  - 固定上下文窗口（O(n) vs O(n²)）
  - 模型学会信息压缩和传递
  - 适合小模型（1.8B）训练
```

## 📦 项目结构

```
code/
├── config.yaml                   # 📋 核心配置文件
├── requirements.txt              # 📦 Python依赖
├── run_pipeline.py              # 🚀 一键运行脚本
├── test_quick.py                # 🧪 快速测试脚本
├── README.md                    # 📖 使用文档
├── PROJECT_OVERVIEW.md          # 📝 本文件
│
├── data_preparation/            # 核心处理模块
│   ├── download_data.py         # 下载 MiniF2F-Lean4
│   ├── chunk_proofs.py          # 证明切分（3种策略）
│   ├── build_training_data.py   # 构造训练样本
│   └── analyze_data.py          # 数据统计分析
│
├── utils/                       # 工具模块（预留）
│
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   │   ├── valid_raw.jsonl
│   │   └── valid_filtered.jsonl
│   └── processed/               # 训练数据
│       ├── train.jsonl          # ⭐ 训练集
│       ├── val.jsonl            # ⭐ 验证集
│       └── plots/               # 统计图表
│
└── output/                      # 测试输出
```

## ⚙️ 三种切分策略

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **line_based** | 按行数均匀切分 | 简单可靠 | 可能打断语义 |
| **syntax_based** | 在关键词处切分<br>(have, cases, induction) | 符合语义边界 | 可能识别失败 |
| **token_based** | 按token数固定切分 | 精确控制长度 | 简化实现 |

## 🔄 完整流程

```
┌─────────────────────────────────────────────────────────────┐
│                    数据准备流程                              │
└─────────────────────────────────────────────────────────────┘

步骤1: 下载数据
  └─> 从 HuggingFace 下载 MiniF2F-Lean4
  └─> 过滤无效证明（空、太长、有sorry等）
  └─> 输出: valid_filtered.jsonl, test_filtered.jsonl

步骤2: 切分证明
  └─> 选择切分策略（line/syntax/token）
  └─> 将每个证明切成 N 块
  └─> 为每块提取摘要（头部+尾部）

步骤3: 构造训练样本
  └─> 第1块：输入=定理声明，输出=第1块内容
  └─> 第2块：输入=定理+第1块摘要，输出=第2块内容
  └─> 第N块：输入=定理+历史摘要，输出=第N块内容
  └─> 输出: train.jsonl, val.jsonl

步骤4: 数据分析
  └─> 统计块数分布、长度分布
  └─> 生成可视化图表
  └─> 展示样本示例
```

## 📊 预期输出

### 数据规模（基于 MiniF2F Valid Set）

```
原始数据: 244 个定理
过滤后: ~200 个定理（去除太短/太长/有sorry）
训练样本: ~600 个（平均每个定理3块）

训练集: ~540 样本 (90%)
验证集: ~60 样本 (10%)
```

### 样本格式示例

```json
{
  "messages": [
    {
      "role": "user",
      "content": "You are a Lean4 theorem prover...\n\nTheorem:\ntheorem add_comm (a b : ℕ) : a + b = b + a := by\n\nPrevious progress:\nby\n  intro a b\n...\n<continue>\n\nContinue the proof:"
    },
    {
      "role": "assistant",
      "content": "<proof>\ninduction a with\n| zero => simp\n| succ n ih => rw [Nat.succ_add]; rw [ih]\n</proof>"
    }
  ],
  "metadata": {
    "chunk_id": 1,
    "total_chunks": 3,
    "is_first_chunk": false,
    "is_last_chunk": false,
    "example_id": 42
  }
}
```

## 🚀 快速开始

### 1. 安装

```bash
cd code
pip install -r requirements.txt
```

### 2. 测试（不下载数据）

```bash
python test_quick.py
```

预期输出：
```
测试1: 证明切分 ✓
测试2: 训练数据构造 ✓
测试3: 不同切分策略对比 ✓
```

### 3. 完整运行

```bash
python run_pipeline.py --steps all
```

预期时间：
- 下载数据: ~2-5分钟（取决于网速）
- 构造数据: ~1-2分钟
- 分析数据: ~1分钟
- **总计: ~5-10分钟**

### 4. 检查输出

```bash
ls data/processed/
# 应该看到:
#   train.jsonl (训练集)
#   val.jsonl (验证集)
#   plots/ (统计图表)
```

## 🎓 下一步：模型训练

使用生成的数据进行 SFT 训练：

```python
# 伪代码
from transformers import AutoModelForCausalLM, Trainer
from datasets import load_dataset

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "internlm/internlm2-math-plus-1_8b"
)

# 2. 加载数据
dataset = load_dataset('json', data_files={
    'train': 'data/processed/train.jsonl',
    'validation': 'data/processed/val.jsonl'
})

# 3. 训练（使用 LoRA）
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=32, lora_alpha=16)
model = get_peft_model(model, lora_config)

# 4. 开始训练
trainer = Trainer(model=model, train_dataset=dataset['train'], ...)
trainer.train()
```

预期训练时间：
- GPU: 1x A100 (40GB)
- 数据: ~600 样本
- Epochs: 3-5
- **时间: 2-4小时**

## 🔧 常用命令

```bash
# 只下载数据
python run_pipeline.py --steps download

# 只构造训练数据
python run_pipeline.py --steps build

# 只分析数据
python run_pipeline.py --steps analyze

# 使用特定策略
python run_pipeline.py --strategy syntax_based

# 自定义配置
python run_pipeline.py --config my_config.yaml
```

## 📈 关键配置参数

编辑 `config.yaml`：

```yaml
# 1. 切分策略
chunking:
  strategy: "line_based"  # 改成 "syntax_based" 试试

  line_based:
    target_chunks: 3      # 改成 2 或 4

# 2. Delethink 参数
delethink:
  keep_head: 100          # 保留的头部行数
  keep_tail: 20           # 保留的尾部行数

# 3. 过滤规则
filtering:
  min_proof_length: 10    # 最短证明（行）
  max_proof_length: 200   # 最长证明（行）
  skip_sorry: true        # 跳过未完成证明

# 4. 训练集比例
training:
  train_split_ratio: 0.9  # 90% 训练，10% 验证
```

## 🐛 故障排除

### 问题1: 下载失败

```bash
❌ Error: Connection timeout

解决方案:
  1. 检查网络连接
  2. 设置代理: export HTTP_PROXY=...
  3. 或手动下载数据集放到 data/raw/
```

### 问题2: 切分结果为空

```bash
⚠️  Generated 0 training samples

解决方案:
  1. 检查过滤条件是否太严格
  2. 放宽 min_proof_length
  3. 尝试不同的切分策略
```

### 问题3: 内存不足

```bash
❌ MemoryError

解决方案:
  1. 减少 target_chunks
  2. 增加 min_lines_per_chunk
  3. 分批处理数据
```

## 📚 技术细节

### Delethink 摘要提取

```python
def extract_summary(chunk):
    """
    保留头部和尾部，删除中间

    输入:
      by
        intro a b
        induction a with
        | zero => simp [10行]
        | succ n ih => ... [30行]
        rw [Nat.add_comm]

    输出:
      头部 (keep_head=10行):
        by
          intro a b
          induction a with
          | zero => simp
          ...

      尾部 (keep_tail=20行):
        ...
        | succ n ih =>
          rw [Nat.succ_add]
          rw [Nat.add_comm]
    """
```

### 训练样本构造逻辑

```python
for chunk in chunks:
    if chunk.is_first:
        context = ""  # 第一块无上下文
    else:
        context = summarize(previous_chunks)  # 历史摘要

    sample = {
        'input': theorem + context,
        'output': chunk.content
    }
```

## 🎯 评估指标（供参考）

训练后，评估模型的分块能力：

```python
指标1: 语法正确率
  - 生成的 Lean4 代码能否编译

指标2: 连贯性
  - 块与块之间是否逻辑连贯

指标3: 完整性
  - 拼接所有块后能否完成证明

指标4: 效率
  - 生成速度和资源消耗
```

## 📞 联系方式

有问题？
- 查看 README.md 详细文档
- 运行 test_quick.py 快速测试
- 查看代码注释

---

**祝实验顺利！🚀**
