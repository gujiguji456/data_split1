# 🚀 快速上手指南

## 30秒快速开始

```bash
# 1. 进入目录
cd code

# 2. 安装依赖
pip install -r requirements.txt

# 3. 快速测试（无需下载数据）
python test_quick.py

# 4. 完整运行
python run_pipeline.py --steps all

# 5. 查看结果
ls data/processed/train.jsonl
```

## 预期结果

运行完成后，你会得到：

```
data/processed/
├── train.jsonl       # ~540 训练样本
├── val.jsonl         # ~60 验证样本
└── plots/            # 统计图表
    ├── chunks_per_proof.png
    ├── chunk_lengths.png
    └── length_distributions.png
```

## 样本示例

打开 `train.jsonl` 会看到：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Theorem:\ntheorem add_comm...\n\nPrevious progress:\n...\n\nContinue:"
    },
    {
      "role": "assistant",
      "content": "<proof>\ninduction a with\n...\n</proof>"
    }
  ]
}
```

## 下一步

1. **检查数据质量**
   ```bash
   python data_preparation/analyze_data.py
   ```

2. **调整配置**
   - 编辑 `config.yaml`
   - 修改切分策略、块大小等

3. **模型训练**
   - 使用 `train.jsonl` 和 `val.jsonl`
   - 参考 README.md 的训练示例

## 常见问题

**Q: 下载很慢？**
A: 正常，数据集较大。可以先运行 `test_quick.py` 测试。

**Q: 想要更多数据？**
A: 修改 `config.yaml` 中的 `dataset.splits: ["valid", "test"]`

**Q: 切分效果不好？**
A: 试试不同策略：
```bash
python run_pipeline.py --strategy syntax_based
```

## 需要帮助？

- 📖 详细文档: `README.md`
- 📝 项目概览: `PROJECT_OVERVIEW.md`
- 🧪 测试代码: `test_quick.py`
- ⚙️ 配置说明: `config.yaml` (有注释)

---

**开始愉快的实验吧！** 🎉
