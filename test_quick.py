#!/usr/bin/env python3
"""
快速测试脚本 - 不下载数据，使用模拟数据测试流程
"""

import sys
from pathlib import Path
import yaml
import json

sys.path.insert(0, str(Path(__file__).parent))

from data_preparation.chunk_proofs import ProofChunker
from data_preparation.build_training_data import TrainingDataBuilder


def create_mock_examples():
    """创建模拟样本"""
    return [
        {
            'id': 0,
            'theorem': 'theorem add_comm (a b : ℕ) : a + b = b + a := by',
            'proof': '''by
  intro a b
  induction a with
  | zero =>
    simp
    apply Nat.zero_add
  | succ n ih =>
    rw [Nat.succ_add]
    rw [ih]
    rw [Nat.add_succ]
    rfl''',
            'split': 'test'
        },
        {
            'id': 1,
            'theorem': 'theorem mul_comm (a b : ℕ) : a * b = b * a := by',
            'proof': '''by
  intro a b
  induction a with
  | zero => simp
  | succ n ih =>
    rw [Nat.succ_mul]
    rw [ih]
    rw [Nat.mul_succ]
    ring''',
            'split': 'test'
        },
        {
            'id': 2,
            'theorem': 'theorem fermat_little (p : ℕ) (hp : Prime p) (a : ℕ) : p ∣ (a^p - a) := by',
            'proof': '''by
  intro p hp a
  induction a with
  | zero =>
    simp
    apply dvd_zero
  | succ n ih =>
    have h1 : p ∣ (n^p - n) := ih
    have h2 : p ∣ p * n^(p-1) := dvd_mul_right p (n^(p-1))
    rw [Nat.succ_eq_add_one]
    rw [Nat.add_pow]
    ring_nf
    apply dvd_add
    · exact h1
    · apply dvd_mul_of_dvd_each_mul_of_dvd_sub
      · exact h2
      · apply Nat.Prime.dvd_factorial
        exact hp
        linarith''',
            'split': 'test'
        }
    ]


def test_chunking(config):
    """测试切分功能"""
    print("\n" + "="*60)
    print("测试1: 证明切分")
    print("="*60)

    examples = create_mock_examples()
    chunker = ProofChunker(config)

    for ex in examples:
        print(f"\n--- 测试样本 {ex['id']} ---")
        print(f"定理: {ex['theorem'][:60]}...")

        chunks = chunker.chunk_proof(ex['proof'], ex['theorem'])

        print(f"切分结果: {len(chunks)} 块")
        for chunk in chunks:
            print(f"  Chunk {chunk['chunk_id']}: lines {chunk['start_line']}-{chunk['end_line']}")
            print(f"    内容前50字: {chunk['content'][:50]}...")

            # 测试摘要
            summary = chunker.extract_summary(chunk)
            print(f"    头部摘要: {summary['head'][:30]}...")
            print(f"    尾部摘要: {summary['tail'][-30:]}...")

    print("\n✅ 切分测试通过!")


def test_training_data(config):
    """测试训练数据构造"""
    print("\n" + "="*60)
    print("测试2: 训练数据构造")
    print("="*60)

    examples = create_mock_examples()
    builder = TrainingDataBuilder(config)

    all_samples = []
    for ex in examples:
        samples = builder.build_samples_for_proof(ex)
        all_samples.extend(samples)

        print(f"\n--- 样本 {ex['id']} ---")
        print(f"原始证明: {len(ex['proof'].split())} 行")
        print(f"生成样本: {len(samples)} 个")

        # 显示第一个样本
        if samples:
            sample = samples[0]
            print(f"\n第一个样本预览:")
            print(f"  Chunk ID: {sample['metadata']['chunk_id']}")
            print(f"  Total Chunks: {sample['metadata']['total_chunks']}")

            user_msg = sample['messages'][0]['content']
            asst_msg = sample['messages'][1]['content']

            print(f"\n  User message (前100字):")
            print(f"    {user_msg[:100]}...")

            print(f"\n  Assistant message (前100字):")
            print(f"    {asst_msg[:100]}...")

    print(f"\n总计生成 {len(all_samples)} 个训练样本")

    # 保存测试数据
    output_file = Path("./output/test_samples.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n💾 测试样本已保存到: {output_file}")
    print("\n✅ 训练数据构造测试通过!")


def test_strategies(config):
    """测试不同切分策略"""
    print("\n" + "="*60)
    print("测试3: 不同切分策略对比")
    print("="*60)

    example = create_mock_examples()[2]  # 使用最长的证明
    strategies = ['line_based', 'syntax_based', 'token_based']

    for strategy in strategies:
        print(f"\n--- {strategy} ---")
        config['chunking']['strategy'] = strategy
        chunker = ProofChunker(config)

        chunks = chunker.chunk_proof(example['proof'], example['theorem'])
        print(f"  切分块数: {len(chunks)}")
        for chunk in chunks:
            lines = chunk['content'].split('\n')
            print(f"    Chunk {chunk['chunk_id']}: {len(lines)} 行")

    print("\n✅ 策略对比测试通过!")


def main():
    """主测试函数"""
    print("="*60)
    print("Lean4 Delethink 快速测试")
    print("="*60)

    # 加载配置
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    try:
        # 运行测试
        test_chunking(config)
        test_training_data(config)
        test_strategies(config)

        print("\n" + "="*60)
        print("🎉 所有测试通过!")
        print("="*60)

        print("\n💡 下一步:")
        print("  1. 运行完整流程: python run_pipeline.py --steps all")
        print("  2. 查看测试输出: output/test_samples.jsonl")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
