#!/usr/bin/env python3
"""
一键运行完整的数据准备流程
"""

import sys
import argparse
from pathlib import Path
import yaml

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from data_preparation.download_data import MiniF2FLoader
from data_preparation.build_training_data import TrainingDataBuilder
from data_preparation.analyze_data import DataAnalyzer


def run_download(config):
    """步骤1: 下载数据"""
    print("\n" + "="*60)
    print("STEP 1: Downloading MiniF2F-Lean4 Dataset")
    print("="*60)

    loader = MiniF2FLoader(config)

    # 下载数据
    data = loader.download()

    # 保存原始数据
    loader.save_raw_data(data)

    # 过滤数据
    for split in data.keys():
        print(f"\n{'='*60}")
        print(f"Filtering {split} split")
        print('='*60)

        filtered = loader.filter_data(data[split])

        # 保存过滤后的数据
        output_file = loader.cache_dir / f"{split}_filtered.jsonl"
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            for ex in filtered:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        print(f"\n✓ Saved filtered data to {output_file}")

    print("\n✅ Step 1 completed!")


def run_build_training_data(config):
    """步骤2: 构造训练数据"""
    print("\n" + "="*60)
    print("STEP 2: Building Training Data")
    print("="*60)

    loader = MiniF2FLoader(config)
    builder = TrainingDataBuilder(config)

    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in config['dataset']['splits']:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print('='*60)

        # 加载过滤后的数据
        try:
            import json
            filtered_file = loader.cache_dir / f"{split}_filtered.jsonl"

            if not filtered_file.exists():
                print(f"⚠️  Filtered file not found: {filtered_file}")
                print(f"💡 Skipping {split}")
                continue

            examples = []
            with open(filtered_file, 'r', encoding='utf-8') as f:
                for line in f:
                    examples.append(json.loads(line))

            print(f"📖 Loaded {len(examples)} filtered examples")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            continue

        # 构造训练样本
        samples = builder.build_dataset(examples)

        # 保存完整数据集
        output_file = output_dir / f"{split}_training.jsonl"
        builder.save_dataset(
            samples,
            output_file,
            format=config['training']['output_format']
        )

        # 如果是 valid split，进一步分割为 train/val
        if split == "valid":
            train_samples, val_samples = builder.split_train_val(
                samples,
                ratio=config['training']['train_split_ratio']
            )

            train_file = output_dir / "train.jsonl"
            val_file = output_dir / "val.jsonl"

            builder.save_dataset(train_samples, train_file, format='jsonl')
            builder.save_dataset(val_samples, val_file, format='jsonl')

    print("\n✅ Step 2 completed!")


def run_analyze(config):
    """步骤3: 数据分析"""
    print("\n" + "="*60)
    print("STEP 3: Analyzing Data")
    print("="*60)

    analyzer = DataAnalyzer(config)
    data_dir = Path(config['training']['output_dir'])

    # 分析所有数据文件
    for data_file in sorted(data_dir.glob("*.jsonl")):
        print(f"\n{'='*60}")
        print(f"Analyzing: {data_file.name}")
        print('='*60)

        # 加载数据
        samples = analyzer.load_samples(data_file)

        # 计算统计
        stats = analyzer.compute_statistics(samples)

        # 打印统计
        analyzer.print_statistics(stats, title=f"Statistics: {data_file.stem}")

        # 绘制分布图
        if config['analysis']['plot_distributions']:
            plot_dir = data_dir / "plots" / data_file.stem
            analyzer.plot_distributions(stats, plot_dir)

        # 展示样本
        if config['analysis']['generate_stats']:
            analyzer.show_samples(samples, n=config['analysis']['sample_size'])

    print("\n✅ Step 3 completed!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Lean4 Delethink Data Preparation Pipeline"
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Steps to run: all, download, build, analyze (comma-separated)'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        choices=['line_based', 'syntax_based', 'token_based'],
        help='Chunking strategy (overrides config)'
    )

    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 覆盖策略
    if args.strategy:
        config['chunking']['strategy'] = args.strategy
        print(f"🔧 Using chunking strategy: {args.strategy}")

    # 确定要运行的步骤
    if args.steps == 'all':
        steps = ['download', 'build', 'analyze']
    else:
        steps = [s.strip() for s in args.steps.split(',')]

    print("\n" + "="*60)
    print("Lean4 Delethink Data Preparation Pipeline")
    print("="*60)
    print(f"\n📋 Steps to run: {', '.join(steps)}")
    print(f"🔧 Chunking strategy: {config['chunking']['strategy']}")

    # 运行步骤
    try:
        if 'download' in steps:
            run_download(config)

        if 'build' in steps:
            run_build_training_data(config)

        if 'analyze' in steps:
            run_analyze(config)

        print("\n" + "="*60)
        print("🎉 Pipeline completed successfully!")
        print("="*60)

        # 输出位置
        output_dir = Path(config['training']['output_dir'])
        print(f"\n📁 Output files:")
        print(f"  Data: {output_dir}")
        if (output_dir / "plots").exists():
            print(f"  Plots: {output_dir / 'plots'}")

        # 下一步提示
        print(f"\n💡 Next steps:")
        print(f"  1. Review the generated samples in {output_dir}")
        print(f"  2. Check the statistics and plots")
        print(f"  3. Use train.jsonl and val.jsonl for model training")

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
