import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def dump_yaml(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def build_variants(args):
    common_adaptive_cfg = {
        'min_stride': args.adaptive_min_stride,
        'max_stride': args.adaptive_max_stride,
        'confidence_threshold': args.adaptive_confidence_threshold,
        'ema_decay': args.adaptive_ema_decay,
        'quantile_low': args.adaptive_quantile_low,
        'quantile_high': args.adaptive_quantile_high,
    }
    return [
        {
            'name': 'stride1',
            'stride': 1,
            'split_size_override': args.split_size_override,
            'adaptive_stride': {'enabled': False, **common_adaptive_cfg},
        },
        {
            'name': 'stride2',
            'stride': 2,
            'split_size_override': args.split_size_override,
            'adaptive_stride': {'enabled': False, **common_adaptive_cfg},
        },
        {
            'name': 'stride4',
            'stride': 4,
            'split_size_override': args.split_size_override,
            'adaptive_stride': {'enabled': False, **common_adaptive_cfg},
        },
        {
            'name': 'adaptive',
            'stride': args.adaptive_base_stride,
            'split_size_override': args.split_size_override,
            'adaptive_stride': {'enabled': True, **common_adaptive_cfg},
        },
    ]


def make_temp_config(base_cfg, variant, temp_dir):
    config_copy = json.loads(json.dumps(base_cfg))
    config_copy['data']['stride'] = variant['stride']
    config_copy['data']['adaptive_stride'] = variant['adaptive_stride']
    if variant.get('split_size_override') is not None:
        config_copy['data']['split_size'] = variant['split_size_override']
    temp_path = Path(temp_dir) / f'{variant["name"]}.yaml'
    dump_yaml(temp_path, config_copy)
    return temp_path


def ensure_checkpoint_available(base_cfg, args):
    model_dir = Path(base_cfg['training']['model_dir']).resolve()
    ckpt_dir = model_dir / 'ckpts'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    expected_ckpt_path = ckpt_dir / args.ckpt_name
    if expected_ckpt_path.exists() or not args.ckpt_path:
        return expected_ckpt_path

    source_ckpt_path = Path(args.ckpt_path).resolve()
    if not source_ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint path does not exist: {source_ckpt_path}')

    if expected_ckpt_path.exists() or expected_ckpt_path.is_symlink():
        expected_ckpt_path.unlink()

    os.symlink(source_ckpt_path, expected_ckpt_path)
    return expected_ckpt_path


def run_prediction(prediction_script, config_path, split, save_subdir, args):
    command = [
        args.python,
        str(prediction_script),
        '--config',
        str(config_path),
        '--split',
        split,
        '--save_subdir',
        save_subdir,
        '--ckpt_name',
        args.ckpt_name,
        '--eval_setting',
        args.eval_setting,
        '--pred_src',
        args.pred_src,
        '--blank_thr',
        str(args.blank_thr),
    ]
    if args.config_ex:
        command.extend(['--config_ex', args.config_ex])
    if args.save_fea:
        command.extend(['--save_fea', '1'])

    subprocess.run(command, check=True, cwd=str(prediction_script.parent))


def load_evaluation_results(model_dir, save_subdir, split):
    result_path = Path(model_dir) / save_subdir / split / f'{split}_evaluation_results.pkl'
    if not result_path.exists():
        raise FileNotFoundError(f'Evaluation results not found: {result_path}')
    with open(result_path, 'rb') as handle:
        return pickle.load(handle), result_path


def flatten_wer_results(variant_name, split, evaluation_results, result_path):
    methods = []
    best_method = None
    best_wer = None
    for key, value in evaluation_results.items():
        if not key.startswith('wer_'):
            continue
        method = key.replace('wer_', '', 1)
        wer = float(value['wer'])
        methods.append(
            {
                'method': method,
                'wer': wer,
                'del': float(value['del']),
                'ins': float(value['ins']),
                'sub': float(value['sub']),
            }
        )
        if best_wer is None or wer < best_wer:
            best_wer = wer
            best_method = method

    methods.sort(key=lambda item: item['wer'])
    return {
        'variant': variant_name,
        'split': split,
        'best_method': best_method,
        'best_wer': best_wer,
        'methods': methods,
        'result_path': str(result_path),
    }


def render_markdown(summary_rows):
    lines = [
        '| Variant | Split | Best Method | Best WER | Result Path |',
        '| --- | --- | --- | ---: | --- |',
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {row['split']} | {row['best_method']} | {row['best_wer']:.2f} | {row['result_path']} |"
        )
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser('Run fixed/adaptive stride ablations for online CSLR')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--python', default=sys.executable, type=str)
    parser.add_argument('--config_ex', default=None, type=str)
    parser.add_argument('--prediction_script', default='prediction_slide.py', type=str)
    parser.add_argument('--ckpt_name', default='best.ckpt', type=str)
    parser.add_argument('--ckpt_path', default=None, type=str)
    parser.add_argument('--pred_src', default='ensemble', choices=['ensemble', 'fuse'], type=str)
    parser.add_argument('--eval_setting', default='stride_ablation', type=str)
    parser.add_argument('--blank_thr', default=0.5, type=float)
    parser.add_argument('--save_fea', action='store_true')
    parser.add_argument('--split_size_override', default=None, type=int)
    parser.add_argument('--adaptive_base_stride', default=1, type=int)
    parser.add_argument('--adaptive_min_stride', default=1, type=int)
    parser.add_argument('--adaptive_max_stride', default=4, type=int)
    parser.add_argument('--adaptive_confidence_threshold', default=0.2, type=float)
    parser.add_argument('--adaptive_ema_decay', default=0.6, type=float)
    parser.add_argument('--adaptive_quantile_low', default=0.3, type=float)
    parser.add_argument('--adaptive_quantile_high', default=0.8, type=float)
    parser.add_argument('--output_json', default=None, type=str)
    parser.add_argument('--output_md', default=None, type=str)
    parser.add_argument('--reuse_existing', action='store_true')
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    prediction_script = Path(args.prediction_script)
    if not prediction_script.is_absolute():
        prediction_script = (config_path.parent.parent / prediction_script).resolve()
    base_cfg = load_yaml(config_path)
    ensure_checkpoint_available(base_cfg, args)
    model_dir = Path(base_cfg['training']['model_dir']).resolve()

    variants = build_variants(args)
    summary_rows = []
    raw_results = {'config': str(config_path), 'prediction_script': str(prediction_script), 'variants': []}

    with tempfile.TemporaryDirectory(prefix='stride_ablation_') as temp_dir:
        for variant in variants:
            temp_config_path = make_temp_config(base_cfg, variant, temp_dir)
            save_subdir = f'stride_ablation_{variant["name"]}'
            variant_entry = {
                'name': variant['name'],
                'stride': variant['stride'],
                'adaptive_stride': variant['adaptive_stride'],
                'split_size_override': variant.get('split_size_override'),
                'splits': [],
            }

            for split in ['dev', 'test']:
                if not args.reuse_existing:
                    run_prediction(prediction_script, temp_config_path, split, save_subdir, args)

                evaluation_results, result_path = load_evaluation_results(model_dir, save_subdir, split)
                flattened = flatten_wer_results(variant['name'], split, evaluation_results, result_path)
                summary_rows.append(flattened)
                variant_entry['splits'].append(flattened)

            raw_results['variants'].append(variant_entry)

    json_output_path = Path(args.output_json) if args.output_json else model_dir / 'stride_ablation_summary.json'
    md_output_path = Path(args.output_md) if args.output_md else model_dir / 'stride_ablation_summary.md'

    json_output_path.write_text(json.dumps(raw_results, ensure_ascii=False, indent=2), encoding='utf-8')
    md_output_path.write_text(render_markdown(summary_rows), encoding='utf-8')

    print(render_markdown(summary_rows))
    print(f'JSON summary saved to: {json_output_path}')
    print(f'Markdown summary saved to: {md_output_path}')


if __name__ == '__main__':
    main()