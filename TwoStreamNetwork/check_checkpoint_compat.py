import argparse
import gzip
import json
import os
import pickle
import sys
import tempfile
from collections import Counter, defaultdict

import torch

sys.path.append(os.getcwd())

from dataset.Dataset import get_keypoints_num
from modelling.model import build_model
from utils.misc import load_checkpoint, load_config, make_logger


def safe_len(path):
    if not os.path.isfile(path):
        return None
    if path.endswith('.pkl'):
        with open(path, 'rb') as handle:
            obj = pickle.load(handle)
        return len(obj)
    if path.endswith('.train') or path.endswith('.dev') or path.endswith('.test'):
        with gzip.open(path, 'rb') as handle:
            obj = pickle.load(handle)
        return len(obj)
    return None


def summarize_dataset(cfg):
    data_cfg = cfg['data']
    summary = {
        'dataset_name': data_cfg.get('dataset_name'),
        'zip_file': data_cfg.get('zip_file'),
        'zip_exists': os.path.exists(data_cfg.get('zip_file', '')),
        'train_file': data_cfg.get('train'),
        'train_exists': os.path.exists(data_cfg.get('train', '')),
        'dev_file': data_cfg.get('dev'),
        'dev_exists': os.path.exists(data_cfg.get('dev', '')),
        'test_file': data_cfg.get('test'),
        'test_exists': os.path.exists(data_cfg.get('test', '')),
        'gloss2id_file': cfg['model']['RecognitionNetwork']['GlossTokenizer'].get('gloss2id_file'),
        'gloss2id_exists': os.path.exists(cfg['model']['RecognitionNetwork']['GlossTokenizer'].get('gloss2id_file', '')),
        'keypoint_file': data_cfg.get('keypoint_file'),
        'keypoint_exists': os.path.exists(data_cfg.get('keypoint_file', '')),
        'use_keypoints': data_cfg.get('use_keypoints', []),
        'batch_size': cfg['training'].get('batch_size'),
        'max_num_frames': data_cfg.get('max_num_frames', 400),
    }
    summary['train_samples'] = safe_len(data_cfg.get('train', ''))
    summary['dev_samples'] = safe_len(data_cfg.get('dev', ''))
    summary['test_samples'] = safe_len(data_cfg.get('test', ''))
    summary['gloss_vocab_size'] = safe_len(summary['gloss2id_file'])
    if summary['keypoint_exists']:
        summary['keypoint_in_channels'] = get_keypoints_num(summary['keypoint_file'], summary['use_keypoints'])
    else:
        summary['keypoint_in_channels'] = None
    return summary


def collect_representative_keys(model_state, ckpt_state):
    candidates = [
        'recognition_network.visual_head.gloss_output_layer.weight',
        'recognition_network.visual_head_keypoint.gloss_output_layer.weight',
        'recognition_network.visual_head_fuse.gloss_output_layer.weight',
        'recognition_network.visual_backbone_twostream.pose_stream.backbone.base.0.conv_s.weight',
        'recognition_network.visual_backbone_twostream.rgb_stream_lateral.0.conv.weight',
        'recognition_network.visual_backbone_twostream.pose_stream_lateral.0.conv.weight',
    ]
    rows = []
    for key in candidates:
        rows.append(
            {
                'key': key,
                'checkpoint_shape': tuple(ckpt_state[key].shape) if key in ckpt_state else None,
                'current_shape': tuple(model_state[key].shape) if key in model_state else None,
                'matched': key in ckpt_state and key in model_state and tuple(ckpt_state[key].shape) == tuple(model_state[key].shape),
            }
        )
    return rows


def compare_states(model_state, ckpt_state, example_limit):
    summary = Counter()
    category_counter = Counter()
    examples = []
    matched_keys = []
    skipped_keys = []
    missing_in_ckpt = []

    for key, tensor in ckpt_state.items():
        if key not in model_state:
            summary['missing_key_in_current_model'] += 1
            skipped_keys.append(key)
            category_counter[key.split('.')[1] if key.startswith('recognition_network.') and len(key.split('.')) > 1 else key.split('.')[0]] += 1
            if len(examples) < example_limit:
                examples.append({'key': key, 'reason': 'missing_key_in_current_model'})
            continue
        current_shape = tuple(model_state[key].shape)
        checkpoint_shape = tuple(tensor.shape)
        if current_shape != checkpoint_shape:
            summary['shape_mismatch'] += 1
            skipped_keys.append(key)
            category_counter[key.split('.')[1] if key.startswith('recognition_network.') and len(key.split('.')) > 1 else key.split('.')[0]] += 1
            if len(examples) < example_limit:
                examples.append({'key': key, 'reason': 'shape_mismatch', 'checkpoint_shape': checkpoint_shape, 'current_shape': current_shape})
            continue
        summary['matched'] += 1
        matched_keys.append(key)

    for key in model_state.keys():
        if key not in ckpt_state:
            summary['missing_key_in_checkpoint'] += 1
            if len(missing_in_ckpt) < example_limit:
                missing_in_ckpt.append(key)

    return {
        'summary': dict(summary),
        'skip_categories': dict(category_counter),
        'matched_keys': matched_keys,
        'skipped_keys': skipped_keys,
        'example_skips': examples,
        'example_missing_in_checkpoint': missing_in_ckpt,
    }


def infer_findings(dataset_summary, representative_rows):
    findings = []

    head_row = next((row for row in representative_rows if row['key'] == 'recognition_network.visual_head.gloss_output_layer.weight'), None)
    if head_row and head_row['checkpoint_shape'] and head_row['current_shape']:
        if head_row['checkpoint_shape'][0] != head_row['current_shape'][0]:
            findings.append(
                'Gloss output layer row count differs. This usually means the checkpoint vocabulary size does not match the current gloss vocabulary.'
            )
        if len(head_row['checkpoint_shape']) > 1 and len(head_row['current_shape']) > 1 and head_row['checkpoint_shape'][1] != head_row['current_shape'][1]:
            findings.append(
                'Gloss output layer hidden dimension differs. This indicates the recognition head configuration changed, not just the dataset.'
            )

    pose_row = next((row for row in representative_rows if row['key'] == 'recognition_network.visual_backbone_twostream.pose_stream.backbone.base.0.conv_s.weight'), None)
    if pose_row and pose_row['checkpoint_shape'] and pose_row['current_shape']:
        if len(pose_row['checkpoint_shape']) > 1 and len(pose_row['current_shape']) > 1 and pose_row['checkpoint_shape'][1] != pose_row['current_shape'][1]:
            findings.append(
                'Pose stream input channels differ. This usually means the checkpoint used a different keypoint selection or keypoint file format than the current config.'
            )

    lateral_row = next((row for row in representative_rows if row['key'] == 'recognition_network.visual_backbone_twostream.rgb_stream_lateral.0.conv.weight'), None)
    if lateral_row and lateral_row['checkpoint_shape'] and lateral_row['current_shape']:
        if lateral_row['checkpoint_shape'] != lateral_row['current_shape']:
            findings.append(
                'Lateral connection kernel shape differs. This indicates a model-structure mismatch in lateral or pyramid settings.'
            )

    if dataset_summary['gloss_vocab_size'] is None:
        findings.append('Current gloss vocabulary file could not be read, so vocabulary compatibility could not be fully checked.')

    if len(findings) == 0:
        findings.append('No obvious dataset- or structure-level mismatch was detected in the representative keys.')

    return findings


def main():
    parser = argparse.ArgumentParser(description='Check whether a checkpoint is compatible with the current TwoStream config and data-derived model definition.')
    parser.add_argument('--config', required=True, help='Path to the YAML config file.')
    parser.add_argument('--ckpt', required=True, help='Path to the checkpoint file.')
    parser.add_argument('--example-limit', type=int, default=12, help='How many example mismatches to print.')
    parser.add_argument('--json', action='store_true', help='Print a JSON report instead of human-readable text.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg['device'] = torch.device('cpu')
    dataset_summary = summarize_dataset(cfg)

    logger_dir = tempfile.mkdtemp(prefix='ckpt_compat_')
    make_logger(model_dir=logger_dir, log_file='check.log')

    model = build_model(cfg)
    model_state = model.state_dict()

    checkpoint = load_checkpoint(args.ckpt, map_location='cpu')
    ckpt_state = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))

    comparison = compare_states(model_state, ckpt_state, example_limit=args.example_limit)
    representative_rows = collect_representative_keys(model_state, ckpt_state)
    findings = infer_findings(dataset_summary, representative_rows)

    report = {
        'config': args.config,
        'checkpoint': args.ckpt,
        'dataset_summary': dataset_summary,
        'comparison': comparison,
        'representative_keys': representative_rows,
        'findings': findings,
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print('=== Dataset / Config Summary ===')
    for key, value in dataset_summary.items():
        print(f'{key}: {value}')

    print('\n=== State Dict Summary ===')
    for key, value in comparison['summary'].items():
        print(f'{key}: {value}')

    print('\n=== Skip Categories ===')
    for key, value in sorted(comparison['skip_categories'].items()):
        print(f'{key}: {value}')

    print('\n=== Representative Keys ===')
    for row in representative_rows:
        print(f"{row['key']}: ckpt={row['checkpoint_shape']} current={row['current_shape']} matched={row['matched']}")

    print('\n=== Example Skips ===')
    for item in comparison['example_skips']:
        print(item)

    print('\n=== Example Missing In Checkpoint ===')
    for key in comparison['example_missing_in_checkpoint']:
        print(key)

    print('\n=== Findings ===')
    for finding in findings:
        print(f'- {finding}')


if __name__ == '__main__':
    main()