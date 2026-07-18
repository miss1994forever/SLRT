#!/usr/bin/env python3
"""
Convert Online CSLR prediction results to G2T training format
"""
import pickle
import os
import sys
from collections import defaultdict

def load_cslr_predictions(pred_dir, split='dev'):
    """Load CSLR prediction results"""
    # Try both locations: direct and in subdirectory
    results_file = os.path.join(pred_dir, f'{split}_results.pkl')
    if not os.path.exists(results_file):
        results_file = os.path.join(pred_dir, split, f'{split}_results.pkl')

    if not os.path.exists(results_file):
        print(f"Error: Prediction file not found. Tried:")
        print(f"  - {os.path.join(pred_dir, f'{split}_results.pkl')}")
        print(f"  - {os.path.join(pred_dir, split, f'{split}_results.pkl')}")
        return None

    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    print(f"Loaded {len(results)} results from {results_file}")
    return results

def convert_to_g2t_format(results, decode_method='window_greedy_5_gls_hyp'):
    """
    Convert CSLR results to G2T format
    Expected output format: [{'name': str, 'num_frames': int, 'gloss': str}, ...]

    Results format from CSLR:
    {
        'video_name': {
            'gls_ref': 'reference gloss',
            'window_greedy_5_gls_hyp': 'predicted gloss',
            ...
        }
    }
    """
    g2t_data = []

    for name, result in results.items():
        # Find prediction key
        pred_key = None
        if decode_method in result:
            pred_key = decode_method
        else:
            # Try to find any available decode method
            for key in result.keys():
                if 'gls_hyp' in key:
                    pred_key = key
                    break

            if pred_key is None:
                print(f"Warning: No gloss hypothesis found for {name}, skipping")
                continue

            if decode_method == 'window_greedy_5_gls_hyp':  # Only print once
                print(f"Using decode method: {pred_key}")
                decode_method = pred_key  # Update for next iterations

        # Extract predicted gloss
        pred_gloss = result[pred_key]

        # Gloss should already be a string
        if not isinstance(pred_gloss, str):
            if isinstance(pred_gloss, list):
                pred_gloss = ' '.join(pred_gloss)
            else:
                print(f"Warning: Unexpected gloss type for {name}: {type(pred_gloss)}")
                continue

        # For num_frames, we don't have it in this format, so set to 0
        # It's not critical for G2T training which uses glosses only
        g2t_data.append({
            'name': name,
            'num_frames': 0,  # Not available in this format
            'gloss': pred_gloss
        })

    return g2t_data

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_g2t_data.py <pred_dir> <output_dir>")
        print("Example: python generate_g2t_data.py results/csl-daily-top-800_ISLR/prediction_slide results/online_slr_csl/prediction_slide")
        sys.exit(1)

    pred_dir = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    for split in ['dev', 'test']:
        print(f"\nProcessing {split} split...")

        results = load_cslr_predictions(pred_dir, split)
        if results is None:
            print(f"Skipping {split} (file not found)")
            continue

        g2t_data = convert_to_g2t_format(results)

        if not g2t_data:
            print(f"Warning: No data generated for {split}")
            continue

        output_file = os.path.join(output_dir, f'csl_pred.{split}')
        with open(output_file, 'wb') as f:
            pickle.dump(g2t_data, f)

        print(f"Saved {len(g2t_data)} entries to {output_file}")

        # Print sample
        if g2t_data:
            print(f"Sample entry: {g2t_data[0]}")

if __name__ == '__main__':
    main()
