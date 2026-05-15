import argparse
import copy
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def ensure_checkpoint_available(base_cfg, ckpt_name, ckpt_path):
    model_dir = Path(base_cfg["training"]["model_dir"]).resolve()
    ckpt_dir = model_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    expected_ckpt_path = ckpt_dir / ckpt_name
    if expected_ckpt_path.exists() or not ckpt_path:
        return expected_ckpt_path

    source_ckpt_path = Path(ckpt_path).resolve()
    if not source_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {source_ckpt_path}")

    if expected_ckpt_path.exists() or expected_ckpt_path.is_symlink():
        expected_ckpt_path.unlink()

    expected_ckpt_path.symlink_to(source_ckpt_path)
    return expected_ckpt_path


def run_prediction(args, prediction_script, config_path, split, save_subdir):
    command = [
        args.python,
        str(prediction_script),
        "--config",
        str(config_path),
        "--split",
        split,
        "--save_subdir",
        save_subdir,
        "--ckpt_name",
        args.ckpt_name,
        "--eval_setting",
        args.eval_setting,
        "--pred_src",
        args.pred_src,
        "--blank_thr",
        str(args.blank_thr),
    ]
    if args.config_ex:
        command.extend(["--config_ex", args.config_ex])
    if args.save_fea:
        command.extend(["--save_fea", "1"])

    subprocess.run(command, check=True, cwd=str(prediction_script.parent))


def load_eval_results(model_dir, save_subdir, split):
    result_path = Path(model_dir) / save_subdir / split / f"{split}_evaluation_results.pkl"
    if not result_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {result_path}")
    with open(result_path, "rb") as handle:
        return pickle.load(handle), result_path


def best_method_wer(evaluation_results):
    best_method = None
    best_wer = None
    for key, value in evaluation_results.items():
        if not key.startswith("wer_"):
            continue
        method = key.replace("wer_", "", 1)
        wer = float(value["wer"])
        if best_wer is None or wer < best_wer:
            best_wer = wer
            best_method = method
    return best_method, best_wer


def candidate_grid():
    candidates = []
    idx = 0
    for min_stride, max_stride in [(1, 3), (1, 4), (1, 5)]:
        for quantile_low, quantile_high in [(0.2, 0.8), (0.3, 0.8), (0.2, 0.7)]:
            for ema_decay in [0.4, 0.6]:
                candidates.append(
                    {
                        "id": idx,
                        "min_stride": min_stride,
                        "max_stride": max_stride,
                        "quantile_low": quantile_low,
                        "quantile_high": quantile_high,
                        "ema_decay": ema_decay,
                    }
                )
                idx += 1
    return candidates


def make_config(base_cfg, args, candidate, temp_dir):
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["stride"] = args.adaptive_base_stride
    cfg["data"]["split_size"] = args.split_size_override
    cfg["data"]["adaptive_stride"] = {
        "enabled": True,
        "min_stride": candidate["min_stride"],
        "max_stride": candidate["max_stride"],
        "confidence_threshold": args.adaptive_confidence_threshold,
        "ema_decay": candidate["ema_decay"],
        "quantile_low": candidate["quantile_low"],
        "quantile_high": candidate["quantile_high"],
    }
    config_path = Path(temp_dir) / f"adaptive_search_{candidate['id']:02d}.yaml"
    dump_yaml(config_path, cfg)
    return config_path


def trial_name(candidate):
    return (
        f"adaptive_search_"
        f"m{candidate['min_stride']}_{candidate['max_stride']}_"
        f"q{int(candidate['quantile_low'] * 100):02d}_{int(candidate['quantile_high'] * 100):02d}_"
        f"e{int(candidate['ema_decay'] * 100):02d}"
    )


def render_markdown(dev_rows, test_rows):
    lines = [
        "| Trial | min/max | q_low/q_high | ema | dev_best_method | dev_best_WER | test_best_method | test_best_WER |",
        "| --- | --- | --- | ---: | --- | ---: | --- | ---: |",
    ]
    test_map = {row["trial"]: row for row in test_rows}
    for row in dev_rows:
        t = test_map.get(row["trial"], {})
        lines.append(
            "| {trial} | {min_stride}/{max_stride} | {quantile_low:.2f}/{quantile_high:.2f} | {ema_decay:.2f} | {dev_best_method} | {dev_best_wer:.2f} | {test_best_method} | {test_best_wer} |".format(
                trial=row["trial"],
                min_stride=row["min_stride"],
                max_stride=row["max_stride"],
                quantile_low=row["quantile_low"],
                quantile_high=row["quantile_high"],
                ema_decay=row["ema_decay"],
                dev_best_method=row["dev_best_method"],
                dev_best_wer=row["dev_best_wer"],
                test_best_method=t.get("test_best_method", "-"),
                test_best_wer=(f"{t['test_best_wer']:.2f}" if "test_best_wer" in t else "-"),
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser("Adaptive stride hyperparameter search for Online CSLR")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--python", default=sys.executable, type=str)
    parser.add_argument("--config_ex", default=None, type=str)
    parser.add_argument("--prediction_script", default="prediction_slide.py", type=str)
    parser.add_argument("--ckpt_name", default="best.ckpt", type=str)
    parser.add_argument("--ckpt_path", default=None, type=str)
    parser.add_argument("--pred_src", default="ensemble", choices=["ensemble", "fuse"], type=str)
    parser.add_argument("--eval_setting", default="adaptive_hparam_search", type=str)
    parser.add_argument("--blank_thr", default=0.5, type=float)
    parser.add_argument("--save_fea", action="store_true")
    parser.add_argument("--split_size_override", default=8, type=int)
    parser.add_argument("--adaptive_base_stride", default=1, type=int)
    parser.add_argument("--adaptive_confidence_threshold", default=0.2, type=float)
    parser.add_argument("--topk_test", default=3, type=int)
    parser.add_argument("--output_json", default=None, type=str)
    parser.add_argument("--output_md", default=None, type=str)
    parser.add_argument("--reuse_existing", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    prediction_script = Path(args.prediction_script)
    if not prediction_script.is_absolute():
        prediction_script = (config_path.parent.parent / prediction_script).resolve()

    base_cfg = load_yaml(config_path)
    ensure_checkpoint_available(base_cfg, args.ckpt_name, args.ckpt_path)
    model_dir = Path(base_cfg["training"]["model_dir"]).resolve()

    candidates = candidate_grid()
    dev_rows = []
    test_rows = []

    with tempfile.TemporaryDirectory(prefix="adaptive_hparam_") as temp_dir:
        for cand in candidates:
            cfg_path = make_config(base_cfg, args, cand, temp_dir)
            save_subdir = trial_name(cand)
            if not args.reuse_existing:
                run_prediction(args, prediction_script, cfg_path, "dev", save_subdir)
            dev_eval, dev_path = load_eval_results(model_dir, save_subdir, "dev")
            dev_best_method, dev_best_wer = best_method_wer(dev_eval)
            dev_rows.append(
                {
                    "trial": save_subdir,
                    "min_stride": cand["min_stride"],
                    "max_stride": cand["max_stride"],
                    "quantile_low": cand["quantile_low"],
                    "quantile_high": cand["quantile_high"],
                    "ema_decay": cand["ema_decay"],
                    "dev_best_method": dev_best_method,
                    "dev_best_wer": dev_best_wer,
                    "dev_result_path": str(dev_path),
                }
            )

        dev_rows.sort(key=lambda x: x["dev_best_wer"])
        chosen = dev_rows[: max(1, args.topk_test)]

        for row in chosen:
            # regenerate config from row metadata
            cand = {
                "min_stride": row["min_stride"],
                "max_stride": row["max_stride"],
                "quantile_low": row["quantile_low"],
                "quantile_high": row["quantile_high"],
                "ema_decay": row["ema_decay"],
                "id": -1,
            }
            cfg_path = make_config(base_cfg, args, cand, temp_dir)
            save_subdir = row["trial"]
            if not args.reuse_existing:
                run_prediction(args, prediction_script, cfg_path, "test", save_subdir)
            test_eval, test_path = load_eval_results(model_dir, save_subdir, "test")
            test_best_method, test_best_wer = best_method_wer(test_eval)
            test_rows.append(
                {
                    "trial": save_subdir,
                    "test_best_method": test_best_method,
                    "test_best_wer": test_best_wer,
                    "test_result_path": str(test_path),
                }
            )

    payload = {
        "config": str(config_path),
        "prediction_script": str(prediction_script),
        "split_size_override": args.split_size_override,
        "adaptive_base_stride": args.adaptive_base_stride,
        "adaptive_confidence_threshold": args.adaptive_confidence_threshold,
        "topk_test": args.topk_test,
        "dev_results": dev_rows,
        "test_results": test_rows,
    }

    json_output_path = Path(args.output_json) if args.output_json else model_dir / "adaptive_hparam_search_summary.json"
    md_output_path = Path(args.output_md) if args.output_md else model_dir / "adaptive_hparam_search_summary.md"

    json_output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_output_path.write_text(render_markdown(dev_rows, test_rows), encoding="utf-8")

    print(render_markdown(dev_rows, test_rows))
    print(f"JSON summary saved to: {json_output_path}")
    print(f"Markdown summary saved to: {md_output_path}")


if __name__ == "__main__":
    main()
