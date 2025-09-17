import os
import sys
import argparse
import csv
from pathlib import Path
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.util import tensor_util
except Exception as e:
    print("Missing tensorboard: pip install tensorboard", file=sys.stderr)
    raise

# Optional fallback parser
try:
    from tbparse import SummaryReader  # type: ignore
except Exception:
    SummaryReader = None  # type: ignore


def export_scalars(run_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    size_guidance = {
        'scalars': 0,
        'tensors': 0,
        'histograms': 0,
        'images': 0,
        'compressedHistograms': 0,
        'audio': 0,
    }
    ea = EventAccumulator(str(run_dir), size_guidance=size_guidance)
    ea.Reload()
    tags_all = ea.Tags()
    print(f"  EventAccumulator TAGS: scalars={len(tags_all.get('scalars', []))} tensors={len(tags_all.get('tensors', []))} hist={len(tags_all.get('histograms', []))}")
    tags = tags_all.get('scalars', [])
    summary = {}
    for tag in tags:
        events = ea.Scalars(tag)
        if not events:
            continue
        csv_path = out_dir / f"{tag.replace('/', '_')}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['step', 'wall_time', 'value'])
            for ev in events:
                w.writerow([ev.step, ev.wall_time, ev.value])
        # quick summary stats
        vals = [ev.value for ev in events]
        summary[tag] = {
            'count': len(vals),
            'min': min(vals),
            'max': max(vals),
            'mean_last_10': sum(vals[-10:]) / max(1, min(10, len(vals))),
        }
    # Also handle tensor summaries (some loggers write scalars as tensors)
    tensor_tags = tags_all.get('tensors', [])
    for tag in tensor_tags:
        events = ea.Tensors(tag)
        if not events:
            continue
        rows = []
        vals = []
        for ev in events:
            try:
                arr = tensor_util.make_ndarray(ev.tensor_proto)
                # Accept 0-D or shape (1,) /(1,1) numeric tensors
                val = None
                if arr.shape == ():
                    val = float(arr)
                elif arr.size == 1:
                    val = float(arr.reshape(-1)[0])
                else:
                    # Skip non-scalar tensors
                    continue
                rows.append((ev.step, ev.wall_time, val))
                vals.append(val)
            except Exception:
                continue
        if not rows:
            continue
        csv_path = out_dir / f"tensor_{tag.replace('/', '_')}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['step', 'wall_time', 'value'])
            for row in rows:
                w.writerow(list(row))
        summary[tag] = {
            'count': len(vals),
            'min': min(vals),
            'max': max(vals),
            'mean_last_10': sum(vals[-10:]) / max(1, min(10, len(vals))),
        }
    # If no data was found via EventAccumulator, try tbparse fallback
    if not summary and SummaryReader is not None:
        try:
            sr = SummaryReader(str(run_dir), pivot=False)
            df = sr.scalars
            # df columns: ['run', 'tag', 'step', 'value', 'wall_time'] when pivot=False
            if df is not None and not df.empty:
                long_df = df
                for tag, sub in long_df.groupby('tag'):
                    sub = sub.sort_values('step')
                    csv_path = out_dir / f"{str(tag).replace('/', '_')}.csv"
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        w = csv.writer(f)
                        w.writerow(['step', 'wall_time', 'value'])
                        for _, row in sub.iterrows():
                            w.writerow([int(row.get('step', 0)), float(row.get('wall_time', 0.0)), float(row.get('value', 0.0))])
                    vals = [float(v) for v in sub['value'].tolist() if v == v]
                    if vals:
                        summary[str(tag)] = {
                            'count': len(vals),
                            'min': min(vals),
                            'max': max(vals),
                            'mean_last_10': sum(vals[-10:]) / max(1, min(10, len(vals))),
                        }
            # Try tensors as well via tbparse
            df_t = getattr(sr, 'tensors', None)
            if df_t is not None and not df_t.empty:
                # Expect columns: ['run','tag','step','value'] where 'value' may be string repr
                for tag, sub in df_t.groupby('tag'):
                    sub = sub.sort_values('step')
                    vals = []
                    rows = []
                    for _, row in sub.iterrows():
                        v = row.get('value')
                        try:
                            # Attempt to coerce to float
                            fv = float(v)
                        except Exception:
                            continue
                        vals.append(fv)
                        rows.append((int(row.get('step', 0)), float(row.get('wall_time', 0.0)), fv))
                    if not rows:
                        continue
                    csv_path = out_dir / f"tensor_{str(tag).replace('/', '_')}.csv"
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        w = csv.writer(f)
                        w.writerow(['step', 'wall_time', 'value'])
                        for r in rows:
                            w.writerow(list(r))
                    summary[str(tag)] = {
                        'count': len(vals),
                        'min': min(vals),
                        'max': max(vals),
                        'mean_last_10': sum(vals[-10:]) / max(1, min(10, len(vals))),
                    }
        except Exception as e:
            print(f"tbparse fallback failed for {run_dir}: {e}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logdir', type=str, help='TensorBoard log directory (contains event files or subdirs)')
    ap.add_argument('--out', type=str, default='tb_exports', help='Output directory for CSV exports')
    args = ap.parse_args()

    root = Path(args.logdir)
    out_root = Path(args.out)

    # Identify runs: either event files directly in root or subfolders
    runs = []
    if any(p.name.startswith('events.out.tfevents') for p in root.iterdir() if p.is_file()):
        runs.append(root)
    for p in root.iterdir():
        if p.is_dir():
            if any(f.name.startswith('events.out.tfevents') for f in p.iterdir() if f.is_file()):
                runs.append(p)

    if not runs:
        print(f"No runs found under {root}")
        return 0

    overall = {}
    for run in runs:
        run_name = run.name
        out_dir = out_root / run_name
        print(f"Processing run: {run}")
        try:
            summary = export_scalars(run, out_dir)
            overall[run_name] = summary
        except Exception as e:
            print(f"Failed to process {run}: {e}")

    # Print concise report
    print("\n==== Summary ====")
    for run_name, summ in overall.items():
        print(f"Run: {run_name}")
        # show some key tags if present
        key_tags = [
            'rollout/ep_rew_mean', 'rollout/ep_len_mean',
            'train/value_loss', 'train/policy_gradient_loss', 'train/entropy_loss',
            'train/approx_kl', 'train/clip_fraction',
        ]
        # reward components
        # stable-baselines3 logger records under custom namespaces; we exported all tags
        comp_prefix = 'rewards/'
        comps = sorted([t for t in summ.keys() if t.startswith(comp_prefix)])
        for tag in key_tags:
            if tag in summ:
                s = summ[tag]
                print(f"  {tag:30s} count={s['count']:5d}  last10_mean={s['mean_last_10']:+.4f}  min={s['min']:+.3f}  max={s['max']:+.3f}")
        if comps:
            print("  Reward components:")
            for tag in comps:
                s = summ[tag]
                short = tag[len(comp_prefix):]
                print(f"    {short:20s} count={s['count']:5d}  last10_mean={s['mean_last_10']:+.4f}  min={s['min']:+.3f}  max={s['max']:+.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
