"""Benchmark three FB training backends under identical YAML settings.

Launches one process per backend to isolate JAX compilation/memory caches.
Example: python benchmark.py --config configs/default.yaml --epochs 20
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--output',default='results/backend_benchmark')
    args=parser.parse_args()
    if args.epochs<1:
        parser.error('--epochs must be positive')
    root=Path(__file__).resolve().parent
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    original=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    original['training']['epochs']=args.epochs
    original['training']['plot']=False
    original['training']['checkpoint_every']=0
    results={}
    for backend in ('jax','numpy_scan','numpy_legacy'):
        cfg=yaml.safe_load(yaml.safe_dump(original))
        cfg['training']['backend']=backend
        cfg['training']['output_dir']=str(out/backend)
        cfgfile=out/f'{backend}.yaml'
        cfgfile.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        print(f'Benchmarking {backend} ({args.epochs} epochs)...',flush=True)
        subprocess.run([sys.executable,str(root/'train.py'),'--config',str(cfgfile)],
                       cwd=root,check=True)
        result=json.loads((out/backend/'summary.json').read_text(encoding='utf-8'))
        results[backend]={k:result[k] for k in ('epochs','total_steps','updates',
                                                 'seconds','steps_per_second','final_loss')}
    (out/'benchmark.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    base=results['numpy_legacy']['seconds']
    for b,row in results.items():
        print(f"{b:14s} {row['seconds']:8.2f}s | {row['steps_per_second']:8.0f} "
              f"steps/s | {base/row['seconds']:.2f}x legacy")
    print('Saved:',out/'benchmark.json')


if __name__=='__main__':
    main()
