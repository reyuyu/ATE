"""Retry a stopped StageA shard without rewriting its immutable collector."""
import argparse
import json
import os
import time
from .protocol import OUT
from .run import verify,worker
from paper_rq1.run import preflight
from benchmarks.dataset_registry import digest


def archive_attempt(folder):
    archive=folder/'attempts'/str(time.time_ns());archive.mkdir(parents=True)
    paths=[folder/n for n in ['environment.json','resource_check.json','worker.log']]+list(folder.glob('failure_*.json'))
    for path in paths:
        if path.exists():os.replace(path,archive/path.name)
    return archive


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['stage_a','smoke']);p.add_argument('--rank',type=int,choices=range(3),required=True);args=p.parse_args()
    config=verify();folder=OUT/f'{args.stage}/shard{args.rank}'
    if (folder/'completion.json').exists():
        r=json.loads((folder/'completion.json').read_text())
        if r['status']!='complete' or r['binding']!=digest(config):raise ValueError('Incompatible completed collection')
        print('Already complete, no GPU allocation');return
    preflight(args.rank+1)
    archive_attempt(folder)
    worker(args.stage,args.rank)


if __name__=='__main__':main()
