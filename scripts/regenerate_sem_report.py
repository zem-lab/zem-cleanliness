"""Regenerate a stopped/completed SEM task's formal report without hardware."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.cleanliness_core.atomic_json import write_json_atomic
from backend.cleanliness_core.sem_direct_report import build_sem_reports
from backend.cleanliness_core.sem_task_store import load_sem_task
from backend.cleanliness_core.sem_progress import refresh_inactive_header

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task_directory', type=Path)
    args = parser.parse_args()
    directory = args.task_directory.resolve()
    manifest = load_sem_task(directory)
    if manifest.get('mode') != 'sem_direct' or manifest['status'] == 'RUNNING':
        raise ValueError('Only inactive direct SEM tasks can regenerate reports')
    build_sem_reports(directory, manifest)
    manifest['reportState'] = 'READY'
    manifest.pop('reportError', None)
    write_json_atomic(directory/'task.json', manifest)
    refresh_inactive_header(directory,manifest)
    print(json.dumps({k:str(directory/v) for k,v in manifest['reports'].items()}, ensure_ascii=False))
