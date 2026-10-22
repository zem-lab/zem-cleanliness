"""Sample 2 manual AB product acceptance. --execute enables SEM validation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import server
from backend.cleanliness_core.atomic_json import write_json_atomic
from backend.cleanliness_core.cleanliness_task import refine_ab_point_calibration
from loguru import logger


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--name', default='sample2-manual-ab')
    parser.add_argument('--min-length', type=float)
    args = parser.parse_args()
    out = ROOT / 'artifacts/manual_ab_sample2_20260908'
    out.mkdir(parents=True, exist_ok=True)
    server.TASK_ROOT = out
    server.SETTINGS_PATH = out / 'app-settings.json'
    task_id = args.name
    task_dir = out / task_id
    if (task_dir / 'task.json').exists():
        raise FileExistsError('Existing test evidence must not be overwritten')
    task_dir.mkdir(exist_ok=True)
    source = Path(r'C:\path\to\optical-data\测试\2\2.fsmpe.json')
    preview = server._load_optical(source, {})
    draft = {'taskId': task_id, 'status': 'DRAFT', 'input': {'path': str(source), 'sha256': preview['sha256']},
             'request': {'mode': 'ab_manual', 'edsEnabled': False, 'restoreStartPosition': True,
                         'filters': {'minLengthUm': args.min_length}}}
    write_json_atomic(task_dir / 'draft.json', draft)
    result = server.save_manual_ab_assignment(task_id, {'coordinateUnit': 'mm',
                         'a': {'x': -20.1, 'y': -18.89}, 'b': {'x': 20.38, 'y': 20.06}})
    config = server._make_config(source, task_dir, {**result['request'], 'mode': 'ab_points'})
    # Broaden the validation evidence to eight spatially distributed particles.
    settings = json.loads(config.settings_path.read_text(encoding='utf-8'))
    settings['abPoint']['semRefine']['validationCount'] = 8
    write_json_atomic(config.settings_path, settings)
    print(json.dumps({'prepared': True, 'taskDirectory': str(task_dir),
                     'manualPoints': result['calibration']['abPointResult']['manualStagePoints']}, ensure_ascii=False), flush=True)
    if args.execute:
        logger.remove()
        logger.add(str(out / 'device.log'), encoding='utf-8')
        result = refine_ab_point_calibration(task_dir, config=config,
            progress=lambda i, n, msg: print(f'[{i}/{n}] {msg}', flush=True),
            should_stop=lambda: (out / 'STOP').exists())
        print(json.dumps({'status': result['status'], 'refinement': result['calibration']['abPointResult']['semRefinement'],
                          'cleanup': result['cleanup']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
