"""Explicit final gun shutdown with verified readback; no stage/magnification move."""
import argparse
from datetime import datetime, timezone
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.cleanliness_core.atomic_json import write_json_atomic
from backend.cleanliness_core.optical_particle_task import ZemMicroscope
from backend.cleanliness_core.microscope_safety import hv_status_name
from loguru import logger


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='192.168.0.15')
    parser.add_argument('--port', type=int, default=61557)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Requires explicit --execute after all acquisition has finished')
    output = args.output.resolve()
    if output.exists():
        parser.error('Do not overwrite an existing shutdown record')
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(str(output.with_suffix('.log')), encoding='utf-8')
    result = {'host': args.host, 'port': args.port, 'verifiedClosed': False,
              'requestedAt': datetime.now(timezone.utc).isoformat(),
              'stageMoveRequested': False, 'magnificationChangeRequested': False}
    scope = ZemMicroscope(args.host, args.port, {})
    try:
        result['before'] = scope.connect()
        result['hvBefore'] = scope.get_hv_status()
        # Product safety helper requires three consecutive closed reads.
        result['shutdownResult'] = scope.gun_off()
        result['hvAfter'] = scope.get_hv_status()
        result['verifiedClosed'] = hv_status_name(result['hvAfter']) == 'close'
        result['positionAfterUm'] = list(scope.get_position())
        if not result['verifiedClosed']:
            raise RuntimeError('Gun closure not confirmed')
    except Exception as exc:
        result['error'] = str(exc)
        raise
    finally:
        result['completedAt'] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(output, result)
        print('verifiedClosed=' + str(result['verifiedClosed']), flush=True)


if __name__ == '__main__':
    main()
