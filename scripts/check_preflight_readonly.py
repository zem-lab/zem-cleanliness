"""Run the same preflight on a saved task; never start acquisition or modify it."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.cleanliness_core.cleanliness_detection import load_cleanliness_config
from backend.cleanliness_core.cleanliness_task import RealSpectrumAnalyzer
from backend.cleanliness_core.preflight import run_preflight


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();task=args.task.resolve();out=args.output.resolve()
    if out.exists() or out.is_relative_to(task):p.error('Use a new output outside the source task')
    original=(task/'task.json').read_bytes()
    m=json.loads(original.decode('utf-8-sig'))
    settings_path=task/(m.get('settingsSnapshot') or '.cleanliness-settings.json')
    settings=load_cleanliness_config(settings_path);device=settings['device']
    config=SimpleNamespace(settings_path=settings_path,output_dir=task,
        experience_calibration_path=Path(settings['experienceCalibration']),
        microscope_host=device['microscopeHost'],microscope_port=device['microscopePort'],
        eds_enabled=settings['task'].get('edsEnabled',True))
    def eds():
        analyzer=RealSpectrumAnalyzer(device['edsHost'],device['edsPort'],config.microscope_host,config.microscope_port,timeout=3)
        try:return analyzer.check_ready()
        finally:analyzer.close()
    seg=settings['calibration'].get('candidateSegmentation') or {}
    models=[Path(seg['modelPath'])] if seg.get('backend')=='yolo' else []
    result=run_preflight(config,m,'analysis',model_paths=models,eds_check=eds)
    assert (task/'task.json').read_bytes()==original
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
