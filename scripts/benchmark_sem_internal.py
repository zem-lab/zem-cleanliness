"""Offline real-frame equivalence and timing benchmark; no network/hardware."""
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.cleanliness_core.sem_segmentation import decode_detections,sam3_config

def main():
    folder=ROOT/'backend/tests/fixtures/sem_performance_20260929'
    spec=importlib.util.spec_from_file_location('reference',folder/'reference_decoder.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    detections=json.loads((folder/'response.json').read_text())['detections']
    config=sam3_config({'scoreThreshold':.01})
    timings={}
    results={}
    for label,fn in [('before',reference.decode_detections),('after',decode_detections)]:
        start=time.perf_counter();results[label]=fn(detections,(1024,1024),config)
        timings[label]=time.perf_counter()-start
        print(label,timings[label],flush=True)
    assert results['before']==results['after'], 'Optimization changed particle contours or duplicate decisions'
    expected=folder/'expected-shapes.json'
    if expected.exists():
        assert json.loads(expected.read_text())==results['before'], 'Existing baseline must never be auto-overwritten'
    else:
        expected.write_text(json.dumps(results['before']),encoding='utf-8')
    report={'detections':len(detections),'unique':len(results['after']),'exactEquality':True,
            'seconds':timings,'speedup':timings['before']/timings['after']}
    out=ROOT/'artifacts/sem-internal-performance';out.mkdir(exist_ok=True)
    (out/'segmentation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
