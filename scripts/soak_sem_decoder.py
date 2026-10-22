"""Repeat real dense-frame decoding and gating, checking exact outputs/resources."""
import gc
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.cleanliness_core.sem_segmentation import decode_detections,sam3_config
from backend.cleanliness_core.sem_direct import evaluate_shapes
from scripts.stress_sem_internal import resources

def main():
    folder=ROOT/'backend/tests/fixtures/sem_performance_20260929'
    detections=json.loads((folder/'response.json').read_text())['detections']
    expected=json.loads((folder/'expected-shapes.json').read_text())
    expected_gates=json.loads((folder/'expected-gates.json').read_text(encoding='utf-8'))
    case=json.loads((folder/'gate-case.json').read_text())
    config=sam3_config({'scoreThreshold':.01});samples=[];started=time.perf_counter()
    for i in range(200):
        shapes=decode_detections(detections,(1024,1024),config)
        assert shapes==expected
        assert evaluate_shapes(shapes,**case)==expected_gates
        if (i+1)%20==0:
            gc.collect();row={'iterations':i+1,'seconds':time.perf_counter()-started,**resources()}
            samples.append(row);print(json.dumps(row),flush=True)
    assert samples[-1]['handles']<=samples[0]['handles']+4
    assert samples[-1]['workingSetMB']<=samples[0]['workingSetMB']+32
    out=ROOT/'artifacts/sem-internal-performance';out.mkdir(exist_ok=True)
    summary={'iterations':200,'decodedMasks':200*len(detections),'allContoursAndGatesExact':True,'samples':samples}
    (out/'decoder-soak.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
