"""Regenerate checked-in EDS V2 bindings with grpcio-tools==1.76.0.

Development-only prerequisite: python -m pip install --no-deps grpcio-tools==1.76.0
Runtime requirements remain grpcio==1.76.0 and protobuf==6.31.1.
"""
from importlib.metadata import version
from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'backend/cleanliness_core/eds/protos'
OUTPUT = ROOT/'backend/cleanliness_core/eds/generated'

if __name__ == '__main__':
    if version('grpcio-tools') != '1.76.0':
        raise RuntimeError('Use grpcio-tools==1.76.0 to match pinned runtime dependencies')
    names = ['session', 'jobs_common', 'job_events', 'jobs']
    result = protoc.main(['grpc_tools.protoc', f'-I{SOURCE}', f'--python_out={OUTPUT}',
                          f'--grpc_python_out={OUTPUT}', *[str(SOURCE/'v2'/f'{n}.proto') for n in names]])
    if result:
        raise SystemExit(result)
    for name in names:
        for suffix in ['_pb2.py', '_pb2_grpc.py']:
            path = OUTPUT/'v2'/(name+suffix)
            content = path.read_text(encoding='utf-8')
            content = content.replace('from v2 import ', 'from backend.cleanliness_core.eds.generated.v2 import ')
            path.write_text(content, encoding='utf-8', newline='\n')
