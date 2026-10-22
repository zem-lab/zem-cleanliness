"""Exercise lazy microscope-driver dependencies without opening a connection."""
import sys


def check_runtime():
    def forbid_network(event, args):
        if event in {"socket.connect", "socket.bind", "socket.sendto"}:
            raise RuntimeError("Runtime dependency check must not contact hardware")
    sys.addaudithook(forbid_network)
    import numpy as np
    import scipy
    import scipy.optimize
    from scipy._external.array_api_compat.numpy import fft
    from jgb.util import cv_util
    from zemrpc.zem.conn.protocol import append_crc
    from zemrpc.zem.models import SnapArgs
    from backend.cleanliness_core.eds.generated.v2 import jobs_pb2
    from backend.cleanliness_core.preflight import ZemPreflightReader
    from backend.sem_mcp import create_server
    from backend.sem_tools import TOOLS
    from mcp.types import Tool
    from dataclasses import replace
    from backend.cleanliness_core.large_particle_center import plan_large_particle_center
    from backend.cleanliness_core.optical_particle_analysis import OpticalParticle
    if 'SetStagePos' in ZemPreflightReader.READS or 'SetHV_ON' in ZemPreflightReader.READS:
        raise RuntimeError('Preflight must not expose actuator commands')
    mcp_server = create_server(None)
    from jsonschema import Draft202012Validator
    from backend.sem_rpc_catalog import READS, WRITES, UNAVAILABLE
    for tool in TOOLS:
        Draft202012Validator.check_schema(tool['inputSchema'])
    from backend.sem_rpc_full import SPECS
    if (mcp_server.name != 'zem-sem' or len([Tool(**tool) for tool in TOOLS]) != 16 or len(SPECS) != 115
            or not {'sem_rpc_call', 'sem_rpc_read', 'sem_rpc_execute', 'sem_get_snapshot', 'sem_wait_job', 'sem_list_rpc'} <= {t['name'] for t in TOOLS}
            or set(READS) & (set(WRITES) | set(UNAVAILABLE))):
        raise RuntimeError('MCP tool contract self-check failed')
    result = scipy.optimize.least_squares(lambda x: x - 2., np.array([0.]))
    if not np.allclose(result.x, [2.]):
        raise RuntimeError("SciPy optimizer self-check failed")
    if not np.allclose(fft.fft(np.array([1., 0.])), [1., 1.]):
        raise RuntimeError("SciPy FFT self-check failed")
    if not append_crc("$GetHVStatus:,127.0.0.1:1,"):
        raise RuntimeError("Microscope protocol serialization failed")
    particle = OpticalParticle(1, "runtime-check", "runtime-check", 0., 0.,
                               160., 40., 6400., 90., "", 0, 0)
    image = np.zeros((64, 64), dtype=np.uint8)
    image[8:56, 8:56] = 200
    center = plan_large_particle_center(image, particle, 2., field_width_um=128.)
    if (center["status"] != "READY" or center.get("pointPx") != [32, 32]
            or center["identityVerified"] is not False or not center["contourMatchingSkipped"]):
        raise RuntimeError("Large-particle center sampling self-check failed")
    background = plan_large_particle_center(np.zeros_like(image), particle, 2., field_width_um=128.)
    ordinary = plan_large_particle_center(image, replace(particle, length_um=60.), 2., field_width_um=128.)
    if background["status"] != "REJECTED" or "pointPx" in background or ordinary["status"] != "NOT_APPLICABLE":
        raise RuntimeError("Large-particle center sampling refusal/routing self-check failed")
    return {"runtimeCheck":"passed", "scipy":scipy.__version__, "driverModels":SnapArgs.__name__,
            "driverImageModule":cv_util.__name__, "edsSchema":jobs_pb2.__name__,
            "largeParticleCenterCheck":"passed", "mcpCheck":"passed", "networkActions":0}
