## copied jacobi 1d and simplified the kernel

import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition
from dace.transformation.auto.auto_optimize import auto_optimize
from dace.config import set_temporary

# Dataset sizes
# TSTEPS, N
sizes = {"mini": (30,30), "small": (120, 120), "medium": (400, 400), "large": (2000, 2000), "extra-large": (4000, 4000)}
N = dc.symbol('N', dtype=dc.int64)


@dc.program
def local_mul_kernel(A: dc.int64[N,N], B: dc.int64[N,N]):
    C = np.empty(A.shape, dtype=A.dtye)
    const = 6
    C = A * B
    B = const * C


def initialize(N, datatype=np.int64):
    A = np.fromfunction(lambda i, j: (i + j + 372036854775807), (N, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: (i - j + 3372036854775807), (N, N), dtype=datatype)
    return A, B


def ground_truth(A, B):
    B = 6 * (A*B)


def run_local_mul(device_type: dace.dtypes.DeviceType):
    '''
    Runs simple add for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    N = sizes["small"]
    A, B = initialize(N)
    A_ref = np.copy(A)
    B_ref = np.copy(B)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = local_mul_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(A, B, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = local_mul_kernel.to_sdfg(simplify=True)
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Dot
        Dot.default_implementation = "FPGA_PartialSums"
        sdfg.expand_library_nodes()
        sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N))
        sdfg(A=A, B=B)

    # Compute ground truth and validate
    ground_truth(A_ref, B_ref)
    assert np.allclose(B, B_ref)
    return sdfg


def test_cpu():
    run_local_mul(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_local_mul(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_local_mul(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_local_mul(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_local_mul(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_local_mul(dace.dtypes.DeviceType.FPGA)