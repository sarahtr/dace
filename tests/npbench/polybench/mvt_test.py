# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test, xilinx_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt, greedy_fuse
from dace.transformation.dataflow import MapCollapse, TrivialMapElimination, MapFusion, ReduceExpansion
from dace.transformation.interstate import LoopToMap, RefineNestedAccess
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.subgraph import helpers as xfsh
from dace.transformation import helpers as xfh
from dace.config import set_temporary
from dace.sdfg.utils import is_fpga_kernel

# Data set sizes
# N
sizes = {"mini": 40, "small": 120, "medium": 400, "large": 2000, "extra-large": 4000}

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def mvt_kernel(x1: dc.float32[N], x2: dc.float32[N], y_1: dc.float32[N], y_2: dc.float32[N], A: dc.float32[N, N]):

    x1 += A @ y_1
    x2 += y_2 @ A


def initialize(N, datatype=np.float32):
    x1 = np.fromfunction(lambda i: (i % N) / N, (N, ), dtype=datatype)
    x2 = np.fromfunction(lambda i: ((i + 1) % N) / N, (N, ), dtype=datatype)
    y_1 = np.fromfunction(lambda i: ((i + 3) % N) / N, (N, ), dtype=datatype)
    y_2 = np.fromfunction(lambda i: ((i + 4) % N) / N, (N, ), dtype=datatype)
    A = np.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=datatype)

    return x1, x2, y_1, y_2, A


def run_mvt(device_type: dace.dtypes.DeviceType):
    '''
    Runs MVT for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    N = sizes["large"]
    x1, x2, y_1, y_2, A = initialize(N)
    x1_ref = np.copy(x1)
    x2_ref = np.copy(x2)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = mvt_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(x1, x2, y_1, y_2, A, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = mvt_kernel.to_sdfg(simplify=True)

        
        '''
        General transformations from auto_optimize:
        '''

        # Simplification and loop parallelization
        transformed = True
        sdfg.apply_transformations_repeated(TrivialMapElimination, validate=True, validate_all=False)
        while transformed:
            sdfg.simplify(validate=False, validate_all=False)
            for s in sdfg.sdfg_list:
                xfh.split_interstate_edges(s)
            l2ms = sdfg.apply_transformations_repeated((LoopToMap, RefineNestedAccess),
                                                    validate=False,
                                                    validate_all=False)
            transformed = l2ms > 0

        # Collapse maps and eliminate trivial dimensions
        sdfg.simplify()
        sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=False)

        # fuse subgraphs greedily
        sdfg.simplify()

        greedy_fuse(sdfg, device=device_type, validate_all=False)

        # fuse stencils greedily
        greedy_fuse(sdfg, device=device_type, validate_all=False, recursive=False, stencil=True)

        # Move Loops inside Maps when possible
        from dace.transformation.interstate import MoveLoopIntoMap
        sdfg.apply_transformations_repeated([MoveLoopIntoMap])

        '''------------'''


        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemv
        Gemv.default_implementation = "FPGA_Accumulate"
        sdfg.expand_library_nodes()
        #sdfg.apply_transformations_repeated([InlineSDFG], print_report=True)
        sdfg.specialize(dict(N=N))
        
        for s in sdfg.states():
            if is_fpga_kernel(sdfg, s):
                s.instrument = dace.InstrumentationType.FPGA
                break
            
        sdfg(x1, x2, y_1, y_2, A)

    # Compute ground truth and validate
    mvt_kernel.f(x1_ref, x2_ref, y_1, y_2, A)
    assert np.allclose(x1, x1_ref)
    assert np.allclose(x2, x2_ref)
    return sdfg


def test_cpu():
    run_mvt(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_mvt(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_mvt(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_mvt(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_mvt(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_mvt(dace.dtypes.DeviceType.FPGA)
