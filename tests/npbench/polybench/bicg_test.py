# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
# Original application code: NPBench - https://github.com/spcl/npbench
import dace.dtypes
import numpy as np
import dace as dc
import pytest
import argparse
from dace.fpga_testing import fpga_test
from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG
from dace.transformation.dataflow import StreamingMemory, StreamingComposition
from dace.transformation.auto.auto_optimize import auto_optimize, fpga_auto_opt, greedy_fuse
from dace.transformation.dataflow import MapCollapse, TrivialMapElimination, MapFusion, ReduceExpansion, PruneConnectors, Vectorization
from dace.transformation.interstate import LoopToMap, RefineNestedAccess
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.subgraph import helpers as xfsh
from dace.transformation import helpers as xfh
from dace.sdfg.utils import is_fpga_kernel

# Data set sizes
# M, N
sizes = {
    "mini": (38, 42),
    "small": (116, 124),
    "medium": (390, 410),
    "large": (1900, 2100),
    "extra-large": (1800, 2200)
}

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


def initialize(M, N, datatype=np.float32):
    A = np.fromfunction(lambda i, j: (i * (j + 1) % N) / N, (N, M), dtype=datatype)
    p = np.fromfunction(lambda i: (i % M) / M, (M, ), dtype=datatype)
    r = np.fromfunction(lambda i: (i % N) / N, (N, ), dtype=datatype)

    return A, p, r


@dc.program
def bicg_kernel(A: dc.float32[N, M], p: dc.float32[M], r: dc.float32[N]):
    return r @ A, A @ p


def run_bicg(device_type: dace.dtypes.DeviceType):
    '''
    Runs BiCG for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    M, N = sizes["large"]
    A, p, r = initialize(M, N)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = bicg_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        s, q = sdfg(A, p, r, M=M, N=N)

    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        # Note: currently the kernel uses double-precision floating point numbers
        sdfg = bicg_kernel.to_sdfg(simplify=True)

        
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
        s = sdfg.simplify()
        print("applied simplifications 1: " + str(s))
        mc = sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=False)
        print("applied MapCollapse: " + str(mc))

        # fuse subgraphs greedily
        sp = sdfg.simplify()
        print("applied simplifications: " + str(sp))

        greedy_fuse(sdfg, device=device_type, validate_all=False)

        # fuse stencils greedily
        greedy_fuse(sdfg, device=device_type, validate_all=False, recursive=False, stencil=True)

        # Move Loops inside Maps when possible
        from dace.transformation.interstate import MoveLoopIntoMap
        mlim = sdfg.apply_transformations_repeated([MoveLoopIntoMap])
        print("applied: MoveLoopIntoMap: " + str(mlim))

        '''------------'''
        applied = sdfg.apply_transformations([FPGATransformSDFG])
        assert applied == 1

        # Use FPGA Expansion for lib nodes, and expand them to enable further optimizations
        from dace.libraries.blas import Gemv
        Gemv.default_implementation = "FPGA_Accumulate"
        sdfg.expand_library_nodes()
        
        # lm_applied = sdfg.apply_transformations_repeated((LoopToMap, RefineNestedAccess),
        #                                             validate=False,
        #                                             validate_all=False)
        # print("Applied LoopToMap & RefineNestedAccess: " + str(lm_applied))

        # sc_applied = sdfg.apply_transformations_repeated([StreamingComposition])
        # print("Applied StreamingComposition: " + str(sc_applied))

        # il_applied = sdfg.apply_transformations_repeated([InlineSDFG])
        # print("Applied InlineSDFG: " + str(il_applied))



        # simplify = sdfg.simplify()
        # print("Applied simplifications 1: " + str(simplify))

        # mf_applied = sdfg.apply_transformations_repeated([MapFusion], print_report=True)
        # print("Applied MapFusion: " + str(mf_applied))

        # simplify = sdfg.simplify()
        # print("Applied simplifications 2: " + str(simplify))

        # ##########################
        # #FPGA Auto Opt
        # fpga_auto_opt.fpga_global_to_local(sdfg)
        # il = fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg, num_banks=2)
        # print("applied interleave: " + str(il))

        # specialize the SDFG (needed by the GEMV expansion)
        sdfg.specialize(dict(M=M, N=N))

        for s in sdfg.states():
            if is_fpga_kernel(sdfg, s):
                s.instrument = dace.InstrumentationType.FPGA
                break
        s, q = sdfg(A=A, p=p, r=r)

    # Compute ground truth and Validate result
    s_ref, q_ref = bicg_kernel.f(A, p, r)
    assert np.allclose(s, s_ref)
    assert np.allclose(q, q_ref)
    return sdfg


def test_cpu():
    run_bicg(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_bicg(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_bicg(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_bicg(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_bicg(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_bicg(dace.dtypes.DeviceType.FPGA)
