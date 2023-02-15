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
from dace.transformation.dataflow import MapCollapse, TrivialMapElimination, MapFusion, ReduceExpansion, PruneConnectors,Vectorization
from dace.transformation.interstate import LoopToMap, RefineNestedAccess
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.subgraph import helpers as xfsh
from dace.transformation import helpers as xfh
from dace.config import set_temporary
from dace.sdfg.utils import is_fpga_kernel

# Dataset sizes
# TSTEPS, N
sizes = {"mini": (20, 30), "small": (40, 120), "medium": (100, 400), "large": (500, 2000), "extra-large": (1000, 4000)}
N = dc.symbol('N', dtype=dc.int64)


@dc.program
def jacobi_1d_kernel(TSTEPS: dc.int32, A: dc.float32[N], B: dc.float32[N]):

    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


def initialize(N, datatype=np.float32):
    A = np.fromfunction(lambda i: (i + 2) / N, (N, ), dtype=datatype)
    B = np.fromfunction(lambda i: (i + 3) / N, (N, ), dtype=datatype)

    return A, B


def ground_truth(TSTEPS, A, B):

    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


def run_jacobi_1d(device_type: dace.dtypes.DeviceType):
    '''
    Runs Jacobi 1d for the given device
    :return: the SDFG
    '''

    # Initialize data (polybench small size)
    TSTEPS, N = sizes["large"]
    A, B = initialize(N)
    A_ref = np.copy(A)
    B_ref = np.copy(B)

    if device_type in {dace.dtypes.DeviceType.CPU, dace.dtypes.DeviceType.GPU}:
        # Parse the SDFG and apply auto-opt
        sdfg = jacobi_1d_kernel.to_sdfg()
        sdfg = auto_optimize(sdfg, device_type)
        sdfg(TSTEPS, A, B, N=N)
    elif device_type == dace.dtypes.DeviceType.FPGA:
        # Parse SDFG and apply FPGA friendly optimization
        sdfg = jacobi_1d_kernel.to_sdfg(simplify=True)

       
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
        
        lm_applied = sdfg.apply_transformations_repeated((LoopToMap, RefineNestedAccess),
                                                    validate=False,
                                                    validate_all=False)
        print("Applied LoopToMap & RefineNestedAccess: " + str(lm_applied))

        sc_applied = sdfg.apply_transformations_repeated([StreamingComposition])
        print("Applied StreamingComposition: " + str(sc_applied))

        il_applied = sdfg.apply_transformations_repeated([InlineSDFG])
        print("Applied InlineSDFG: " + str(il_applied))



        simplify = sdfg.simplify()
        print("Applied simplifications 1: " + str(simplify))

        mf_applied = sdfg.apply_transformations_repeated([MapFusion], print_report=True)
        print("Applied MapFusion: " + str(mf_applied))

        simplify = sdfg.simplify()
        print("Applied simplifications 2: " + str(simplify))

        ##########################
        #FPGA Auto Opt
        fpga_auto_opt.fpga_global_to_local(sdfg)
        il = fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg, num_banks=2)
        print("applied interleave: " + str(il))

        ##########################
        #FPGA Auto Opt
        fpga_auto_opt.fpga_global_to_local(sdfg)
        il = fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg, num_banks=2)
        print("applied interleave: " + str(il))

        #sdfg.states()[0].location["is_FPGA_kernel"] = False
        sdfg.specialize(dict(N=N))
        #sdfg.states()[0].nodes()[0].sdfg.specialize(dict(N=N))
        
        for s in sdfg.states():
            if is_fpga_kernel(sdfg, s):
                s.instrument = dace.InstrumentationType.FPGA
                break
        
        

        sdfg(TSTEPS=TSTEPS, A=A, B=B)

    # Compute ground truth and validate
    ground_truth(TSTEPS, A_ref, B_ref)
    assert np.allclose(A, A_ref)
    return sdfg


def test_cpu():
    run_jacobi_1d(dace.dtypes.DeviceType.CPU)


@pytest.mark.gpu
def test_gpu():
    run_jacobi_1d(dace.dtypes.DeviceType.GPU)


@fpga_test(assert_ii_1=False)
def test_fpga():
    return run_jacobi_1d(dace.dtypes.DeviceType.FPGA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", default='cpu', choices=['cpu', 'gpu', 'fpga'], help='Target platform')

    args = vars(parser.parse_args())
    target = args["target"]

    if target == "cpu":
        run_jacobi_1d(dace.dtypes.DeviceType.CPU)
    elif target == "gpu":
        run_jacobi_1d(dace.dtypes.DeviceType.GPU)
    elif target == "fpga":
        run_jacobi_1d(dace.dtypes.DeviceType.FPGA)
