#!/usr/bin/env python3
# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import click
import os
from pathlib import Path
import re
import subprocess as sp
import sys
from typing import Any, Iterable, Union

TEST_TIMEOUT = 600  # Seconds

from fpga_testing import (Colors, DACE_DIR, TEST_DIR, cli, dump_logs,
                          print_status, print_success, print_error)

# (relative path, sdfg name(s), run synthesis, assert II=1, args to executable)
TESTS = [
    # RTL cores
    ("tests/rtl/hardware_test.py", "floating_point_vector_plus_scalar", True,
     False, [1]),
]


def run(path: Path, sdfg_names: Union[str, Iterable[str]], run_synthesis: bool,
        assert_ii_1: bool, args: Iterable[Any]):

    # Set environment variables
    env = os.environ.copy()
    env["DACE_compiler_fpga_vendor"] = "xilinx"
    env["DACE_compiler_use_cache"] = "0"
    # We would like to use DACE_cache=hash, but we need to know which folder to
    # run synthesis in.
    env["DACE_cache"] = "name"
    env["DACE_compiler_xilinx_mode"] = "simulation"
    os.environ["DACE_optimizer_transform_on_call"] = "0"
    os.environ["DACE_optimizer_interface"] = ""
    os.environ["DACE_optimizer_autooptimize"] = "0"

    path = DACE_DIR / path
    if not path.exists():
        print_error(f"Path {path} does not exist.")
        return False
    base_name = f"{Colors.UNDERLINE}{path.stem}{Colors.END}"

    # Simulation in software
    print_status(f"{base_name}: Running simulation.")
    if "rtl" in path.parts:
        env["DACE_compiler_xilinx_mode"] = "hardware_emulation"
        if "LIBRARY_PATH" not in env:
            env["LIBRARY_PATH"] = ""
        env["LIBRARY_PATH"] += ":/usr/lib/x86_64-linux-gnu"
    try:
        proc = sp.Popen(map(str, [sys.executable, path] + args),
                        env=env,
                        cwd=TEST_DIR,
                        stdout=sp.PIPE,
                        stderr=sp.PIPE,
                        encoding="utf-8")
        sim_out, sim_err = proc.communicate(timeout=TEST_TIMEOUT)
    except sp.TimeoutExpired:
        dump_logs(proc)
        print_error(f"{base_name}: Simulation timed out "
                    f"after {TEST_TIMEOUT} seconds.")
        return False
    if proc.returncode != 0:
        dump_logs((sim_out, sim_err))
        print_error(f"{base_name}: Simulation failed.")
        return False
    print_success(f"{base_name}: Simulation successful.")

    if isinstance(sdfg_names, str):
        sdfg_names = [sdfg_names]
    for sdfg_name in sdfg_names:
        build_folder = TEST_DIR / ".dacecache" / sdfg_name / "build"
        if not build_folder.exists():
            print_error(f"Invalid SDFG name {sdfg_name} for {base_name}.")
            return False
        open(build_folder / "simulation.out", "w").write(sim_out)
        open(build_folder / "simulation.err", "w").write(sim_err)

        # High-level synthesis
        if run_synthesis:
            print_status(
                f"{base_name}: Running high-level synthesis for {sdfg_name}.")
            try:
                proc = sp.Popen(["make", "xilinx_synthesis"],
                                env=env,
                                cwd=build_folder,
                                stdout=sp.PIPE,
                                stderr=sp.PIPE,
                                encoding="utf=8")
                syn_out, syn_err = proc.communicate(timeout=TEST_TIMEOUT)
            except sp.TimeoutExpired:
                dump_logs(proc)
                print_error(f"{base_name}: High-level synthesis timed out "
                            f"after {TEST_TIMEOUT} seconds.")
                return False
            if proc.returncode != 0:
                dump_logs(proc)
                print_error(f"{base_name}: High-level synthesis failed.")
                return False
            print_success(f"{base_name}: High-level synthesis "
                          f"successful for {sdfg_name}.")
            open(build_folder / "synthesis.out", "w").write(syn_out)
            open(build_folder / "synthesis.err", "w").write(syn_err)

            # Check if loops were pipelined with II=1
            if assert_ii_1:
                loops_found = False
                for f in build_folder.iterdir():
                    if "hls.log" in f.name:
                        hls_log = f
                        break
                else:
                    print_error(f"{base_name}: HLS log file not found.")
                    return False
                hls_log = open(hls_log, "r").read()
                for m in re.finditer(r"Final II = ([0-9]+)", hls_log):
                    loops_found = True
                    if int(m.group(1)) != 1:
                        dump_logs((syn_out, syn_err))
                        print_error(f"{base_name}: Failed to achieve II=1.")
                        return False
                if not loops_found:
                    dump_logs((syn_out, syn_err))
                    print_error(f"{base_name}: No pipelined loops found.")
                    return False
                print_success(f"{base_name}: II=1 achieved.")

    return True


@click.command()
@click.option("--parallel/--no-parallel", default=True)
@click.argument("tests", nargs=-1)
def xilinx_cli(parallel, tests):
    """
    If no arguments are specified, runs all tests. If any arguments are
    specified, runs only the tests specified (matching on file name or SDFG
    name).
    """
    cli(TESTS, run, tests, parallel)


if __name__ == "__main__":
    xilinx_cli()
