#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

import os
import itertools

from typing import List

import dace
from dace import registry, symbolic, nodes, StorageType
from dace.config import Config
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.dispatcher import TargetDispatcher
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import StateSubgraphView
from dace.sdfg import find_input_arraynode, find_output_arraynode
from dace.codegen.codeobject import CodeObject


@registry.autoregister_params(name='rtl')
class RTLCodeGen(TargetCodeGenerator):
    """ RTL Code Generator (SystemVerilog) """

    title = 'RTL'
    target_name = 'rtl'
    language = dace.Language.RTL

    def __init__(self,
                 frame_codegen: DaCeCodeGenerator,
                 sdfg: dace.SDFG):
        # store reference to sdfg
        self.sdfg: dace.SDFG = sdfg
        # store reference to frame code generator
        self.frame: DaCeCodeGenerator = frame_codegen
        # get dispatcher to register callbacks for allocation/nodes/.. code generators
        self.dispatcher: TargetDispatcher = frame_codegen.dispatcher
        # register node dispatcher -> generate_node(), predicate: process tasklets only
        self.dispatcher.register_node_dispatcher(self,
                                                 lambda sdfg, node: isinstance(node, nodes.Tasklet))
        # register all cpu type copies
        cpu_storage_types = [StorageType.CPU_Pinned, StorageType.CPU_Heap, StorageType.CPU_ThreadLocal,
                             StorageType.Register, StorageType.Default]
        for src_storage, dst_storage in itertools.product(cpu_storage_types, cpu_storage_types):
            self.dispatcher.register_copy_dispatcher(src_storage, dst_storage, None, self)
        # local variables
        self.code_objects: List[CodeObject] = list()

    def generate_node(self,
                      sdfg: dace.SDFG,
                      dfg: StateSubgraphView,
                      state_id: int,
                      node: nodes.Node,
                      function_stream: CodeIOStream,
                      callsite_stream: CodeIOStream):
        # check instance type
        if isinstance(node, dace.nodes.Tasklet):
            """ 
            handle Tasklet: 
                (1) generate in->tasklet
                (2) generate tasklet->out 
                (3) generate tasklet 
            """
            # generate code to handle data input to the tasklet
            for edge in dfg.in_edges(node):
                # find input array
                src_node = find_input_arraynode(dfg, edge)
                # dispatch code gen (copy_memory)
                self.dispatcher.dispatch_copy(src_node, node, edge, sdfg, dfg, state_id, function_stream,
                                              callsite_stream)
            # generate code to handle data output from the tasklet
            for edge in dfg.out_edges(node):
                # find output array
                dst_node = find_output_arraynode(dfg, edge)
                # dispatch code gen (define_out_memlet)
                self.dispatcher.dispatch_output_definition(node, dst_node, edge, sdfg, dfg, state_id, function_stream,
                                                           callsite_stream)
            # generate tasklet code
            self.unparse_tasklet(sdfg, dfg, state_id, node, function_stream, callsite_stream)
        else:
            raise RuntimeError(
                "Only tasklets are handled here, not {}. This should have been filtered by the predicate".format(
                    type(node)))

    def copy_memory(self,
                    sdfg: dace.SDFG,
                    dfg: StateSubgraphView,
                    state_id: int,
                    src_node: nodes.Node,
                    dst_node: nodes.Node,
                    edge: MultiConnectorEdge,
                    function_stream: CodeIOStream,
                    callsite_stream: CodeIOStream):
        """
            Generate input/output memory copies from the array references to local variables (i.e. for the tasklet code).
        """
        if isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.Tasklet):  # handle AccessNode->Tasklet
            if isinstance(dst_node.in_connectors[edge.dst_conn], dace.pointer):  # pointer accessor
                line: str = "{} {} = &{}[0];".format(dst_node.in_connectors[edge.dst_conn].ctype, edge.dst_conn,
                                                     edge.src.data)
            elif isinstance(dst_node.in_connectors[edge.dst_conn], dace.vector):  # vector accessor
                line: str = "{} {} = *({} *)(&{}[0]);".format(dst_node.in_connectors[edge.dst_conn].ctype,
                                                              edge.dst_conn,
                                                              dst_node.in_connectors[edge.dst_conn].ctype,
                                                              edge.src.data)
            else:  # scalar accessor
                line: str = "{}* {} = &{}[0];".format(dst_node.in_connectors[edge.dst_conn].ctype, edge.dst_conn,
                                                      edge.src.data)
        else:
            raise RuntimeError("Not handling copy_memory case of type {} -> {}.".format(type(edge.src), type(edge.dst)))
        # write accessor to file
        callsite_stream.write(line)

    def define_out_memlet(self,
                          sdfg: dace.SDFG,
                          dfg: StateSubgraphView,
                          state_id: int,
                          src_node: nodes.Node,
                          dst_node: nodes.Node,
                          edge: MultiConnectorEdge,
                          function_stream: CodeIOStream,
                          callsite_stream: CodeIOStream):
        """
            Generate output copy code (handled within the rtl tasklet code).
        """
        if isinstance(edge.src, nodes.Tasklet) and isinstance(edge.dst, nodes.AccessNode):
            if isinstance(src_node.out_connectors[edge.src_conn], dace.pointer):  # pointer accessor
                line: str = "{} {} = &{}[0];".format(src_node.out_connectors[edge.src_conn].ctype, edge.src_conn,
                                                     edge.dst.data)
            elif isinstance(src_node.out_connectors[edge.src_conn], dace.vector):  # vector accessor
                line: str = "{} {} = *({} *)(&{}[0]);".format(src_node.out_connectors[edge.src_conn].ctype,
                                                              edge.src_conn,
                                                              src_node.out_connectors[edge.src_conn].ctype,
                                                              edge.dst.data)
            else:  # scalar accessor
                line: str = "{}* {} = &{}[0];".format(src_node.out_connectors[edge.src_conn].ctype, edge.src_conn,
                                                      edge.dst.data)
        else:
            raise RuntimeError("Not handling define_out_memlet case of type {} -> {}.".format(type(edge.src), type(edge.dst)))
        # write accessor to file
        callsite_stream.write(line)

    def get_generated_codeobjects(self):
        """
            Return list of code objects (that are later generating code files).
        """
        return self.code_objects

    @property
    def has_initializer(self):
        """
            Disable initializer method generation.
        """
        return False

    @property
    def has_finalizer(self):
        """
            Disable exit/finalizer method generation.
        """
        return False

    @staticmethod
    def cmake_options():
        """
            Process variables to be exposed to the CMakeList.txt script.
        """
        # get flags from config
        verbose = Config.get_bool("compiler", "rtl", "verbose")
        verilator_flags = Config.get("compiler", "rtl", "verilator_flags")
        verilator_lint_warnings = Config.get_bool("compiler", "rtl", "verilator_lint_warnings")
        # create options list
        options = [
            "-DDACE_RTL_VERBOSE=\"{}\"".format(verbose),
            "-DDACE_RTL_VERILATOR_FLAGS=\"{}\"".format(verilator_flags),
            "-DDACE_RTL_VERILATOR_LINT_WARNINGS=\"{}\"".format(verilator_lint_warnings)
        ]
        return options

    def generate_rtl_parameters(self, constants):
        pass

    def unparse_tasklet(self,
                        sdfg: dace.SDFG,
                        dfg: StateSubgraphView,
                        state_id: int,
                        node: nodes.Node,
                        function_stream: CodeIOStream,
                        callsite_stream: CodeIOStream):

        # extract data
        state = sdfg.nodes()[state_id]
        tasklet = node

        # construct paths
        unique_name = "top_{}_{}_{}".format(sdfg.sdfg_id, sdfg.node_id(state), state.node_id(tasklet))
        base_path = os.path.join(sdfg.build_folder, "src", "rtl")
        absolut_path = os.path.abspath(base_path)

        # construct parameters module header
        if len(sdfg.constants) == 0:
            parameter_string = ""
        else:
            parameter_string = """\
        #(
        {}
        )""".format(" " + "\n".join(
                ["{} parameter {} = {}".format("," if i > 0 else "", key, sdfg.constants[key]) for i, key in
                 enumerate(sdfg.constants)]))

        # construct input / output module header
        MAX_PADDING = 17
        inputs = list()
        for inp in tasklet.in_connectors:
            # add vector index
            idx_str = ""
            # catch symbolic (compile time variables)
            if symbolic.issymbolic(tasklet.in_connectors[inp].veclen, sdfg.constants):
                raise RuntimeError("Please use sdfg.specialize to specialize the symbol in expression: {}".format(
                    tasklet.in_connectors[inp].veclen))
            if symbolic.issymbolic(tasklet.in_connectors[inp].bytes, sdfg.constants):
                raise RuntimeError("Please use sdfg.specialize to specialize the symbol in expression: {}".format(
                    tasklet.in_connectors[inp].bytes))
            # extract parameters
            vec_len = int(symbolic.evaluate(tasklet.in_connectors[inp].veclen, sdfg.constants))
            total_size = int(symbolic.evaluate(tasklet.in_connectors[inp].bytes, sdfg.constants))
            # generate vector representation
            if vec_len > 1:
                idx_str = "[{}:0]".format(vec_len - 1)
            # add element index
            idx_str += "[{}:0]".format(int(total_size / vec_len) * 8 - 1)
            # generate padded string and add to list
            inputs.append(", input{padding}{idx_str} {name}".format(padding=" " * (MAX_PADDING - len(idx_str)),
                                                                    idx_str=idx_str,
                                                                    name=inp))
        MAX_PADDING = 12
        outputs = list()
        for inp in tasklet.out_connectors:
            # add vector index
            idx_str = ""
            # catch symbolic (compile time variables)
            if symbolic.issymbolic(tasklet.out_connectors[inp].veclen, sdfg.constants):
                raise RuntimeError("Please use sdfg.specialize to specialize the symbol in expression: {}".format(
                    tasklet.in_connectors[inp].veclen))
            if symbolic.issymbolic(tasklet.out_connectors[inp].bytes, sdfg.constants):
                raise RuntimeError("Please use sdfg.specialize to specialize the symbol in expression: {}".format(
                    tasklet.in_connectors[inp].bytes))
            # extract parameters
            vec_len = int(symbolic.evaluate(tasklet.out_connectors[inp].veclen, sdfg.constants))
            total_size = int(symbolic.evaluate(tasklet.out_connectors[inp].bytes, sdfg.constants))
            # generate vector representation
            if vec_len > 1:
                idx_str = "[{}:0]".format(vec_len - 1)
            # add element index
            idx_str += "[{}:0]".format(int(total_size / vec_len) * 8 - 1)
            # generate padded string and add to list
            outputs.append(
                ", output reg{padding}{idx_str} {name}".format(padding=" " * (MAX_PADDING - len(idx_str)),
                                                               idx_str=idx_str,
                                                               name=inp))
        # generate cpp input reading/output writing code
        """
        input:
        for vectors:
            for (int i = 0; i < WIDTH; i++){{ 
                model->a[i] = a[i];
            }}
        for scalars:
            model->a = a;

        output:
        for vectors:
            for(int i = 0; i < WIDTH; i++){{
                b[i] = (int)model->b[i];
            }}
        for scalars:
            b = (int)model->b;
        """

        input_read_string = "\n".join(
            ["model->{name} = {name}[in_ptr++];".format(name=var_name)
             if isinstance(tasklet.in_connectors[var_name], dace.dtypes.pointer) else
             """\
             for(int i = 0; i < {veclen}; i++){{
               model->{name}[i] = {name}[i];
             }}\
             """.format(veclen=tasklet.in_connectors[var_name].veclen, name=var_name)
             if isinstance(tasklet.in_connectors[var_name], dace.dtypes.vector) else
             "model->{name} = {name}[in_ptr++];".format(name=var_name)  # model->{name} = {name}; in_ptr++;
             for var_name in tasklet.in_connectors])

        output_read_string = "\n".join(["{name}[out_ptr++] = (int)model->{name};".format(name=var_name)
                                        if isinstance(tasklet.out_connectors[var_name], dace.dtypes.pointer) else
                                        """\
                                        for(int i = 0; i < {veclen}; i++){{
                                          {name}[i] = (int)model->{name}[i];
                                        }}\
                                        """.format(veclen=tasklet.out_connectors[var_name].veclen, name=var_name)
                                        if isinstance(tasklet.out_connectors[var_name], dace.dtypes.vector) else
                                        "{name}[out_ptr++] = (int)model->{name};".format(name=var_name)
                                        # {name} = (int)model->{name}; out_ptr++;
                                        for var_name in tasklet.out_connectors])

        init_vector_string = "\n".join(["""\
                                       for(int i = 0; i < {veclen}; i++){{
                                         model->{name}[i] = 0;
                                       }}\
                                       """.format(veclen=tasklet.in_connectors[var_name].veclen, name=var_name)
                                        if isinstance(tasklet.in_connectors[var_name], dace.dtypes.vector) else ""
                                        for var_name in tasklet.in_connectors])

        # create rtl code object (that is later written to file)
        self.code_objects.append(CodeObject(name="{}".format(unique_name),
                                            code=RTLCodeGen.rtl_header().format(name=unique_name,
                                                                                parameters=parameter_string,
                                                                                inputs="\n".join(inputs),
                                                                                outputs="\n".join(outputs)) + tasklet.code.code + RTLCodeGen.rtl_footer(),
                                            language="sv",
                                            target=RTLCodeGen,
                                            title="rtl",
                                            target_type="",
                                            additional_compiler_kwargs="",
                                            linkable=True,
                                            environments=None))

        # compute num_elements=#elements that enter/leave the pipeline, for now we assume in_elem=out_elem (i.e. no reduction)
        num_elements_string = "int num_elements = {};".format(1)

        sdfg.append_global_code(cpp_code=RTLCodeGen.header_template().format(name=unique_name,
                                                                             debug="// enable/disable debug log\n" +
                                                                                   "bool DEBUG = false;" if "DEBUG" not in sdfg.constants else ""))
        # dace.config.Config.get()
        callsite_stream.write(contents=RTLCodeGen.main_template().format(name=unique_name,
                                                                         inputs=input_read_string,
                                                                         outputs=output_read_string,
                                                                         num_elements=num_elements_string,
                                                                         vector_init=init_vector_string,
                                                                         internal_state_str=" ".join(
                                                                             ["{}=0x%x".format(var_name) for
                                                                              var_name
                                                                              in
                                                                              {**tasklet.in_connectors,
                                                                               **tasklet.out_connectors}]),
                                                                         internal_state_var=", ".join(
                                                                             ["model->{}".format(var_name) for
                                                                              var_name in
                                                                              {**tasklet.in_connectors,
                                                                               **tasklet.out_connectors}])),
                              sdfg=sdfg,
                              state_id=state_id,
                              node_id=node)


    # define cpp code templates
    @staticmethod
    def header_template():
        return """
                                    // generic includes
                                    #include <iostream>
                            
                                    // verilator includes
                                    #include <verilated.h>
                                    
                                    // include model header, generated from verilating the sv design
                                    #include "V{name}.h"
                                                               
                                    {debug}
                                    """


    @staticmethod
    def main_template():
        return """
                                std::cout << "SIM START" << std::endl;
        
                                vluint64_t main_time = 0;
                                
                                // instantiate model
                                V{name}* model = new V{name};
                            
                                // apply initial input values
                                model->rst_i = 0;  // no reset
                                model->clk_i = 0; // neg clock
                                model->valid_i = 0; // not valid
                                model->ready_i = 0; // not ready 
                                model->eval();
                                
                                // read inputs
                                //{{inputs}}
                                //model->eval();
                                
                                // init vector
                                {vector_init}
                            
                                // reset design
                                model->rst_i = 1;
                                model->clk_i = 1; // rising
                                model->eval();
                                model->clk_i = 0; // falling
                                model->eval();
                                model->rst_i = 0;
                                model->clk_i = 1; // rising
                                model->eval();
                                model->clk_i = 0; // falling
                                model->eval();
                                
                                // simulate until in_handshakes = out_handshakes = num_elements
                                bool read_input_hs = false, write_output_hs = false;
                                int in_ptr = 0, out_ptr = 0;
                                {num_elements}
                                
                                while (out_ptr < num_elements) {{
                            
                                    // increment time
                                    main_time++;
                            
                                    // check if valid_i and ready_o have been asserted at the rising clock edge -> input read handshake
                                    if (model->ready_o == 1 && model->valid_i == 1){{
                                        read_input_hs = true;
                                    }} 
                                    // feed new element
                                    if(model->valid_i == 0 && in_ptr < num_elements){{
                                        std::cout << "feed new element" << std::endl;
                                        //model->a = a[in_ptr++];
                                        {inputs}
                                        model->valid_i = 1;
                                    }}
                            
                                    // export element
                                    if(model->valid_o == 1){{
                                        std::cout << "export element" << std::endl;
                                        //b[out_ptr++] = model->b;
                                        {outputs}
                                        model->ready_i = 1;
                                    }}
                                    // check if valid_o and ready_i have been asserted at the rising clock edge -> output write handshake
                                    if (model->ready_i == 1 && model->valid_o == 1){{
                                        write_output_hs = true;
                                    }}
                            
                                    // positive clock edge
                                    model->clk_i = !model->clk_i;
                                    model->eval();
                            
                                    // report internal state
                                    if(DEBUG){{
                                        VL_PRINTF("[t=%lu] clk_i=%u rst_i=%u valid_i=%u ready_i=%u valid_o=%u ready_o=%u \\n", main_time, model->clk_i, model->rst_i, model->valid_i, model->ready_i, model->valid_o, model->ready_o);
                                        VL_PRINTF("{internal_state_str}", {internal_state_var});
                                        std::cout << std::endl;
                                    }}
                                    
                                    // check if valid_i and ready_o have been asserted at the rising clock edge
                                    if (read_input_hs){{
                                        // remove valid_i flag
                                        std::cout << "remove read_input_hs flag" << std::endl;
                                        model->valid_i = 0;
                                        read_input_hs = false;
                                    }}
                            
                                    // check if valid_o and ready_i have been asserted at the rising clock edge
                                    if (write_output_hs){{
                                        // remove valid_i flag
                                        std::cout << "remove write_output_hs flag" << std::endl;
                                        model->ready_i = 0;
                                        write_output_hs = false;
                                    }}
                                    
                                    
                                    // negative clock edge
                                    model->clk_i = !model->clk_i;
                                    model->eval();
                                }}
                           
                                // report internal state
                                if(DEBUG){{
                                    VL_PRINTF("[t=%lu] clk_i=%u rst_i=%u valid_i=%u ready_i=%u valid_o=%u ready_o=%u \\n", main_time, model->clk_i, model->rst_i, model->valid_i, model->ready_i, model->valid_o, model->ready_o);
                                    VL_PRINTF("{internal_state_str}", {internal_state_var});
                                    std::cout << std::endl;
                                }}
                           
                                // write result
                                //{{outputs}}
                                                    
                                // final model cleanup
                                model->final();
                            
                                // clean up resources
                                delete model;
                                model = NULL;
                                
                                std::cout << "SIM END" << std::endl;
                                """

    @staticmethod
    def rtl_header():
        return """\
        module {name}
        {parameters}
        ( input                  clk_i  // convention: clk_i clocks the design
        , input                  rst_i  // convention: rst_i resets the design
        , input                  valid_i
        , input                  ready_i
        {inputs}
        , output reg             ready_o
        , output reg             valid_o
        {outputs}
        );"""


    @staticmethod
    def rtl_footer(): return """
        endmodule
        """
