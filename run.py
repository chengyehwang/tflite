#!/usr/bin/env python3
import glob
import tflite
import numpy as np

def dump_shape(tensor):
    shape_len = tensor.ShapeLength()
    result = []
    for shape_i in range(shape_len):
        shape = tensor.Shape(shape_i)
        result.append(str(shape))
    return 'x'.join(result)

def dump_type(tensor):
    type = tensor.Type()
    if type == tflite.TensorType.UINT8:
        return "UINT8"
    elif type == tflite.TensorType.INT8:
        return "INT8"
    elif type == tflite.TensorType.INT16:
        return "INT16"
    elif type == tflite.TensorType.INT32:
        return "INT32"
    elif type == tflite.TensorType.FLOAT16:
        return "FLOAT16"
    elif type == tflite.TensorType.FLOAT32:
        return "FLOAT32"
    else:
        return "UNKNOWN(%d)"%type

param = {}
def dump_var(tensor):
    global model
    buf = model.Buffers(tensor.Buffer())
    nonzero = np.count_nonzero(buf.DataAsNumpy())
    #return tensor.IsVariable()
    if nonzero != 0:
        return 'param'
    else:
        return 'var'

if True:
    ## ref https://github.com/jackwish/tflite/blob/master/tests/test_mobilenet.py
    files = glob.glob('*.tflite')
    path = files[0]
    with open(path, 'rb') as f:
        buf = f.read()
        model = tflite.Model.GetRootAsModel(buf, 0)
    
    # Version of the TFLite Converter.
    assert(model.Version() == 3)
    
    # Strings are binary format, need to decode.
    # Description is useful when exchanging models.
    print('Description:', model.Description().decode('utf-8'))

    # How many operator types in this model.
    print('OperatorCodesLength:', model.OperatorCodesLength())

    # A model may have multiple subgraphs.
    print('SubgraphsLength:', model.SubgraphsLength())

    # How many tensor buffer.
    print('BuffersLength:', model.BuffersLength())

    # Chose one subgraph.
    graph = model.Subgraphs(0)

    # Tensors in the subgraph are represented by index description.
    assert(graph.InputsLength() == 1)
    assert(graph.OutputsLength() == 1)

    # Name may used to debug or check for model containing multiple subgraphs.
    print('Name:', graph.Name())

    # Operators in the subgraph.
    print('OperatorsLength:', graph.OperatorsLength())

    is_var = {}
    # dump input
    input_len = graph.InputsLength()
    print('input:')
    for input_i in range(input_len):
        tensor_index = graph.Inputs(input_i)
        tensor = graph.Tensors(tensor_index)
        shape = dump_shape(tensor)
        type = dump_type(tensor)
        var = dump_var(tensor)
        is_var[tensor_index] = 1
        print('\toutput:', input_i, 'index:', tensor_index, 'shape:', shape, 'type:', type, var)
    # dump each op
    op_len = graph.OperatorsLength()
    for op_i in range(op_len):
        op = graph.Operators(op_i)
        op_code = tflite.opcode2name(op.OpcodeIndex())

        print('layer:', op_i, 'op_code:', op_code)

        input_len = op.InputsLength()
        for input_i in range(input_len):
            tensor_index = op.Inputs(input_i)
            tensor = graph.Tensors(tensor_index)
            shape = dump_shape(tensor)
            type = dump_type(tensor)
            var = dump_var(tensor)
            if var == 'var':
                if tensor_index in is_var:
                    pass
                else:
                    print("Error:unknown var")
            print('\tinput:', input_i, 'index:', tensor_index, 'shape:', shape, 'type:', type, var)

        output_len = op.OutputsLength()
        for output_i in range(output_len):
            tensor_index = op.Outputs(output_i)
            tensor = graph.Tensors(tensor_index)
            shape = dump_shape(tensor)
            type = dump_type(tensor)
            var = dump_var(tensor)
            is_var[tensor_index] = 1
            print('\toutput:', output_i, 'index:', tensor_index, 'shape:', shape, 'type:', type, var)

        # Operator Type is also stored as index, which can obtain from `Model` object.

    if False:
        meta_len = model.MetadataLength()
        print("meta_len:", meta_len)
        for meta_i in range(meta_len):
            meta_data = model.Metadata(meta_i)
            buf = meta_data.Buffer()
            print(meta_i, buf, meta_data)

        meta_len = model.MetadataBufferLength()
        print("meta_len:", meta_len)
        for meta_i in range(meta_len):
            meta_data = model.MetadataBuffer(meta_i)
            buf = meta_data.Buffer()
            print(meta_i, buf, meta_data)

