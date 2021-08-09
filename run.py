#!/usr/bin/env python3
import glob
import tflite

def dump_tensor(tensor):
    shape_len = tensor.ShapeLength()
    result = []
    for shape_i in range(shape_len):
        shape = tensor.Shape(shape_i)
        result.append(str(shape))
    return 'x'.join(result)

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
    assert(model.Description().decode('utf-8') == 'TOCO Converted.')

    # How many operator types in this model.
    assert(model.OperatorCodesLength() == 5)

    # A model may have multiple subgraphs.
    assert(model.SubgraphsLength() == 1)

    # How many tensor buffer.
    assert(model.BuffersLength() == 90)

    # Chose one subgraph.
    graph = model.Subgraphs(0)

    # Tensors in the subgraph are represented by index description.
    assert(graph.InputsLength() == 1)
    assert(graph.OutputsLength() == 1)
    assert(graph.InputsAsNumpy()[0] == 88)
    assert(graph.OutputsAsNumpy()[0] == 87)
    # All arrays can dump as Numpy array, or access individually.
    assert(graph.Inputs(0) == 88)
    assert(graph.Outputs(0) == 87)

    # Name may used to debug or check for model containing multiple subgraphs.
    assert(graph.Name() == None)

    # Operators in the subgraph.
    assert(graph.OperatorsLength() == 31)

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
            print('\tinput:', input_i, 'index:', tensor_index, 'shape:', dump_tensor(tensor))

        output_len = op.OutputsLength()
        for output_i in range(output_len):
            tensor_index = op.Outputs(output_i)
            print('\toutput:', output_i, 'index:', tensor_index, 'shape:', dump_tensor(tensor))

        # Operator Type is also stored as index, which can obtain from `Model` object.


