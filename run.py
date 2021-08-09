#!/usr/bin/env python3
import glob
import tflite

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

    # Let's use the first operator.
    op = graph.Operators(0)

    # Operator Type is also stored as index, which can obtain from `Model` object.
    op_code = model.OperatorCodes(op.OpcodeIndex())

    print(op_code)

