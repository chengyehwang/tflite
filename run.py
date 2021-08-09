import os
import tflite

if True:
    ## ref https://github.com/jackwish/tflite/blob/master/tests/test_mobilenet.py
    path = os.path.join(tflm_dir, tflm_name)
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

    # Name may used to debug or check for model containing multiple subgraphs.
    assert(graph.Name() == None)

    # Operators in the subgraph.
    assert(graph.OperatorsLength() == 31)

    # Let's use the first operator.
    op = graph.Operators(0)

    # Operator Type is also stored as index, which can obtain from `Model` object.
    op_code = model.OperatorCodes(op.OpcodeIndex())

    print(op_code)

