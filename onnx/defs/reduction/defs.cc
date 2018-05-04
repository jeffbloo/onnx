// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include <functional>
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

std::function<void(OpSchema&)> ReduceDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
Computes the {name} of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.)DOC";
    ReplaceAll(doc, "{name}", name);
    schema.SetDoc(doc);
    schema.Attr(
        "axes",
        "A list of integers, along which to reduce. The default is to reduce over "
        "all the dimensions of the input tensor.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "keepdims",
        "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Input(0, "data", "An input tensor.", "T");
    schema.Output(0, "reduced", "Reduced output tensor.", "T");
    schema.TypeConstraint(
        "T",
        OpSchema::high_precision_numeric_types(),
        "Constrain input and output types to high-precision numeric tensors.");
  };
}

ONNX_OPERATOR_SCHEMA(ReduceMax, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(ReduceDocGenerator("max")));

ONNX_OPERATOR_SCHEMA(ReduceMin, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(ReduceDocGenerator("min")));

ONNX_OPERATOR_SCHEMA(ReduceSum, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(ReduceDocGenerator("sum")));

ONNX_OPERATOR_SCHEMA(ReduceSumSquare, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(ReduceDocGenerator("sum square")));

ONNX_OPERATOR_SCHEMA(ReduceMean, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(ReduceDocGenerator("mean")));

ONNX_OPERATOR_SCHEMA(ReduceProd, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(ReduceDocGenerator("product")));

ONNX_OPERATOR_SCHEMA(ReduceLogSum, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(ReduceDocGenerator("log sum")));

ONNX_OPERATOR_SCHEMA(ReduceLogSumExp, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(ReduceDocGenerator("log sum exponent")));

ONNX_OPERATOR_SCHEMA(ReduceL1, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(ReduceDocGenerator("L1 norm")));

ONNX_OPERATOR_SCHEMA(ReduceL2, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(ReduceDocGenerator("L2 norm")));

std::function<void(OpSchema&)> ArgReduceDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
Computes the indices of the {name} elements of the input tensor's element along the 
provided axis. The resulted tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned. 
The type of the output tensor is integer.)DOC";
    ReplaceAll(doc, "{name}", name);
    schema.SetDoc(doc);
    schema.Attr(
        "axis",
        "The axis in which to compute the arg indices. Default is 0.",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Attr(
        "keepdims",
        "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Input(0, "data", "An input tensor.", "T");
    schema.Output(
        0,
        "reduced",
        "Reduced output tensor with integer data type.",
        "tensor(int64)");
    schema.TypeConstraint(
        "T",
        OpSchema::all_numeric_types(),
        "Constrain input and output types to all numeric tensors.");
  };
}

ONNX_OPERATOR_SCHEMA(ArgMax, ONNX_DOMAIN, 1, OpSchema().FillUsing(ArgReduceDocGenerator("max")));

ONNX_OPERATOR_SCHEMA(ArgMin, ONNX_DOMAIN, 1, OpSchema().FillUsing(ArgReduceDocGenerator("min")));

} // namespace ONNX_NAMESPACE
