// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
namespace ONNX_NAMESPACE
{
  using SupportType = OpSchema::SupportType;

using SupportType = ONNX_NAMESPACE::OpSchema::SupportType;

ONNX_OPERATOR_SCHEMA(Affine, ONNX_DOMAIN, 1, OpSchema()
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .SetDoc(R"DOC(
Affine takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the affine function, y = alpha * x + beta,
is applied to the tensor elementwise.
)DOC")
    .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, 1.0f)
    .Attr("beta" , "Value of beta", AttributeProto::FLOAT, 0.0f)
    .Input(0, "X", "1D input tensor", "T")
    .Output(0, "Y", "1D output tensor", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.")
    .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SCHEMA(ThresholdedRelu, ONNX_DOMAIN, 1, OpSchema()
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .SetDoc(R"DOC(
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.
)DOC")
    .Attr("alpha",
          "Threshold value",
          AttributeProto::FLOAT,
          1.0f)
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.")
    .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SCHEMA(ScaledTanh, ONNX_DOMAIN, 1, OpSchema()
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .SetDoc(R"DOC(
Calculates the scaled hyperbolic tangent of the given input tensor element-wise,
alpha * tanh(beta * x). This operation can be done in an in-place fashion too,
by providing the same input and output blobs.
    )DOC")
    .Attr("alpha", "Scaling value", AttributeProto::FLOAT, OPTIONAL)
    .Attr("beta", "Scaling value", AttributeProto::FLOAT, OPTIONAL)
    .Input(0, "input", "Input tensor", "T")
    .Output(0, "output", "The scaled hyperbolic tangent values of the input tensor "
        "computed element-wise", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.")
    .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SCHEMA(ParametricSoftplus, ONNX_DOMAIN, 1, OpSchema()
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .SetDoc(R"DOC(
ParametricSoftplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = alpha * ln(exp(beta * x) + 1), is applied to
the tensor elementwise.
)DOC")
    .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, OPTIONAL)
    .Attr("beta", "Value of beta", AttributeProto::FLOAT, OPTIONAL)
    .Input(0, "X", "1D input tensor", "T")
    .Output(0, "Y", "1D input tensor", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.")
    .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SCHEMA(ConstantFill, ONNX_DOMAIN, 1, OpSchema()
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .SetDoc(R"DOC(
The operator fills the elements of the output tensor with a constant value
specified by the 'value' attribute.

The data type is specified by the 'dtype' attribute. The 'dtype' attribute must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message. If the 'dtype' attribute is not provided, the data type of
'value' is used.

The output tensor shape is specified by the 'shape' attribute. If the number of
input is 1, the shape will be identical to that of the input at run time with
optional additional dimensions appended at the end as specified by 'extra_shape'
attribute. In that case the 'shape' attribute should not be set.

If input_as_shape is set to true, then the input should be a 1D tensor
containing the desired output shape (the dimensions specified in extra_shape
will also be appended)

NOTE: Currently, it supports data type of float, int32, int64, and bool.
)DOC")
    .Attr(
        "value",
        "The value for the elements of the output tensor. Default is 0.",
        AttributeProto::FLOAT,
        0.0f)
    .Attr(
        "dtype",
        "The data type for the elements of the output tensor."
        "Strictly must be one of the types from DataType enum in TensorProto.",
        AttributeProto::INT,
        static_cast<int64_t>(TensorProto::FLOAT))
    .Attr(
        "shape",
        "The shape of the output tensor. "
        "Cannot set the shape argument and pass in an input at the same time.",
        AttributeProto::INTS,
        OPTIONAL)
    .Attr(
        "extra_shape",
        "The additional dimensions appended at the end of the shape indicated"
        "by the input blob."
        "Cannot set the extra_shape argument when there is no input blob.",
        AttributeProto::INTS,
        OPTIONAL)
    .Attr(
        "input_as_shape",
        "1D tensor containing the desired output shape.  First input must be in "
        "CPU context.",
        AttributeProto::INT,
        OPTIONAL)
    .Input(
        0,
        "input",
        "Input tensor (optional) to provide shape information.",
        "T1",
        OpSchema::Optional)
    .Output(
        0,
        "output",
        "Output tensor of constant values specified by 'value'"
        "argument and its type is specified by the 'dtype' argument",
        "T2")
    .TypeConstraint(
        "T1",
        {"tensor(float)", "tensor(int32)", "tensor(int64)", "tensor(bool)"},
        "Constrain input types to float, int32, int64, bool tensors.")
    .TypeConstraint(
        "T2",
        {"tensor(float)", "tensor(int32)", "tensor(int64)", "tensor(bool)"},
        "Constrain output types to float, int32, int64, bool tensors.")
    .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0, TensorProto::FLOAT);
        if (ctx.getAttribute("shape") != nullptr) {
            propagateShapeFromAttributeToOutput(ctx, "shape", 0);
            return;
        }
        if (getAttribute(ctx, "input_as_shape", 0) != 0) // dynamic shape
            return;
        std::vector<int64_t> extra_shape;
        getRepeatedAttribute(ctx, "extra_shape", extra_shape);
        if (hasInputShape(ctx, 0)) {
            TensorShapeProto shape = ctx.getInputType(0)->tensor_type().shape();
            for (auto extra_dim_val : extra_shape) {
                if (extra_dim_val < 0) return;
                shape.add_dim()->set_dim_value(extra_dim_val);
            }
            updateOutputShape(ctx, 0, shape);
        }
    }));

ONNX_OPERATOR_SCHEMA(GivenTensorFill, ONNX_DOMAIN, 1, OpSchema()
.SetSupportLevel(SupportType::EXPERIMENTAL)
.Input(0, "shape", "The shape of filled tensor", "T", OpSchema::Optional)
.Output(0, "X", "The filled tensor", "T")
.TypeConstraint(
    "T",
    { "tensor(float16)", "tensor(float)", "tensor(double)" },
    "Constrain input and output types to float tensors.")
    .Attr("values", "", AttributeProto::FLOATS, OPTIONAL)
    .Attr("shape", "", AttributeProto::INTS, OPTIONAL)
    .Attr("input_as_shape", "", AttributeProto::INT, OPTIONAL)
    .Attr("extra_shape", "", AttributeProto::INTS, OPTIONAL)
    .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (ctx.getAttribute("shape") != nullptr) {
            propagateShapeFromAttributeToOutput(ctx, "shape", 0);
            return;
        }
        // The type constraints above do not allow for input_as_shape
        // and may need to be fixed.
        if (getAttribute(ctx, "input_as_shape", 0) != 0) // dynamic shape
            return;
        std::vector<int64_t> extra_shape;
        getRepeatedAttribute(ctx, "extra_shape", extra_shape);
        if (hasInputShape(ctx, 0)) {
            TensorShapeProto shape = ctx.getInputType(0)->tensor_type().shape();
            for (auto extra_dim_val : extra_shape) {
                if (extra_dim_val < 0) return;
                shape.add_dim()->set_dim_value(extra_dim_val);
            }
            updateOutputShape(ctx, 0, shape);
        }
    }));

ONNX_OPERATOR_SCHEMA(Scale, ONNX_DOMAIN, 1, OpSchema()
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .Input(0, "input", "Input data to be scaled", "T")
    .Output(0, "output", "Output data after scaling", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .SetDoc(R"DOC(
Scale takes one input data (Tensor<float>) and produces one output data
(Tensor<float>) whose value is the input data tensor scaled element-wise.
)DOC")
    .Attr("scale", "(float, default 1.0) the scale to apply.", AttributeProto::FLOAT, 1.0f)
    .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SCHEMA(GRUUnit, ONNX_DOMAIN, 1, OpSchema()
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .SetDoc(R"DOC(
GRUUnit computes the activations of a standard GRU,
in a sequence-length aware fashion.
Concretely, given the (fused) inputs X (TxNxD), the previous hidden
state (NxD), and the sequence lengths (N), computes the GRU
activations, avoiding computation if the input is invalid (as in, the
value at X[t][n] >= seqLengths[n].
)DOC")
    .Attr(
        "drop_states",
        "Bool to determine if hidden state is zeroes or passed "
        "along for timesteps past the given sequence_length.",
        AttributeProto::INT,
        OPTIONAL)
    .Input(0, "hidden_prev", "The previous GRU hidden state.", "T")
    .Input(
        1,
        "gates",
        "Unactivated gate outputs from forget, update, "
        "and output gates, pre-activation.",
        "T")
    .Input(
        2,
        "seq_lengths",
        "Array of sequence lengths.  "
        "len(seq_lengths) should equal batch size N.",
        "T")
    .Input(3, "t", "The timestep for this operation.", "T")
    .Output(0, "hidden", "The new GRU hidden state calculated by this op.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors."));

ONNX_OPERATOR_SCHEMA(ATen, ONNX_DOMAIN, 1, OpSchema()
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .AllowUncheckedAttributes()
    .SetDoc(R"DOC(
Experimental allowing ATen operations to be accessed directly from Caffe2
to allow for quick prototyping when ONNX is missing standard versions of
and op)DOC")
    .Input(0, "input", "Arbitrary input", "T", OpSchema::Variadic)
    .Output(0, "output", "Arbitrary output", "T", OpSchema::Variadic)
    .TypeConstraint("T",
        { "tensor(bool)", "tensor(int32)", "tensor(int64)",
        "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain output types to bool, int32, int64, float16, float, double tensors."));

ONNX_OPERATOR_SCHEMA(ImageScaler, ONNX_DOMAIN, 1, OpSchema()
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .SetDoc(R"DOC(Scale and bias the input image. Bias values are stored in
the same ordering as the image pixel format.)DOC")
    .Attr("bias", "Bias applied to each channel, same size as C.", AttributeProto::FLOATS, OPTIONAL)
    .Attr("scale", "(float, default 1.0) the scale to apply.", AttributeProto::FLOAT, 1.0f)
    .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
    .Output(0, "output", "Result, has same shape and type as input", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SCHEMA(MeanVarianceNormalization, ONNX_DOMAIN, 1, OpSchema()
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .SetDoc(R"DOC(Perform mean variance normalization.)DOC")
    .Attr("across_channels", "If 1, mean and variance are computed across channels. Default is 0.", AttributeProto::INT, static_cast<int64_t>(0))
    .Attr("normalize_variance", "If 0, normalize the mean only.  Default is 1.", AttributeProto::INT, static_cast<int64_t>(1))
    .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
    .Output(0, "output", "Result, has same shape and type as input", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SCHEMA(Crop, ONNX_DOMAIN, 1, OpSchema()
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .SetDoc(R"DOC(Crop and image to the specified spatial dimensions. If scale is given,
then optionally start the crop offset by the left/top border amounts.
If scale is not provided, crop the borders as provided.)DOC")
    .Attr("border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).", AttributeProto::INTS, OPTIONAL)
    .Attr("scale", "A 1-D values of (height, width).", AttributeProto::INTS, OPTIONAL)
    .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
    .Output(0, "output", "Result, has same type as input, with H and W dimensions reduced.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors."));
}
