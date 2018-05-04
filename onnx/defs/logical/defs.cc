// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

inline void logicalOpInference(InferenceContext& ctx) {
	updateOutputElemType(ctx, 0, TensorProto::BOOL);
	if (hasInputShape(ctx, 0)) {
		propagateShapeFromInputToOutput(ctx, 0, 0);
	}
}

std::function<void(OpSchema&)> BinaryLogicDocGenerator(const char* name) {
    return [=](OpSchema& schema) {
        std::string doc = R"DOC(
Returns the tensor resulted from performing the `{name}` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
)DOC";
        ReplaceAll(doc, "{name}", name);
        schema.SetDoc(doc);
        schema.Attr("broadcast", "Enable broadcasting", AttributeProto::INT, static_cast<int64_t>(0));
        schema.Attr("axis", "If set, defines the broadcast dimensions.",
                    AttributeProto::INT,
                    OPTIONAL);
        schema.Input(0, "A", "Left input tensor for the logical operator.", "T");
        schema.Input(1, "B", "Right input tensor for the logical operator.", "T");
		schema.Output(0, "C", "Result tensor.", "T1");
		schema.TypeAndShapeInferenceFunction(logicalOpInference);
    };
}

ONNX_OPERATOR_SCHEMA(And, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(BinaryLogicDocGenerator("and"))
    .TypeConstraint("T", { "tensor(bool)" },
                    "Constrains input to boolean tensor.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor."));

ONNX_OPERATOR_SCHEMA(Or, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(BinaryLogicDocGenerator("or"))
    .TypeConstraint("T", { "tensor(bool)" },
                    "Constrains input to boolean tensor.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor."));

ONNX_OPERATOR_SCHEMA(Xor, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(BinaryLogicDocGenerator("xor"))
    .TypeConstraint("T", { "tensor(bool)" },
                    "Constrains input to boolean tensor.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor."));

ONNX_OPERATOR_SCHEMA(Greater, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(BinaryLogicDocGenerator("greater"))
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                    "Constrains input to float tensors.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor."));

ONNX_OPERATOR_SCHEMA(Less, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(BinaryLogicDocGenerator("less"))
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                    "Constrains input to float tensors.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor."));

ONNX_OPERATOR_SCHEMA(Equal, ONNX_DOMAIN, 1, OpSchema()
    .FillUsing(BinaryLogicDocGenerator("equal"))
    .TypeConstraint("T", { "tensor(bool)", "tensor(int32)", "tensor(int64)" },
                    "Constrains input to integral tensors.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor."));

ONNX_OPERATOR_SCHEMA(Not, ONNX_DOMAIN, 1, OpSchema()
    .SetDoc(R"DOC(
Returns the negation of the input tensor element-wise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint("T", { "tensor(bool)" },
                    "Constrains input/output to boolean tensors.")
	.TypeAndShapeInferenceFunction(logicalOpInference));

}  // namespace ONNX_NAMESPACE
