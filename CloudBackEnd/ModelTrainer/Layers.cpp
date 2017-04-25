// Network layers - extracted from the test and example programs of CNTK.
//

#include "stdafx.h"
#include "Layers.h"

using namespace std;
using namespace CntkTraining;
using namespace CNTK;

FunctionPtr Layers::UniformLinear(Variable input, size_t outputDim, float bias, const DeviceDescriptor& device)
{
	return Dense(input, outputDim, CNTK::UniformInitializer(0.05), Identity, true, bias, device);

	/* assert(input.Shape().Rank() == 1);
	size_t inputDim = input.Shape()[0];

	auto weightParam = Parameter(NDArrayView::RandomUniform<float>({ outputDim, inputDim }, -0.05, 0.05, 1, device));
	auto biasParam = Parameter({ outputDim }, bias, device);

	return Plus(Times(weightParam, input), biasParam); */
}

FunctionPtr Layers::Embedding(Variable input, size_t embeddingDim, const DeviceDescriptor& device)
{
	assert(input.Shape().Rank() == 1);
	size_t inputDim = input.Shape()[0];

	auto embeddingParam = Parameter(NDArrayView::RandomUniform<float>({ inputDim, embeddingDim }, -0.05, 0.05, 1, device));

	return Times(input, embeddingParam);
}

// Layer factory function to create an instance of a fully-connected linear layer of the form
// activation(W * input + b).
FunctionPtr Layers::Dense(Variable input, size_t outputDim, ParameterInitializer& initializer, const std::function<FunctionPtr(const FunctionPtr&)>& activation,
								bool bias, float init_bias, const DeviceDescriptor& device)
{
	auto weightParam = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, initializer, device);
	auto biasParam = Parameter({ outputDim }, init_bias, device);

	// Wx + b
	return activation(Plus(Times(weightParam, input), biasParam));
}

FunctionPtr Layers::Dense(CNTK::Variable input, size_t outputDim, const DeviceDescriptor& device)
{
	return Dense(input, outputDim, CNTK::GlorotUniformInitializer(), Identity, true, 0.0f, device);
}


FunctionPtr Layers::LSTM(Variable input, size_t numOutputClasses, size_t hiddenDim, size_t cellDim, size_t lstmCells, const DeviceDescriptor& device)
{
	FunctionPtr classifierRoot = input;
	auto pastValueRecurrenceHook = [](const Variable& x) { return PastValue(x); };
	// auto futureValueRecurrenceHook = [](const Variable& x) { return FutureValue(x); };

	for (size_t i = 0; i < lstmCells; i++)
	{
		classifierRoot = LSTMPComponentWithSelfStabilization(classifierRoot, { hiddenDim }, { cellDim }, pastValueRecurrenceHook, pastValueRecurrenceHook, device).first;
	}

	return classifierRoot;
}

std::pair<FunctionPtr, FunctionPtr> Layers::LSTMPComponentWithSelfStabilization(Variable input, const NDShape& outputShape, const NDShape& cellShape,
	const std::function<FunctionPtr(const Variable&)>& recurrenceHookH, const std::function<FunctionPtr(const Variable&)>& recurrenceHookC,
	const DeviceDescriptor& device)
{
	auto dh = PlaceholderVariable(outputShape, input.DynamicAxes());
	auto dc = PlaceholderVariable(cellShape, input.DynamicAxes());

	auto LSTMCell = LSTMPCellWithSelfStabilization(input, dh, dc, device);
	auto actualDh = recurrenceHookH(LSTMCell.first);
	auto actualDc = recurrenceHookC(LSTMCell.second);

	// Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
	LSTMCell.first->ReplacePlaceholders({ { dh, actualDh },{ dc, actualDc } });

	return{ LSTMCell.first, LSTMCell.second };
}

std::pair<FunctionPtr, FunctionPtr> Layers::LSTMPCellWithSelfStabilization(Variable input, Variable prevOutput, Variable prevCellState, const DeviceDescriptor& device)
{
	size_t outputDim = prevOutput.Shape()[0];
    size_t cellDim = prevCellState.Shape()[0];

    auto createBiasParam = [device](size_t dim) {
        return Parameter({ dim }, 0.0f, device);
    };

    unsigned long seed2 = 1;
    auto createProjectionParam = [device, &seed2](size_t outputDim) {
        return Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(1.0, 1, 0, seed2++), device);
    };

    auto createDiagWeightParam = [device, &seed2](size_t dim) {
        return Parameter({ dim }, DataType::Float, GlorotUniformInitializer(1.0, 1, 0, seed2++), device);
    };

    auto stabilizedPrevOutput = Stabilize(prevOutput, device);
    auto stabilizedPrevCellState = Stabilize(prevCellState, device);

    auto projectInput = [input, cellDim, createBiasParam, createProjectionParam]() {
        return createBiasParam(cellDim) + Times(createProjectionParam(cellDim), input);
    };

    // Input gate
    auto it = Sigmoid(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput) + ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
    auto bit = ElementTimes(it, Tanh(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput)));

    // Forget-me-not gate
    auto ft = Sigmoid(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput) + ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
    auto bft = ElementTimes(ft, prevCellState);

    auto ct = bft + bit;

    // Output gate
    auto ot = Sigmoid(projectInput() + Times(createProjectionParam(cellDim), stabilizedPrevOutput) + ElementTimes(createDiagWeightParam(cellDim), Stabilize(ct, device)));
    auto ht = ElementTimes(ot, Tanh(ct));

    auto c = ct;
    auto h = (outputDim != cellDim) ? Times(createProjectionParam(outputDim), Stabilize(ht, device)) : ht;

    return{ h, c };
}

// Utility functions
inline FunctionPtr Layers::Identity(CNTK::Variable keep)
{
	return CNTK::Combine({ keep });
}

inline FunctionPtr Layers::SelectLast(Variable operand)
{
	return CNTK::Slice(operand, { CNTK::Axis::DefaultDynamicAxis() }, { -1 }, { 0 });
}

inline FunctionPtr Layers::Stabilize(const Variable& x, const DeviceDescriptor& device)
{
	float scalarConstant = 4.0f;
	auto f = Constant::Scalar(scalarConstant);
	auto fInv = Constant::Scalar(DataType::Float, 1.0 / scalarConstant);

	auto beta = ElementTimes(fInv, Log(Constant::Scalar(DataType::Float, 1.0) + Exp(ElementTimes(f, Parameter({}, DataType::Float, 0.99537863 /* 1/f*ln (e^f-1) */, device)))));
	return ElementTimes(beta, x);
}