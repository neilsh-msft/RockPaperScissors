// Network layers - extracted from the test and example programs of CNTK.
//

#include "stdafx.h"
#include "Layers.h"

using namespace std;
using namespace CntkTraining;
using namespace CNTK;

FunctionPtr Layers::UniformLinear(Variable input, size_t outputDim, float bias, const DeviceDescriptor& device)
{
	assert(input.Shape().Rank() == 1);
	size_t inputDim = input.Shape()[0];

	auto weightParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ inputDim, outputDim }, -0.05, 0.05, 1, device));
	auto biasParam = CNTK::Parameter({ outputDim }, bias, device);

	return CNTK::Plus(biasParam, CNTK::Times(input, weightParam));
}

FunctionPtr Layers::Embedding(Variable input, size_t embeddingDim, const DeviceDescriptor& device)
{
	assert(input.Shape().Rank() == 1);
	size_t inputDim = input.Shape()[0];

	auto embeddingParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ inputDim, embeddingDim }, -0.05, 0.05, 1, device));

	return CNTK::Times(input, embeddingParam);
}

// Layer factory function to create an instance of a fully-connected linear layer of the form
// activation(input * W + b).
FunctionPtr Layers::Dense(size_t outputDim, const std::function<FunctionPtr(const FunctionPtr&)>& activation,
								ParameterInitializer& initializer,
								float bias, float init_bias, const DeviceDescriptor& device)
{
	auto weightParam = CNTK::Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, initializer, device);
	auto biasParam = CNTK::Parameter({ outputDim }, bias, device);
	auto input = CNTK::PlaceholderVariable();

	// activation = std::bind(Sigmoid, std::placeholders::_1, L"");

	// Wx + b
	return activation(CNTK::Plus(biasParam, CNTK::Times(input, weightParam)));
}

FunctionPtr Layers::LSTM(Variable input, size_t numOutputClasses, size_t hiddenDim, size_t cellDim, size_t lstmDim, const DeviceDescriptor& device)
{
	FunctionPtr classifierRoot = input;
	auto pastValueRecurrenceHook = [](const Variable& x) { return PastValue(x); };

	for (size_t i = 0; i < lstmDim; i++)
	{
		classifierRoot = LSTMPComponentWithSelfStabilization(classifierRoot, { hiddenDim }, { cellDim }, pastValueRecurrenceHook, pastValueRecurrenceHook, device).first;
	}

	auto W = Parameter(NDArrayView::RandomUniform<float>({ numOutputClasses, hiddenDim }, -0.5, 0.5, 1, device));
	auto b = Parameter({ numOutputClasses }, 0.0f, device);

	auto sW = Parameter({}, 0.0f, device);
	auto expsW = Exp(sW);

	return Plus(Times(W, ElementTimes(expsW, classifierRoot)), b);
}

std::pair<FunctionPtr, FunctionPtr> Layers::LSTMPComponentWithSelfStabilization(Variable input, const NDShape& outputShape, const NDShape& cellShape,
	const std::function<FunctionPtr(const Variable&)>& recurrenceHookH, const std::function<FunctionPtr(const Variable&)>& recurrenceHookC,
	const DeviceDescriptor& device)
{
	auto dh = CNTK::PlaceholderVariable(outputShape, input.DynamicAxes());
	auto dc = CNTK::PlaceholderVariable(cellShape, input.DynamicAxes());

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
	return CNTK::Slice(operand, CNTK::Axis::DefaultDynamicAxis(), -1, 0);
}

inline FunctionPtr Layers::Stabilize(const Variable& x, const DeviceDescriptor& device)
{
	float scalarConstant = 4.0f;
	auto f = Constant::Scalar(scalarConstant);
	auto fInv = Constant::Scalar(DataType::Float, 1.0 / scalarConstant);

	auto beta = ElementTimes(fInv, Log(Constant::Scalar(DataType::Float, 1.0) + Exp(ElementTimes(f, Parameter({}, DataType::Float, 0.99537863 /* 1/f*ln (e^f-1) */, device)))));
	return ElementTimes(beta, x);
}