#pragma once

#include <string>
#include "CNTKLibrary.h"

namespace CntkTraining
{
	class Layers
	{
	public:
		Layers() {}

		static CNTK::FunctionPtr Dense(size_t outputDim, const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&)>& activation = Identity,
			CNTK::ParameterInitializer& initializer = CNTK::GlorotUniformInitializer(),
			float bias = 0.0f, float init_bias = 0.0f, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());
		static CNTK::FunctionPtr Layers::UniformLinear(CNTK::Variable input, size_t outputDim, float bias, 
			const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());
		//static CNTK::FunctionPtr Recurrence(CNTK::FunctionPtr stepFunction);
		static CNTK::FunctionPtr LSTM(CNTK::Variable input, size_t numOutputClasses, size_t hiddenDim, size_t cellDim, size_t lstmDim, 
			const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());

	protected:
		static std::pair<CNTK::FunctionPtr, CNTK::FunctionPtr> LSTMPComponentWithSelfStabilization(CNTK::Variable input, const CNTK::NDShape& outputShape, const CNTK::NDShape& cellShape,
			const std::function<CNTK::FunctionPtr(const CNTK::Variable&)>& recurrenceHookH = [](const CNTK::Variable& x) { return CNTK::PastValue(x); },
			const std::function<CNTK::FunctionPtr(const CNTK::Variable&)>& recurrenceHookC = [](const CNTK::Variable& x) { return CNTK::PastValue(x); },
			const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());
		static std::pair<CNTK::FunctionPtr, CNTK::FunctionPtr> Layers::LSTMPCellWithSelfStabilization(CNTK::Variable input, CNTK::Variable prevOutput, CNTK::Variable prevCellState,
			const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());

		static CNTK::FunctionPtr Identity(CNTK::Variable keep);
		static CNTK::FunctionPtr Layers::Embedding(CNTK::Variable input, size_t embeddingDim, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());
		static CNTK::FunctionPtr SelectLast(CNTK::Variable operand);
		static CNTK::FunctionPtr Stabilize(const CNTK::Variable& x, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());
	};
}

