#pragma once

#include <string>
#include "CNTKLibrary.h"

namespace CntkTraining
{
	class Layers
	{
	public:
		Layers() {}

		static CNTK::FunctionPtr Dense(CNTK::Variable input, size_t outputDim, CNTK::ParameterInitializer& initializer = CNTK::GlorotUniformInitializer(),
			const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&)>& activation = Identity,
			bool bias = true, float init_bias = 0.0f, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());
		static CNTK::FunctionPtr Dense(CNTK::Variable input, size_t outputDim, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());
		static CNTK::FunctionPtr UniformLinear(CNTK::Variable input, size_t outputDim, float bias = 0.0f,
			const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());
		static CNTK::FunctionPtr LSTM(CNTK::Variable input, size_t numOutputClasses, size_t hiddenDim, size_t cellDim, size_t lstmCells, bool enableSelfStabilization = false,
			const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());
		static CNTK::FunctionPtr Embedding(CNTK::Variable input, size_t embeddingDim, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());
		//static CNTK::FunctionPtr Recurrence(CNTK::FunctionPtr stepFunction);

	protected:
		static std::pair<CNTK::FunctionPtr, CNTK::FunctionPtr> LSTMPComponent(CNTK::Variable input, const CNTK::NDShape& outputShape, const CNTK::NDShape& cellShape,
			const std::function<CNTK::FunctionPtr(const CNTK::Variable&)>& recurrenceHookH = [](const CNTK::Variable& x) { return CNTK::PastValue(x); },
			const std::function<CNTK::FunctionPtr(const CNTK::Variable&)>& recurrenceHookC = [](const CNTK::Variable& x) { return CNTK::PastValue(x); },
			bool enableSelfStabilization = false, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());
		static std::pair<CNTK::FunctionPtr, CNTK::FunctionPtr> LSTMPCell(CNTK::Variable input, CNTK::Variable prevOutput, CNTK::Variable prevCellState,
			bool enableSelfStabilization = false, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());

		static CNTK::FunctionPtr Identity(CNTK::Variable keep);
		static CNTK::FunctionPtr SelectLast(CNTK::Variable operand);
		static CNTK::FunctionPtr Stabilize(const CNTK::Variable& x, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());
	};
}

