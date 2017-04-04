#pragma once

#include <string>
#include "CNTKLibrary.h"

#define LOOKBACK_MOVES	5
#define GAME_LENGTH		20
#define HIDDEN_LAYERS	3

namespace CntkTraining
{
	class ModelTrainer
	{
	public:
		ModelTrainer(const std::wstring& modelFile, const std::wstring& dataFile);

		void Train();

	protected:
		void LoadModel();
		void CreateModel();
		CNTK::TrainerPtr CreateTrainerForModel();

		CNTK::FunctionPtr LinearLayer(CNTK::Variable input, size_t outputDim, const CNTK::DeviceDescriptor& device);
		CNTK::FunctionPtr Embedding(CNTK::Variable input, size_t embeddingDim, const CNTK::DeviceDescriptor& device);
		CNTK::FunctionPtr SelectLast(CNTK::Variable operand);
		std::vector<CNTK::FunctionPtr> LSTMPCellWithSelfStabilization(CNTK::Variable input, CNTK::Variable prevOutput, CNTK::Variable prevCellState, const CNTK::DeviceDescriptor& device);
		std::vector<CNTK::FunctionPtr> LSTMPComponentWithSelfStabilization(CNTK::Variable input, size_t outputDim, size_t cellDim, const CNTK::DeviceDescriptor& device);
		CNTK::FunctionPtr LSTMSequenceClassifierNet(CNTK::Variable input, size_t outputClasses, size_t embeddingDim,
			size_t lstmDim, size_t cellDim, const CNTK::DeviceDescriptor& device);

		CNTK::FunctionPtr _model;
		CNTK::Variable _inputs, _labels;
		std::wstring _modelFile, _dataFile;
	};
}
