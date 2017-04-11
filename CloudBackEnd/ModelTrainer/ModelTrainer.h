#pragma once

#include <string>
#include "CNTKLibrary.h"

#define TRAINING_FACTOR   3
#define GAME_LENGTH		  20
#define HIDDEN_LAYERS	  3
#define HIDDEN_LAYERS_DIM 64
#define LSTM_NETWORK      1
#define LSTM_CELL_DIM     128

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
		CNTK::FunctionPtr LSTMSequenceClassifierNet(CNTK::Variable input, size_t outputClasses, size_t hiddenDim, size_t cellDim, size_t lstmCells, 
			const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());
		CNTK::FunctionPtr FeedForwardClassifier(CNTK::Variable input, size_t outputClasses, size_t hiddenLayersDim, 
			const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::UseDefaultDevice());

		CNTK::FunctionPtr _model;
		CNTK::Variable _inputs, _labels;
		std::wstring _modelFile, _dataFile;
		int lookbackMoves;
	};
}
