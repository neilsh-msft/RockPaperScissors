#pragma once

#include <string>
#include "CNTKLibrary.h"

#include "TrainingData.h"

#define TRAINING_FACTOR    3
#define NUMBER_OF_FEATURES 7
#define GAME_LENGTH		   20
#define HIDDEN_LAYERS	   3
#define HIDDEN_LAYERS_DIM  64
#define LSTM_CELL_DIM      128

#define LSTM_NETWORK       1
#define TRAIN_ONLY         0

namespace CntkTraining
{
	class ModelTrainer
	{
	public:
		ModelTrainer(const std::wstring& dataFile, const std::wstring& modelFile);

		void Train();

	private:
		void CreateModel();
		CNTK::TrainerPtr CreateTrainerForModel();
		void ModelTrainer::EncodeBatch(TrainingData *loader, int trainingPosition, std::vector<std::vector<float>> trainingData,
			std::vector<std::vector<float>>& features, std::vector<std::vector<float>>& labels);
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
