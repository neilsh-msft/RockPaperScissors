#pragma once

#include <string>
#include "CNTKLibrary.h"

#define LOOKBACK_MOVES	5
#define GAME_LENGTH		20
#define HIDDEN_LAYERS	10

namespace CntkTraining
{
	class ModelTrainer
	{
	public:
		ModelTrainer(const std::wstring& modelFile, const std::wstring& dataFile);

		void Train();

	protected:
		void LoadModel();
		CNTK::TrainerPtr CreateTrainerForModel();

		CNTK::FunctionPtr _model;
		CNTK::Variable _inputs, _labels;
		std::wstring _modelFile, _dataFile;
	};
}
