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
		void CreateModel();
		CNTK::TrainerPtr CreateTrainerForModel();

		CNTK::FunctionPtr FullyConnectedFeedForwardClassifierNetWithSharedParameters(CNTK::Variable input,
			size_t numHiddenLayers,
			const CNTK::Parameter& inputTimesParam,
			const CNTK::Parameter& inputPlusParam,
			const CNTK::Parameter hiddenLayerTimesParam[],
			const CNTK::Parameter hiddenLayerPlusParam[],
			const CNTK::Parameter& outputTimesParam,
			const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&)>& nonLinearity);
		CNTK::FunctionPtr FullyConnectedDNNLayerWithSharedParameters(CNTK::Variable input, const CNTK::Parameter& timesParam, const CNTK::Parameter& plusParam,
			const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&)>& nonLinearity);
		CNTK::FunctionPtr SetupFullyConnectedLinearLayer(CNTK::Variable input, size_t outputDim, const CNTK::DeviceDescriptor& device,
			const std::wstring& outputName = L"");
		CNTK::FunctionPtr SetupFullyConnectedDNNLayer(CNTK::Variable input, size_t outputDim, const CNTK::DeviceDescriptor& device,
			const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&)>& nonLinearity);

		CNTK::FunctionPtr _model;
		CNTK::Variable _inputs, _labels;
		std::wstring _modelFile, _dataFile;
	};
}
