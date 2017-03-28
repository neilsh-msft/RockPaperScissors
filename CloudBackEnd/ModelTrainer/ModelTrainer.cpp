// ModelTrainer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ModelTrainer.h"
#include "TrainingData.h"

using namespace std;
using namespace CntkTraining;
using namespace CNTK;

#ifndef MODELTRAINERLIB

int TrainModel(const wchar_t* modelFilePath, const wchar_t* dataFilePath);

int wmain(int argc, wchar_t**argv)
{
	if (argc != 3)
	{
		cout << "Syntax: ModelTrainer.exe <modelfile.model> <gamefile.csv>\n";
		return 1;
	}

	const wchar_t* modelFilePath = argv[1];
	const wchar_t* dataFilePath = argv[2];

	return TrainModel(modelFilePath, dataFilePath);
}
#endif

inline bool DoesFileExist(const wchar_t* fileName) {
	ifstream f(fileName);
	return f.good();
}

#ifdef MODELTRAINERLIB
extern "C" __declspec(dllexport) 
#endif
int TrainModel(const wchar_t* modelFilePath, const wchar_t* dataFilePath)
{
#ifdef MODELTRAINERLIB
	OutputDebugString(L"Invoked TrainModel");
#endif

	try
	{
		if (!DoesFileExist(modelFilePath))
			return 10;

		if (!DoesFileExist(dataFilePath))
			return 11;

		if (!DoesFileExist(modelFilePath) || !DoesFileExist(dataFilePath))
		{
			cout << "Cannot find one of the input files. Both must exist";
			return 1;
		}

		ModelTrainer *trainer = new ModelTrainer(modelFilePath, dataFilePath);
		trainer->Train();
		delete(trainer);
		return 0;
	}
	catch (...)
	{
		return -1;
	}
}

ModelTrainer::ModelTrainer(const wstring& modelFile, const wstring& dataFile)
{
	_modelFile = modelFile;
	_dataFile = dataFile;
}

void ModelTrainer::LoadModel()
{
	// We could create the model from scratch, but its schema is already defined by the model file...
	_model = Function::LoadModel(_modelFile);

	_inputs = _model->Arguments()[0];
	_labels = _model->Output();
}

void ModelTrainer::CreateModel()
{
	// Define the model from scratch for training
	size_t inputDim = 7 * LOOKBACK_MOVES;
	size_t outputClasses = 3;
	const size_t numHiddenLayers = HIDDEN_LAYERS;
	size_t hiddenLayersDim = 32;
	CNTK::DeviceDescriptor device = DeviceDescriptor::DefaultDevice();

	_inputs = InputVariable({ inputDim }, CNTK::DataType::Float, L"Feature Vector");
	_labels = InputVariable({ outputClasses }, CNTK::DataType::Float, L"Labels");
	
	// Define model parameters that should be shared among evaluation requests against the same model
	auto inputTimesParam = Parameter(NDArrayView::RandomUniform<float>({ hiddenLayersDim, inputDim }, -0.5, 0.5, 1, device));
	auto inputPlusParam = Parameter({ hiddenLayersDim }, 0.0f, device);
	Parameter hiddenLayerTimesParam[numHiddenLayers - 1] = {
		Parameter(NDArrayView::RandomUniform<float>({ hiddenLayersDim, hiddenLayersDim }, -0.5, 0.5, 1, device)),
		Parameter(NDArrayView::RandomUniform<float>({ hiddenLayersDim, hiddenLayersDim }, -0.5, 0.5, 1, device)),
		Parameter(NDArrayView::RandomUniform<float>({ hiddenLayersDim, hiddenLayersDim }, -0.5, 0.5, 1, device)),
		Parameter(NDArrayView::RandomUniform<float>({ hiddenLayersDim, hiddenLayersDim }, -0.5, 0.5, 1, device)),
		Parameter(NDArrayView::RandomUniform<float>({ hiddenLayersDim, hiddenLayersDim }, -0.5, 0.5, 1, device)),
		Parameter(NDArrayView::RandomUniform<float>({ hiddenLayersDim, hiddenLayersDim }, -0.5, 0.5, 1, device)),
		Parameter(NDArrayView::RandomUniform<float>({ hiddenLayersDim, hiddenLayersDim }, -0.5, 0.5, 1, device)),
		Parameter(NDArrayView::RandomUniform<float>({ hiddenLayersDim, hiddenLayersDim }, -0.5, 0.5, 1, device)),
		Parameter(NDArrayView::RandomUniform<float>({ hiddenLayersDim, hiddenLayersDim }, -0.5, 0.5, 1, device))
	};
	Parameter hiddenLayerPlusParam[numHiddenLayers - 1] = {
		Parameter({ hiddenLayersDim }, 0.0f, device),
		Parameter({ hiddenLayersDim }, 0.0f, device),
		Parameter({ hiddenLayersDim }, 0.0f, device),
		Parameter({ hiddenLayersDim }, 0.0f, device),
		Parameter({ hiddenLayersDim }, 0.0f, device),
		Parameter({ hiddenLayersDim }, 0.0f, device),
		Parameter({ hiddenLayersDim }, 0.0f, device),
		Parameter({ hiddenLayersDim }, 0.0f, device),
		Parameter({ hiddenLayersDim }, 0.0f, device),
	};
	auto outputTimesParam = Parameter(NDArrayView::RandomUniform<float>({ outputClasses, hiddenLayersDim }, -0.5, 0.5, 1, device));

	_model = FullyConnectedFeedForwardClassifierNetWithSharedParameters(_inputs,
																		numHiddenLayers,
																		inputTimesParam,
																		inputPlusParam,
																		hiddenLayerTimesParam,
																		hiddenLayerPlusParam,
																		outputTimesParam,
																		std::bind(Sigmoid, std::placeholders::_1, L""));
}

inline FunctionPtr ModelTrainer::FullyConnectedFeedForwardClassifierNetWithSharedParameters(Variable input,
	size_t numHiddenLayers,
	const Parameter& inputTimesParam,
	const Parameter& inputPlusParam,
	const Parameter hiddenLayerTimesParam[],
	const Parameter hiddenLayerPlusParam[],
	const Parameter& outputTimesParam,
	const std::function<FunctionPtr(const FunctionPtr&)>& nonLinearity)
{
	assert(numHiddenLayers >= 1);
	auto classifierRoot = FullyConnectedDNNLayerWithSharedParameters(input, inputTimesParam, inputPlusParam, nonLinearity);

	for (size_t i = 1; i < numHiddenLayers; ++i)
	{
		classifierRoot = FullyConnectedDNNLayerWithSharedParameters(classifierRoot, hiddenLayerTimesParam[i - 1], hiddenLayerPlusParam[i - 1], nonLinearity);
	}

	// Todo: assume that outputTimesParam has matched output dim and hiddenLayerDim
	classifierRoot = Times(outputTimesParam, classifierRoot);
	return classifierRoot;
}

// Taken from the CNTK Examples
inline FunctionPtr ModelTrainer::FullyConnectedDNNLayerWithSharedParameters(Variable input,
	const Parameter& timesParam,
	const Parameter& plusParam,
	const std::function<FunctionPtr(const FunctionPtr&)>& nonLinearity)
{
	assert(input.Shape().Rank() == 1);

	// Todo: assume that timesParam has matched outputDim and inputDim 
	auto timesFunction = Times(timesParam, input);

	// Todo: assume that timesParam has matched outputDim 
	auto plusFunction = Plus(plusParam, timesFunction);

	return nonLinearity(plusFunction);
}

inline FunctionPtr ModelTrainer::SetupFullyConnectedLinearLayer(Variable input, size_t outputDim, const DeviceDescriptor& device, const std::wstring& outputName)
{
	assert(input.Shape().Rank() == 1);
	size_t inputDim = input.Shape()[0];

	auto timesParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim, inputDim }, -0.05, 0.05, 1, device));
	auto timesFunction = CNTK::Times(timesParam, input);

	auto plusParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim }, -0.05, 0.05, 1, device));
	return CNTK::Plus(plusParam, timesFunction, outputName);
}

inline FunctionPtr ModelTrainer::SetupFullyConnectedDNNLayer(Variable input, size_t outputDim, const DeviceDescriptor& device, const std::function<FunctionPtr(const FunctionPtr&)>& nonLinearity)
{
	return nonLinearity(SetupFullyConnectedLinearLayer(input, outputDim, device));
}

CNTK::TrainerPtr ModelTrainer::CreateTrainerForModel()
{
	// Create a trainer
	auto lossFunction = CrossEntropyWithSoftmax(_model->Output(), _labels, L"Loss Function");
	auto evalFunction = ClassificationError(_model->Output(), _labels, L"Classification Error");
	auto learningFunction = SGDLearner(_model->Parameters(), LearningRateSchedule(0.125, LearningRateSchedule::UnitType::Minibatch));
	return CreateTrainer(_model, lossFunction, evalFunction, { learningFunction });
}

void ModelTrainer::Train()
{
	vector<vector<float>> trainingData;
	TrainingData *loader = new TrainingData();
	loader->LoadFromFile(_dataFile, trainingData);

	CreateModel();
	auto trainer = CreateTrainerForModel();

	int minibatchSize = GAME_LENGTH;
	int numMinibatches = trainingData.size() / minibatchSize;

	int trainingPosition = 0;
	for (int i = 0; i < numMinibatches; i++)
	{
		// Carve off a batch of samples and create feature data
		vector<vector<float>> inputData;
		vector<vector<float>> labelData;
		vector<float> previousMove = loader->EncodeDefaultFeature();
		for (int j = trainingPosition; j < trainingPosition + GAME_LENGTH; j++)
		{
			inputData.push_back(previousMove);
			
			// Get the subset of the current training data that reflects the human move
			auto humanMove = vector<float>(trainingData[j].begin(), trainingData[j].begin() + 3);
			labelData.push_back(humanMove);

			// Shift the previous move state for encoding the next move
			previousMove.insert(previousMove.begin(), trainingData[j].begin(), trainingData[j].end());
			previousMove.resize(LOOKBACK_MOVES * 7);
		}

		// This is a simpler way to do value creation
		ValuePtr inputValues = Value::Create<float>(_inputs.Shape(), inputData, DeviceDescriptor::DefaultDevice(), true);
		ValuePtr labelValues = Value::Create<float>(_labels.Shape(), labelData, DeviceDescriptor::DefaultDevice(), true);

		std::unordered_map<Variable, ValuePtr> arguments = { { _inputs, inputValues },{ _labels, labelValues } };
		trainer->TrainMinibatch(arguments, DeviceDescriptor::DefaultDevice());
		
		// TODO: Output some progress data....
		cout << "Minibatch: " << i << ", Loss: " << trainer->PreviousMinibatchLossAverage() << ", Error: " << trainer->PreviousMinibatchEvaluationAverage() * 100 << "%\n";
		trainingPosition += minibatchSize;
	}

	trainer->Model()->SaveModel(_modelFile);
	wcout << L"New model saved to " << _modelFile;
}
