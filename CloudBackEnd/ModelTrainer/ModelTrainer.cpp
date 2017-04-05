// ModelTrainer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ModelTrainer.h"
#include "TrainingData.h"
#include "Layers.h"

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
		if (!DoesFileExist(dataFilePath))
		{
			cout << "Cannot find the game input data file.";
			return 1;
		}

		ModelTrainer *trainer = new ModelTrainer(modelFilePath, dataFilePath);
		trainer->Train();
		delete(trainer);
		return 0;
	}
	catch (std::exception exc)
	{
		cout << exc.what();
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
	size_t cellDim = 25;
	size_t hiddenDim = 25;
	size_t embeddingDim = 50;
	size_t numOutputClasses = 3;
	CNTK::DeviceDescriptor device = DeviceDescriptor::DefaultDevice();

	_inputs = InputVariable({ inputDim }, CNTK::DataType::Float, L"Feature Vector");
	_labels = InputVariable({ numOutputClasses }, CNTK::DataType::Float, L"Labels");

	_model = LSTMSequenceClassifierNet(_inputs, numOutputClasses, hiddenDim, LOOKBACK_MOVES, cellDim, device);
}

// LSTM Network - Derived from the Python NN Layers

inline FunctionPtr ModelTrainer::LSTMSequenceClassifierNet(Variable input, size_t outputClasses, size_t embeddingDim, size_t lstmDim, size_t cellDim, const DeviceDescriptor& device)
{
	// Ignore the embedding layer for now.
	auto lstmFunction = Layers::LSTM(input, outputClasses, embeddingDim, cellDim, lstmDim, device);
	//auto thoughtVector = CNTK::Sequence::Last(lstmFunction);
	//return Layers::UniformLinear(CNTK::Dropout(thoughtVector, 0.2f), outputClasses, 0.0f, device);
	return lstmFunction;

	//auto embeddingFunction = Layers::Embedding(input, embeddingDim, device);
	//auto lstmFunction = LSTMPComponentWithSelfStabilization(embeddingFunction->Output(), lstmDim, cellDim, device)[0];
	//auto lstmFunction = LSTMPComponentWithSelfStabilization(input, lstmDim, cellDim, device)[0];
	//auto thoughtVector = CNTK::Sequence::Last(lstmFunction);
	//return Layers::UniformLinear(thoughtVector, outputClasses, 0.0f, device);
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
