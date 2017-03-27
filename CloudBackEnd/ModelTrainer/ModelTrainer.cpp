// ModelTrainer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ModelTrainer.h"
#include "TrainingData.h"
#include <string>
#include <iostream>

using namespace std;
using namespace CntkTraining;
using namespace CNTK;

int main()
{
	// These are the two arguments that need to be passed on the command line
	wstring modelFilePath = L"C:\\Temp\\RPS\\rps.model";
	wstring dataFilePath = L"C:\\Temp\\RPS\\rps.csv";

	ModelTrainer *trainer = new ModelTrainer(modelFilePath, dataFilePath);
	trainer->Train();

	delete(trainer);
}

ModelTrainer::ModelTrainer(const wstring& modelFile, const wstring& dataFile)
{
	_modelFile = modelFile;
	_dataFile = dataFile;
}

void ModelTrainer::LoadModel()
{
	// Load the model definition from a file
	_model = Function::LoadModel(_modelFile);

	_inputs = _model->Arguments()[0];
	_labels = _model->Outputs()[0];

	// We could create the model from scratch, but its schema is already defined by the model file...
	// _inputs = InputVariable({ 7 * LOOKBACK_MOVES }, CNTK::DataType::Float, L"Feature Vector");
	// _labels = InputVariable({ 3 }, CNTK::DataType::Float, L"Labels");
	// ...
}

CNTK::TrainerPtr ModelTrainer::CreateTrainerForModel()
{
	// Create a trainer
	auto lossFunction = CrossEntropyWithSoftmax(_model->Output(), _labels, L"Loss Function");
	auto evalFunction = ClassificationError(_model->RootFunction(), _labels, L"Error");
	auto learningFunction = SGDLearner(_model->Parameters(), LearningRateSchedule(0.125, LearningRateSchedule::UnitType::Minibatch));
	return CreateTrainer(_model, lossFunction, evalFunction, { learningFunction });
}

void ModelTrainer::Train()
{
	vector<vector<float>> trainingData;
	TrainingData *loader = new TrainingData();
	loader->LoadFromFile(_dataFile, trainingData);

	LoadModel();
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
		cout << "Minibatch: " << trainer->PreviousMinibatchSampleCount() << ", Loss: " << trainer->PreviousMinibatchLossAverage() << ", Error: " << trainer->PreviousMinibatchEvaluationAverage();
		trainingPosition += minibatchSize;
	}

	// trainer->Model()->SaveModel(_modelFile);
}
