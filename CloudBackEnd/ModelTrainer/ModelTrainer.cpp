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

int TrainModel(const wchar_t* dataFilePath, const wchar_t* modelFilePath);

int wmain(int argc, wchar_t**argv)
{
	if (argc != 3)
	{
		cout << "Syntax: ModelTrainer.exe <gamefile.csv> <output-modelfile.model>\n";
		return 1;
	}

	const wchar_t* dataFilePath = argv[1];
	const wchar_t* modelFilePath = argv[2];

	return TrainModel(dataFilePath, modelFilePath);
}
#endif

inline bool DoesFileExist(const wchar_t* fileName) {
	ifstream f(fileName);
	return f.good();
}

#ifdef MODELTRAINERLIB
extern "C" __declspec(dllexport) 
#endif
int TrainModel(const wchar_t* dataFilePath, const wchar_t* modelFilePath)
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

		ModelTrainer *trainer = new ModelTrainer(dataFilePath, modelFilePath);
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

ModelTrainer::ModelTrainer(const wstring& dataFile, const wstring& modelFile)
{
	_dataFile = dataFile;
	_modelFile = modelFile;
}

void ModelTrainer::CreateModel()
{
	// Define the model from scratch for training
	size_t inputDim = NUMBER_OF_FEATURES;
	size_t numOutputClasses = 3;
	size_t cellDim = inputDim;  // This follows the Python Library default
	size_t hiddenLayersDim = numOutputClasses; // This follows the Python Library default
	CNTK::DeviceDescriptor device = DeviceDescriptor::UseDefaultDevice();

	_inputs = InputVariable({ inputDim }, false, CNTK::DataType::Float, L"Feature Vector");

#ifdef LSTM_NETWORK
	_model = LSTMSequenceClassifierNet(_inputs, numOutputClasses, hiddenLayersDim, cellDim, lookbackMoves, device);
#else
	_model = FeedForwardClassifier(_inputs, numOutputClasses, hiddenLayersDim, device);
#endif

	_labels = InputVariable({ numOutputClasses }, false, CNTK::DataType::Float, L"Labels", { Axis::DefaultBatchAxis() });
}

FunctionPtr ModelTrainer::FeedForwardClassifier(Variable input, size_t outputClasses, size_t hiddenLayersDim, const DeviceDescriptor& device)
{
	// init = glorotuniform, activation = sigmoid
	auto classifierRoot = input;
	for (int i = 0; i < HIDDEN_LAYERS; i++)
	{
		classifierRoot = Layers::Dense(classifierRoot, hiddenLayersDim, CNTK::GlorotUniformInitializer(),
			std::bind(Sigmoid, std::placeholders::_1, L""), true, 0.0f, device);
	}

	return Layers::Dense(classifierRoot, outputClasses, device);
}

FunctionPtr ModelTrainer::LSTMSequenceClassifierNet(Variable input, size_t outputClasses, size_t hiddenDim, size_t cellDim, size_t lstmCells, const DeviceDescriptor& device)
{
	auto lstmFunction = Layers::LSTM(input, outputClasses, hiddenDim, cellDim, lstmCells);
	auto thoughtVector = CNTK::Sequence::Last(lstmFunction);
	auto dropoutFunction = CNTK::Dropout(thoughtVector, 0.2f);
	return Layers::Dense(dropoutFunction, outputClasses, CNTK::GlorotUniformInitializer(),
		std::bind(Sigmoid, std::placeholders::_1, L""), true, 0.0f, device);
}

CNTK::TrainerPtr ModelTrainer::CreateTrainerForModel()
{
	// Create a trainer
#ifdef LSTM_NETWORK
	auto momentum = -1.0 * (GAME_LENGTH / std::log(0.9));
	auto lossFunction = SquaredError(_model->Output(), _labels, L"Loss Function");
	auto evalFunction = ClassificationError(_model->Output(), _labels, L"Classification Error"); // SquaredError(_model->Output(), _labels, L"Classification Error");
	auto learningFunction = FSAdaGradLearner(_model->Parameters(), LearningRatePerSampleSchedule(0.01), MomentumAsTimeConstantSchedule(momentum), true);
#else
	auto lossFunction = CrossEntropyWithSoftmax(_model->Output(), _labels, L"Loss Function");
	auto evalFunction = ClassificationError(_model->Output(), _labels, L"Classification Error");
	auto learningFunction = SGDLearner(_model->Parameters(), LearningRatePerMinibatchSchedule(0.125));
#endif
	return CreateTrainer(_model, lossFunction, evalFunction, { learningFunction });
}

void ModelTrainer::Train()
{
	vector<vector<float>> trainingData;
	TrainingData *loader = new TrainingData();
	loader->LoadFromFile(_dataFile, trainingData);

	// The amount of training data necessary is proportional to the number of lookback moves
	// Given that any one game has 9 possible combinations and all games are independent events,
	// the number of possibilites is 9^(lookback_moves) = [ 1, 9, 81, 729, 6561, 59049, 531441, .... ]
	// The best possible strategy will be to increase the amount of lookback as we get more training data.
	for (lookbackMoves = 0; trainingData.size() > std::pow(9, lookbackMoves + 1) * TRAINING_FACTOR && lookbackMoves < GAME_LENGTH; lookbackMoves++);

	CreateModel();
	auto trainer = CreateTrainerForModel();

	int minibatchSize = GAME_LENGTH;
	int numMinibatches = trainingData.size() / minibatchSize;
	int validateBatches = int(numMinibatches * 0.2);
	int testBatches = int(numMinibatches * 0.1);
#if (TRAIN_ONLY == 1)
	int trainBatches = numMinibatches;
#else
	int trainBatches = numMinibatches - validateBatches - testBatches;
#endif
	int outputFrequency = 10;
	double averageTrainingError = 0.0;
	int predictedWins = 0, predictedGames = 0;
	auto device = DeviceDescriptor::UseDefaultDevice();

	int trainingPosition = 0;
	for (int i = 0; i < numMinibatches; i++)
	{
		vector<vector<float>> inputData;
		vector<vector<float>> labelData;
		EncodeBatch(loader, trainingPosition, trainingData, inputData, labelData);

		// Do the training
		if (i < trainBatches)
		{
			// This is a simpler way to do value creation
			ValuePtr inputValues = Value::Create<float>(_inputs.Shape(), inputData, device, true);
			ValuePtr labelValues = Value::Create<float>(_labels.Shape(), labelData, device, true);
			std::unordered_map<Variable, ValuePtr> arguments = { { _inputs, inputValues },{ _labels, labelValues } };

			trainer->TrainMinibatch(arguments, device);

			// TODO: Output some progress data....
			if (i % outputFrequency == 0)
			{
				cout << "Minibatch: " << i << ", Loss: " << trainer->PreviousMinibatchLossAverage() << ", Error: " << trainer->PreviousMinibatchEvaluationAverage() * 100 << "%\n";
			}
		}
		else if (i < trainBatches + testBatches)
		{
			// This is a simpler way to do value creation
			ValuePtr inputValues = Value::Create<float>(_inputs.Shape(), inputData, device, true);
			ValuePtr labelValues = Value::Create<float>(_labels.Shape(), labelData, device, true);
			std::unordered_map<Variable, ValuePtr> arguments = { { _inputs, inputValues },{ _labels, labelValues } };

			averageTrainingError += trainer->TestMinibatch(arguments, device);
		}
		else
		{
			auto model = trainer->Model();

			// Iterate through the games in the batch
			for (int k = 0; k < GAME_LENGTH; k++)
			{
				ValuePtr inputValues = Value::Create<float>(_inputs.Shape(), { inputData[k] }, device, true);
				unordered_map<Variable, ValuePtr> inputs = { { _inputs, inputValues } };

				auto labels = model->Output();
				unordered_map<Variable, ValuePtr> outputs = { { labels, nullptr } };

				// TODO: The Python model uses softmax on the output
				model->Evaluate(inputs, outputs, device);
				
				vector<vector<float>> outputBuffer;
				auto outputVal = outputs[model->Output()];
				outputVal->CopyVariableValueTo(model->Output(), outputBuffer);

				auto probabilities = outputBuffer[0];
				auto indexOfMax = distance(probabilities.begin(), max_element(probabilities.begin(), probabilities.end()));

				if (labelData[k][indexOfMax] == 1.0)
				{
					predictedWins++;
				}

				predictedGames++;
			}
		}

		trainingPosition += minibatchSize;
	}

#if (TRAIN_ONLY != 1)
	cout << "Mean squared testing error: " << (averageTrainingError / testBatches) * 100.0 << "%\n";
	cout << "Evaluation predictions: " << predictedWins << " / " << predictedGames << " (" << ((double)predictedWins / (double)predictedGames) * 100.0 << "%)";
#endif

	trainer->Model()->SaveModel(_modelFile);
	wcout << L"New model saved to " << _modelFile;
}

void ModelTrainer::EncodeBatch(TrainingData *loader, int trainingPosition, vector<vector<float>> trainingData, 
	vector<vector<float>>& features, vector<vector<float>>& labels)
{
	// Carve off a batch of samples and create feature data
	vector<float> previousMove = loader->EncodeDefaultFeature(1);
	for (int j = trainingPosition; j < trainingPosition + GAME_LENGTH; j++)
	{
		features.push_back(previousMove);

		// Get the subset of the current training data that reflects the human move
		auto humanMove = vector<float>(trainingData[j].begin(), trainingData[j].begin() + 3);
		labels.push_back(humanMove);

		// Shift the previous move state for encoding the next move
		// We append the next move, then shift left and truncate....
		previousMove.insert(previousMove.end(), trainingData[j].begin(), trainingData[j].end());

		if ((j + 1) % GAME_LENGTH >= lookbackMoves)
		{
			std::rotate(previousMove.begin(), previousMove.begin() + 7, previousMove.end());
			previousMove.resize(lookbackMoves * 7);
		}
	}
}