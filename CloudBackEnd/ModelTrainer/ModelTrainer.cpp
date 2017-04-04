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
	_labels = InputVariable({ numOutputClasses }, CNTK::DataType::Float, L"Labels", { Axis::DefaultBatchAxis() });

	_model = LSTMSequenceClassifierNet(_inputs, numOutputClasses, embeddingDim, hiddenDim, cellDim, device);
}

// LSTM Network - Derived from the Python NN Layers

inline FunctionPtr ModelTrainer::LSTMSequenceClassifierNet(Variable input, size_t outputClasses, size_t embeddingDim, size_t lstmDim, size_t cellDim, const DeviceDescriptor& device)
{
	// Ignore the embedding layer for now.
	//auto embeddingFunction = Embedding(input, embeddingDim, device);
	//auto lstmFunction = LSTMPComponentWithSelfStabilization(embeddingFunction->Output(), lstmDim, cellDim, device)[0];
	auto lstmFunction = LSTMPComponentWithSelfStabilization(input, lstmDim, cellDim, device)[0];
	auto thoughtVector = CNTK::Sequence::Last(lstmFunction);
	return LinearLayer(thoughtVector, outputClasses, device);
}

inline FunctionPtr ModelTrainer::Embedding(Variable input, size_t embeddingDim, const DeviceDescriptor& device)
{
	assert(input.Shape().Rank() == 1);
	size_t inputDim = input.Shape()[0];

	auto embeddingParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ inputDim, embeddingDim }, -0.05, 0.05, 1, device));

	return CNTK::Times(input, embeddingParam);
}

inline std::vector<FunctionPtr> ModelTrainer::LSTMPComponentWithSelfStabilization(Variable input, size_t outputDim, size_t cellDim, const DeviceDescriptor& device)
{
	auto dh = CNTK::PlaceholderVariable({ outputDim }, input.DynamicAxes());
	auto dc = CNTK::PlaceholderVariable({ cellDim }, input.DynamicAxes());

	auto LSTMCell = LSTMPCellWithSelfStabilization(input, dh, dc, device);
	auto actualDh = CNTK::PastValue(LSTMCell[0]);
	auto actualDc = CNTK::PastValue(LSTMCell[1]);

	// Form the recurrence loop by replacing the dh and dc placeholders with
	// the actualDh and actualDc
	std::unordered_map<Variable, Variable> placeholders = { { dh, actualDh->Output() },{ dc, actualDc->Output() } };
	LSTMCell[0]->ReplacePlaceholders(placeholders);

	std::vector<FunctionPtr> returnVector = { LSTMCell[0], LSTMCell[1] };
	return returnVector;
}

std::vector<FunctionPtr> ModelTrainer::LSTMPCellWithSelfStabilization(Variable input, Variable prevOutput, Variable prevCellState, const DeviceDescriptor& device)
{
	size_t inputDim = input.Shape()[0];
	size_t outputDim = prevOutput.Shape()[0];
	size_t cellDim = prevCellState.Shape()[0];

	auto Wxo = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ inputDim, cellDim }, -0.05, 0.05, 1, device));
	auto Wxi = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ inputDim, cellDim }, -0.05, 0.05, 1, device));
	auto Wxf = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ inputDim, cellDim }, -0.05, 0.05, 1, device));
	auto Wxc = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ inputDim, cellDim }, -0.05, 0.05, 1, device));

	auto Bo = CNTK::Parameter({ cellDim }, 0.0f, device);
	auto Bc = CNTK::Parameter({ cellDim }, 0.0f, device);
	auto Bi = CNTK::Parameter({ cellDim }, 0.0f, device);
	auto Bf = CNTK::Parameter({ cellDim }, 0.0f, device);

	auto Whi = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim, cellDim }, -0.05, 0.05, 1, device));
	auto Wci = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ cellDim }, -0.05, 0.05, 1, device));

	auto Whf = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim, cellDim }, -0.05, 0.05, 1, device));
	auto Wcf = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ cellDim }, -0.05, 0.05, 1, device));

	auto Who = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim, cellDim }, -0.05, 0.05, 1, device));
	auto Wco = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ cellDim }, -0.05, 0.05, 1, device));

	auto Whc = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim, cellDim }, -0.05, 0.05, 1, device));

	auto Wmr = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ cellDim, outputDim }, -0.05, 0.05, 1, device));

	// Stabilization by routing input through an extra scalar parameter
	auto expsWxo = CNTK::Exp(CNTK::Parameter({ 1 }, 0.0f, device));
	auto expsWxi = CNTK::Exp(CNTK::Parameter({ 1 }, 0.0f, device));
	auto expsWxf = CNTK::Exp(CNTK::Parameter({ 1 }, 0.0f, device));
	auto expsWxc = CNTK::Exp(CNTK::Parameter({ 1 }, 0.0f, device));

	auto expsWhi = CNTK::Exp(CNTK::Parameter({ 1 }, 0.0f, device));
	auto expsWci = CNTK::Exp(CNTK::Parameter({ 1 }, 0.0f, device));

	auto expsWhf = CNTK::Exp(CNTK::Parameter({ 1 }, 0.0f, device));
	auto expsWcf = CNTK::Exp(CNTK::Parameter({ 1 }, 0.0f, device));
	auto expsWho = CNTK::Exp(CNTK::Parameter({ 1 }, 0.0f, device));
	auto expsWco = CNTK::Exp(CNTK::Parameter({ 1 }, 0.0f, device));
	auto expsWhc = CNTK::Exp(CNTK::Parameter({ 1 }, 0.0f, device));

	auto expsWmr = CNTK::Exp(CNTK::Parameter({ 1 }, 0.0f, device));

	auto Wxix = CNTK::Times(CNTK::ElementTimes(expsWxi, input), Wxi);
	auto Whidh = CNTK::Times(CNTK::ElementTimes(expsWhi, prevOutput), Whi);
	auto Wcidc = CNTK::ElementTimes(Wci, CNTK::ElementTimes(expsWci, prevCellState));

	auto it = CNTK::Sigmoid(Wxix + Bi + Whidh + Wcidc);
	auto Wxcx = CNTK::Times(CNTK::ElementTimes(expsWxc, input), Wxc);
	auto Whcdh = CNTK::Times(CNTK::ElementTimes(expsWhc, prevOutput), Whc);
	auto bit = CNTK::ElementTimes(it, CNTK::Tanh(Wxcx + Whcdh + Bc)); 
	auto Wxfx = CNTK::Times(CNTK::ElementTimes(expsWxf, input), Wxf);
	auto Whfdh = CNTK::Times(CNTK::ElementTimes(expsWhf, prevOutput), Whf);
	auto Wcfdc = CNTK::ElementTimes(Wcf, CNTK::ElementTimes(expsWcf, prevCellState));

	auto ft = CNTK::Sigmoid(Wxfx + Bf + Whfdh + Wcfdc);
	auto bft = CNTK::ElementTimes(ft, prevCellState);

	auto ct = bft + bit;

	auto Wxox = CNTK::Times(CNTK::ElementTimes(expsWxo, input), Wxo);
	auto Whodh = CNTK::Times(CNTK::ElementTimes(expsWho, prevOutput), Who);
	auto Wcoct = CNTK::ElementTimes(Wco, CNTK::ElementTimes(expsWco, ct));

	auto ot = CNTK::Sigmoid(Wxox + Bo + Whodh + Wcoct);

	auto mt = CNTK::ElementTimes(ot, CNTK::Tanh(ct));
	std::vector<FunctionPtr> returnVector = { CNTK::Times(CNTK::ElementTimes(expsWmr, mt), Wmr), ct };
	return returnVector;
}

inline FunctionPtr ModelTrainer::SelectLast(Variable operand)
{
	return CNTK::Slice(operand, CNTK::Axis::DefaultDynamicAxis(), -1, 0);
}

inline FunctionPtr ModelTrainer::LinearLayer(Variable input, size_t outputDim, const DeviceDescriptor& device)
{
	assert(input.Shape().Rank() == 1);
	size_t inputDim = input.Shape()[0];

	auto timesParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim, inputDim }, -0.05, 0.05, 1, device));
	auto biasParam = CNTK::Parameter({ outputDim }, 0, device);
	auto timesFunction = CNTK::Times(input, timesParam);
	return CNTK::Plus(biasParam, timesFunction);
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
