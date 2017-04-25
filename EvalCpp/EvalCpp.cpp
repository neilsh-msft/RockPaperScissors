// EvalCpp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CNTKLibrary.h"

#include <string>
#include <deque>

using namespace std;
using namespace CNTK;

unordered_map<string, vector<float>> unpack = 
{
	{ "R", { 1.0f, 0.0f, 0.0f } },
	{ "P", { 0.0f, 1.0f, 0.0f } },
	{ "S", { 0.0f, 0.0f, 1.0f } }
};

unordered_map<string, int> winLossStates = 
{
	{ "RR", 0 },
	{ "RP", 1 },
	{ "RS", -1 },
	{ "PR", -1 },
	{ "PP", 0 },
	{ "PS", 1 },
	{ "SR", 1 },
	{ "SP", -1 },
	{ "SS", 0 }
};

inline bool DoesFileExist(const wchar_t* fileName) {
	ifstream f(fileName);
	return f.good();
}

string GetInputBytes()
{
	string inputBytes;
	getline(cin, inputBytes);

	return inputBytes;
}

//
// requiredSize is the required size in pairs of move. Example: "R P R S" is 2 pairs
// The function will take the last requiredSize pairs from the input
// and pad it up to requiredSize if the input is not large enough
//
vector<float> PrepareEvalData(string input, int requiredSize)
{
	vector<float> evalData;

	istringstream iss(input);
	deque<string> stringArray{ istream_iterator<string>{iss}, istream_iterator<string>{} };
	size_t initialSize = stringArray.size();

	// Pop off extra elements if input is too large
	while (stringArray.size() > requiredSize * 2)
	{
		stringArray.pop_front();
		stringArray.pop_front();
	}

	// Now populate the data
	while (stringArray.size() > 0)
	{
		auto humanMove = stringArray.front(); stringArray.pop_front();
		auto computerMove = stringArray.front(); stringArray.pop_front();

		auto pushFunc = [&](const float& m) { evalData.push_back(m); };

		for_each(unpack[humanMove].begin(), unpack[humanMove].end(), pushFunc);
		for_each(unpack[computerMove].begin(), unpack[computerMove].end(), pushFunc);

		int winLoss = winLossStates[humanMove + computerMove];
		if (winLoss == 0) winLoss = 1;
		if (winLoss == -1) winLoss = 0;
		evalData.push_back((float)winLoss);
	}

	// Deal with padding if the input size was not large enough
	for (int i = 0; i < (int)(requiredSize - initialSize / 2); ++i)
	{
		for (int j = 0; j < 6; ++j) evalData.push_back(0);
		evalData.push_back(1);
	}

	return evalData;
}

std::wstring InvokeEval(const string& input, const wstring& modelFile)
{
	CNTK::DeviceDescriptor device = DeviceDescriptor::UseDefaultDevice();

	// Load the trained model from a file
	auto model = Function::LoadModel(modelFile);

	// Create the inputs and outputs
	auto inputVariable = model->Arguments()[0];
	auto shape = inputVariable.Shape();
	int depth = shape[0];

	// Prepare input data using the depth from the input file
	auto inputData = PrepareEvalData(input, depth);
	assert(evalData.size() == 7 * depth);

	ValuePtr inputValues = Value::Create<float>(inputVariable.Shape(), { inputData }, device, true);
	unordered_map<Variable, ValuePtr> inputs = { { inputVariable, inputValues } };

	auto labels = model->Output();
	unordered_map<Variable, ValuePtr> outputs = { { labels, nullptr } };

	model->Evaluate(inputs, outputs, device);

	// Get evaluate result as dense output
	vector<vector<float>> outputBuffer;
	auto outputVal = outputs[model->Output()];
	outputVal->CopyVariableValueTo(model->Output(), outputBuffer);

	auto probabilities = outputBuffer[0];
	auto indexOfMax = distance(probabilities.begin(), max_element(probabilities.begin(), probabilities.end()));

	vector<wstring> computerMoves = { L"P", L"S", L"R" };
	return computerMoves[indexOfMax];
}

void WriteToStdOut(const wstring& str)
{
	HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);

	DWORD byteWrittenCount = 0;
	BOOL success = WriteFile(h, str.c_str(), str.size() * sizeof(wchar_t), &byteWrittenCount, NULL);
	CloseHandle(h);
}

int wmain(int argc, wchar_t**argv)
{
//	WriteToStdOut(L"X");
//	return 0;

	if (argc != 2)
	{
		cout << "Syntax: EvalCpp.exe <modelfile.model>\n";
		return 1;
	}

	const wchar_t* modelFilePath = argv[1];

	if (!DoesFileExist(modelFilePath))
	{
		cout << "Cannot find the game input data file.";
		return 1;
	}

#if true// real input
	string input = GetInputBytes();
#else // useful for debugging
	//var input = "R R R R R R R R R R";
	//var input = "S S S S S S S S S S";
	// string input = "S S P P P P P P P P";
	string input = "R P R S P P S P R P";
#endif

	try
	{
		auto output = InvokeEval(input, modelFilePath);
		WriteToStdOut(output);
		return 0;
	}
	catch (std::exception exc)
	{
		std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		std::wstring wide = converter.from_bytes(exc.what());
		WriteToStdOut(wide);
		return 1;
	}
}