// TrainingData.cpp : Load and transform the training data to CNTK variables and values.
//

#include "stdafx.h"
#include "TrainingData.h"
#include "ModelTrainer.h"

using namespace CntkTraining;
using namespace std;

#include <fstream>
#include <iostream>
#include <string>

TrainingData::TrainingData()
{
}

vector<float> TrainingData::EncodeLabel(string move)
{
	float rock = (move == "R" ? 1.0f : 0.0f);
	float paper = (move == "P" ? 1.0f : 0.0f);
	float scissors = (move == "S" ? 1.0f : 0.0f);
	return { rock, paper, scissors };
}

vector<float> TrainingData::EncodeFeature(string hm, string cm, string wld)
{
	auto hFeature = EncodeLabel(hm);
	auto cFeature = EncodeLabel(cm);
	hFeature.insert(end(hFeature), begin(cFeature), end(cFeature));
	hFeature.push_back(atof(wld.data()));

	return hFeature;
}

vector<float> TrainingData::EncodeDefaultFeature()
{
	vector<float> featureData;

	for (int i = 0; i < LOOKBACK_MOVES; i++)
	{
		auto feature = EncodeFeature("", "", "1");
		featureData.insert(end(featureData), begin(feature), end(feature));
	}

	return featureData;
}

void TrainingData::LoadFromFile(const std::wstring& modelFile, vector<vector<float>>& data)
{	
	int moveNumber = 0;

	// Read in the data from the CSV file
	ifstream infile(modelFile);
	if (infile)
	{
		while (infile.good())
		{
			// Read the row, one entry at a time
			vector<string> row;
			string rowData;
			getline(infile, rowData);
			istringstream rowStream(rowData);

			for (int i = 0; i < 3; i++)
			{
				string entry;
				getline(rowStream, entry, ',');
				if (i == 2)
				{
					// Our game definition says a win or a draw is a win
					if (entry == "0") entry = "1";
					if (entry == "-1") entry = "0";
				}

				row.push_back(entry);
			}

			auto encodedFeature = EncodeFeature(row[0], row[1], row[2]);

			data.push_back(encodedFeature);
		}

		infile.close();
	}
}