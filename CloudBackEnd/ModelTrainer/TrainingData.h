#pragma once

#include <string>
#include "CNTKLibrary.h"

namespace CntkTraining
{
	class TrainingData
	{
	public:
		TrainingData();

		void LoadFromFile(const std::wstring& modelFile, std::vector<std::vector<float>>& data);
		std::vector<float> EncodeDefaultFeature();
		std::vector<float> EncodeLabel(std::string move);
		std::vector<float> EncodeFeature(std::string hm, std::string cm, std::string wld);
	};
}
