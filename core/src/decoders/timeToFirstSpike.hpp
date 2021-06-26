#pragma once

#include <Eigen/Dense>

#include "decoder.hpp"

#include <iostream>

namespace SNN {
	namespace Decoder {
		template<typename T = double>
		class TimeToFirstSpike : public IDecoder<T> {
		public:
			TimeToFirstSpike(uint32_t maxPeriod, uint32_t inputSize) : history(inputSize, 1), maxPeriod(maxPeriod)
			{
				history.setConstant(-1);
			}

			Eigen::Matrix<T, Eigen::Dynamic, 1> decode(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& input) override {
				Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp = (input.array() == 35);

				if (currPeriod == maxPeriod)
				{
					history.setConstant(-1);
					currPeriod = 0;
				}

				for (uint32_t i = 0; i < history.rows(); i++)
				{
					if (tmp[i] == 1 && history[i] == -1)
					{
						history[i] = currPeriod;
					}
				}

				currPeriod++;
				
				return history;
			}

		private:
			Eigen::Matrix<T, Eigen::Dynamic, 1> history;
			uint32_t currPeriod = 0;
			uint32_t maxPeriod;
		};
	}
}
