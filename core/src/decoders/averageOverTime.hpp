#pragma once

#include <Eigen/Dense>

#include "decoder.hpp"

namespace SNN {
	namespace Decoder {
		template<typename T = double>
		class AverageOverTime : public IDecoder<T> {
		public:
			AverageOverTime(uint32_t period, uint32_t inputSize) : state(inputSize), history(inputSize, period)
			{
				this->period = period;
				state.setZero();
				history.setZero();
			}

			Eigen::Matrix<T, Eigen::Dynamic, 1> decode(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& input) override {
				Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp = (input.array() == 35);
				Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp2 = storeHistoryVector(tmp);
				state = state + (Eigen::VectorXi)tmp.cast<int>() - (Eigen::VectorXi)tmp2.cast<int>();
				return state.cast<T>() / period;
			}
			
		private:
			uint32_t period;
			Eigen::VectorXi state;
			Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> history;
			uint32_t historyOffset = 0;

			Eigen::Matrix<bool, Eigen::Dynamic, 1> storeHistoryVector(const Eigen::Ref<const Eigen::Matrix<bool, Eigen::Dynamic, 1>>& input)
			{
				Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp = history.col(historyOffset);
				history.col(historyOffset) = input;
				if (++historyOffset == period)
					historyOffset = 0;
				return tmp;
			}
		};
	}
}