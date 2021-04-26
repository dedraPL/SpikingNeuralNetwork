#pragma once

#include <Eigen/Dense>

#include "encoder.hpp"

namespace SNN {
	namespace Encoder {
		template<typename T>
		class Transparent : public IEncoder<T>{
		public:
			Eigen::Matrix<T, Eigen::Dynamic, 1> encode(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& input) override {
				return input;
			}
		};
	}
}