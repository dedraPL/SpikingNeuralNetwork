#pragma once

#include <Eigen/Dense>

#include "decoder.hpp"

namespace SNN {
	namespace Decoder {
		template<typename T = double>
		class Binary : public IDecoder<T> {
		public:
			Eigen::Matrix<T, Eigen::Dynamic, 1> decode(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& input) override {

				return (input.array() == 35).cast<T>();
			}
		};
	}
}
