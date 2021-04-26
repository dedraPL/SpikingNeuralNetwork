#pragma once

#include <vector>

#include "decoder.hpp"

namespace SNN {
	namespace Decoder {
		template<typename T>
		class Transparent : public IDecoder<T> {
		public:
			Eigen::Matrix<T, Eigen::Dynamic, 1> decode(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& input) {
				return input;
			}
		};
	}
}