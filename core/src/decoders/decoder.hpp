#pragma once

#include <Eigen/Dense>

namespace SNN {
	namespace Decoder {
		template<typename T>
		class IDecoder {
			static_assert(std::is_same<double, T>::value || std::is_same<float, T>::value, "type must be float or double");
		public:
			virtual Eigen::Matrix<T, Eigen::Dynamic, 1> decode(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& input) = 0;
			virtual ~IDecoder() = 0;
		};

		template<class T>
		inline IDecoder<T>::~IDecoder()
		{
		}
	}
}