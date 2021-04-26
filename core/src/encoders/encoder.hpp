#pragma once

#include <Eigen/Dense>

namespace SNN {
	namespace Encoder {
		template<typename T>
		class IEncoder {
			static_assert(std::is_same<double, T>::value || std::is_same<float, T>::value, "type must be float or double");
		public:
			virtual Eigen::Matrix<T, Eigen::Dynamic, 1> encode(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& input) = 0;
			virtual ~IEncoder() = 0;
		};

		template<class T>
		inline IEncoder<T>::~IEncoder()
		{
		}
	}
}