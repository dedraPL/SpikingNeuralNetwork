#include "catch.hpp"

#include "decoders/averageOverTime.hpp"
#include "decoders/timeToFirstSpike.hpp"
#include "decoders/decoder.hpp"

#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace SNN {
	TEST_CASE("SNN.decoders.AverageOverTime", "[Decoders]") {
		Decoder::IDecoder<float>* decoder = new Decoder::AverageOverTime<float>(5, 5);

		std::vector<Eigen::Matrix<float, 5, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 5, 1> > > inputs;
		std::vector<Eigen::Matrix<float, 5, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 5, 1> > > outputs;

		for(uint32_t i = 0; i < 5; i++)
			inputs.push_back(Eigen::VectorXf(5));

		for (uint32_t i = 0; i < 10; i++)
			outputs.push_back(Eigen::VectorXf(5));

		inputs[0] << 34, 35, 34, 35, 00;
		inputs[1] << 00, 00, 00, 00, 00;
		inputs[2] << 35, 00, 34, 35, 00;
		inputs[3] << 00, 34, 00, 00, 34;
		inputs[4] << 00, 35, 00, 35, 00;

		outputs[0] << 0.0, 0.2, 0.0, 0.2, 0.0;
		outputs[1] << 0.0, 0.2, 0.0, 0.2, 0.0;
		outputs[2] << 0.2, 0.2, 0.0, 0.4, 0.0;
		outputs[3] << 0.2, 0.2, 0.0, 0.4, 0.0;
		outputs[4] << 0.2, 0.4, 0.0, 0.6, 0.0;
		outputs[5] << 0.2, 0.2, 0.0, 0.4, 0.0;
		outputs[6] << 0.2, 0.2, 0.0, 0.4, 0.0;
		outputs[7] << 0.0, 0.2, 0.0, 0.2, 0.0;
		outputs[8] << 0.0, 0.2, 0.0, 0.2, 0.0;
		outputs[9] << 0.0, 0.0, 0.0, 0.0, 0.0;

		for (uint32_t i = 0; i < 5; i++)
		{
			Eigen::VectorXf result = decoder->decode(inputs[i]);
			REQUIRE(result.isApprox(outputs[i]));
		}

		for (uint32_t i = 5; i < 10; i++)
		{
			Eigen::VectorXf result = decoder->decode(Eigen::Matrix<float, 5, 1>::Constant(0));
			REQUIRE(result.isApprox(outputs[i]));
		}
	}

	TEST_CASE("SNN.decoders.TimeToFirstSpike", "[Decoders]") {
		Decoder::IDecoder<float>* decoder = new Decoder::TimeToFirstSpike<float>(5, 5);

		std::vector<Eigen::Matrix<float, 5, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 5, 1> > > inputs;
		std::vector<Eigen::Matrix<float, 5, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 5, 1> > > outputs;

		for (uint32_t i = 0; i < 10; i++)
			inputs.push_back(Eigen::VectorXf(5));

		for (uint32_t i = 0; i < 10; i++)
			outputs.push_back(Eigen::VectorXf(5));

		inputs[0] << 34, 35, 34, 35, 00;
		inputs[1] << 00, 00, 00, 00, 00;
		inputs[2] << 35, 00, 34, 35, 00;
		inputs[3] << 00, 34, 00, 00, 34;
		inputs[4] << 00, 35, 00, 35, 00;

		inputs[5] << 00, 00, 00, 00, 00;
		inputs[6] << 34, 00, 35, 00, 34;
		inputs[7] << 00, 00, 00, 00, 35;
		inputs[8] << 35, 35, 35, 35, 35;
		inputs[9] << 35, 35, 00, 00, 00;

		outputs[0] << -1,  0, -1,  0, -1;
		outputs[1] << -1,  0, -1,  0, -1;
		outputs[2] <<  2,  0, -1,  0, -1;
		outputs[3] <<  2,  0, -1,  0, -1;
		outputs[4] <<  2,  0, -1,  0, -1;

		outputs[5] << -1, -1, -1, -1, -1;
		outputs[6] << -1, -1,  1, -1, -1;
		outputs[7] << -1, -1,  1, -1,  2;
		outputs[8] <<  3,  3,  1,  3,  2;
		outputs[9] <<  3,  3,  1,  3,  2;

		for (uint32_t i = 0; i < 10; i++)
		{
			Eigen::VectorXf result = decoder->decode(inputs[i]);
			REQUIRE(result.isApprox(outputs[i]));
		}
	}
}