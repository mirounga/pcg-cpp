// test-perf.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

#include "pcg_fill.hpp"

template<class R>
void test_float()
{
	size_t M = 100000041;

	std::vector<float> kx(M), vx(M), dx(M);

	R debug_rng(42u, 54u);

	float low = 0.0f, high = 1.0f;
	float range = high - low;

	auto start = std::chrono::high_resolution_clock::now();

	for (uint64_t i = 0; i < M; i++)
	{
		//dx[i] = static_cast<float>(pcg32_random_r(&debug_rng));
		dx[i] = ldexpf(static_cast<float>(debug_rng()), -32) *range + low;
	}

	auto end = std::chrono::high_resolution_clock::now();

	double elapsed = std::chrono::duration<double, std::micro>(end - start).count() / 1.0e6;

	printf("Sequential fill in %.3g seconds; %.3g items/sec.\n", elapsed, M / elapsed);

	R vector_rng(42u, 54u);


	start = std::chrono::high_resolution_clock::now();

	pcg_fill::fill(vector_rng, vx.data(), M, low, high);

	end = std::chrono::high_resolution_clock::now();

	elapsed = std::chrono::duration<double, std::micro>(end - start).count() / 1.0e6;

	printf("Vector fill in %.3g seconds; %.3g items/sec.\n", elapsed, M / elapsed);

	for (uint64_t i = 0; i < M; i++)
	{
		assert(vx[i] == dx[i]);
	}
}

template<class R>
void test_double()
{
	size_t M = 100000097;

	std::vector<double> kx(M), vx(M), dx(M);

	R debug_rng0(42u, 54u);
	R debug_rng1(47u, 66u);

	double low = 0.0f, high = 1.0f;
	double range = high - low;

	auto start = std::chrono::high_resolution_clock::now();

	for (uint64_t i = 0; i < M; i++)
	{
		uint64_t val = (static_cast<uint64_t>(debug_rng1()) << 32) | debug_rng0();
		dx[i] = ldexp(val, -64) *range + low;
	}

	auto end = std::chrono::high_resolution_clock::now();

	double elapsed = std::chrono::duration<double, std::micro>(end - start).count() / 1.0e6;

	printf("Sequential fill in %.3g seconds; %.3g items/sec.\n", elapsed, M / elapsed);

	R vector_rng0(42u, 54u);
	R vector_rng1(47u, 66u);

	start = std::chrono::high_resolution_clock::now();

	pcg_fill::fill(vector_rng0, vector_rng1, vx.data(), M, low, high);

	end = std::chrono::high_resolution_clock::now();

	elapsed = std::chrono::duration<double, std::micro>(end - start).count() / 1.0e6;

	printf("Vector fill in %.3g seconds; %.3g items/sec.\n", elapsed, M / elapsed);

	for (uint64_t i = 0; i < M; i++)
	{
		assert(vx[i] == dx[i]);
	}
}

int main()
{
    std::cout << "pcg32\nfloat:\n";
	test_float<pcg32>();

    std::cout << "double:\n";
	test_double<pcg32>();

    std::cout << "\nXSH RS\nfloat:\n";
	test_float<pcg_engines::setseq_xsh_rs_64_32>();

    std::cout << "double:\n";
	test_double<pcg_engines::setseq_xsh_rs_64_32>();

    std::cout << "\nXSH RR\nfloat:\n";
	test_float<pcg_engines::setseq_xsh_rr_64_32>();

    std::cout << "double:\n";
	test_double<pcg_engines::setseq_xsh_rr_64_32>();

    std::cout << "\nRXS M\nfloat:\n";
	test_float<pcg_engines::setseq_rxs_m_64_32>();

    std::cout << "double:\n";
	test_double<pcg_engines::setseq_rxs_m_64_32>();

    std::cout << "\nDXSM\nfloat:\n";
	test_float<pcg_engines::setseq_dxsm_64_32>();

    std::cout << "double:\n";
	test_double<pcg_engines::setseq_dxsm_64_32>();

    std::cout << "\nXSL RR\nfloat:\n";
	test_float<pcg_engines::setseq_xsl_rr_64_32>();

    std::cout << "double:\n";
	test_double<pcg_engines::setseq_xsl_rr_64_32>();
}
