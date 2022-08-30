/*
 * Vectorized PCG Random Number Generation for C++
 *
 * Copyright 2022 Roman Snytsar <roman.snytsar@microsoft.com>.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 * Licensed under the Apache License, Version 2.0 (provided in
 * LICENSE-APACHE.txt and at http://www.apache.org/licenses/LICENSE-2.0)
 * or under the MIT license (provided in LICENSE-MIT.txt and at
 * http://opensource.org/licenses/MIT), at your option. This file may not
 * be copied, modified, or distributed except according to those terms.
 *
 * Distributed on an "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, either
 * express or implied.  See your chosen license for details.
 *
 * For additional information about the PCG random number generation scheme,
 * visit http://www.pcg-random.org/.
 */

#ifndef PCG_FILL_HPP_INCLUDED
#define PCG_FILL_HPP_INCLUDED 1

#include <math.h>
#include <sstream>

#include "pcg_random.hpp"

namespace pcg_fill {

	template<template<typename XT, typename IT> class output_mixin>
	void fill(pcg_engines::setseq_base<uint32_t, uint64_t, output_mixin>& rng,
		float* ptr, size_t size, float low, float high)
	{
		const float range = high - low;

		for (size_t i = 0; i < size; i++)
		{
			uint32_t val = rng();

			ptr[i] = ldexpf(static_cast<float>(val), -32) * range + low;
		}
	}

	template<template<typename XT, typename IT> class output_mixin>
	void fill(pcg_engines::setseq_base<uint32_t, uint64_t, output_mixin>& rng0,
		pcg_engines::setseq_base<uint32_t, uint64_t, output_mixin>& rng1,
		double* ptr, size_t size, double low, double high)
	{
		const double range = high - low;

		for (size_t i = 0; i < size; i++)
		{
			uint64_t val = (static_cast<uint64_t>(rng1()) << 32) | rng0();

			ptr[i] = ldexp(static_cast<double>(val), -64) * range + low;
		}
	}

	template<typename engine>
	void scan_state(engine& rng, uint64_t& multiplier, uint64_t& increment, uint64_t& state)
	{
		std::stringstream dump;

		dump << rng;

		dump >> multiplier >> increment >> state;
	}
}

#if defined __AVX512F__

#include "pcg_fill_avx512.hpp"

#elif defined __ARM_FEATURE_SVE

#include "pcg_fill_sve2.hpp"

#endif

#endif // PCG_FILL_HPP_INCLUDED
