/*
 * AVX 512 Vectorized PCG Random Number Generation for C++
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

#ifndef PCG_FILL_AVX512_HPP_INCLUDED

#include <immintrin.h>

namespace pcg_fill {

	PCG_ALWAYS_INLINE __m512i _mm512_ziplo_epi32(const __m512i _a, const __m512i _b)
	{
		const __m512i _a0 = _mm512_shuffle_epi32(_a, _MM_PERM_ENUM::_MM_PERM_DBCA);
		const __m512i _b0 = _mm512_shuffle_epi32(_b, _MM_PERM_ENUM::_MM_PERM_DBCA);

		return _mm512_unpacklo_epi32(_a0, _b0);
	}

	PCG_ALWAYS_INLINE __m512i _mm512_ziphi_epi32(const __m512i _a, const __m512i _b)
	{
		const __m512i _a0 = _mm512_shuffle_epi32(_a, _MM_PERM_ENUM::_MM_PERM_DBCA);
		const __m512i _b0 = _mm512_shuffle_epi32(_b, _MM_PERM_ENUM::_MM_PERM_DBCA);

		return _mm512_unpackhi_epi32(_a0, _b0);
	}

	PCG_ALWAYS_INLINE void _preadvance_straight(const __m512i _pcg_mult, const __m512i _inc, __m512i& _acc_mult0, __m512i& _acc_plus0, __m512i& _acc_mult1, __m512i& _acc_plus1)
	{
		__m512i _cur_mult = _pcg_mult;
		__m512i _cur_plus = _inc;

		_acc_mult0 = _mm512_set1_epi64(1ull);
		_acc_plus0 = _mm512_setzero_si512();

		_acc_mult0 = _mm512_mask_mullo_epi64(_acc_mult0, 0x55, _acc_mult0, _cur_mult);
		_acc_plus0 = _mm512_mask_add_epi64(_acc_plus0, 0x55, _mm512_mullo_epi64(_acc_plus0, _cur_mult), _cur_plus);

		_cur_plus = _mm512_add_epi64(_mm512_mullo_epi64(_cur_mult, _cur_plus), _cur_plus);
		_cur_mult = _mm512_mullo_epi64(_cur_mult, _cur_mult);

		_acc_mult0 = _mm512_mask_mullo_epi64(_acc_mult0, 0x66, _acc_mult0, _cur_mult);
		_acc_plus0 = _mm512_mask_add_epi64(_acc_plus0, 0x66, _mm512_mullo_epi64(_acc_plus0, _cur_mult), _cur_plus);

		_cur_plus = _mm512_add_epi64(_mm512_mullo_epi64(_cur_mult, _cur_plus), _cur_plus);
		_cur_mult = _mm512_mullo_epi64(_cur_mult, _cur_mult);

		_acc_mult0 = _mm512_mask_mullo_epi64(_acc_mult0, 0x78, _acc_mult0, _cur_mult);
		_acc_plus0 = _mm512_mask_add_epi64(_acc_plus0, 0x78, _mm512_mullo_epi64(_acc_plus0, _cur_mult), _cur_plus);

		_cur_plus = _mm512_add_epi64(_mm512_mullo_epi64(_cur_mult, _cur_plus), _cur_plus);
		_cur_mult = _mm512_mullo_epi64(_cur_mult, _cur_mult);

		_acc_mult1 = _mm512_mask_mullo_epi64(_acc_mult0, 0x7f, _acc_mult0, _cur_mult);
		_acc_plus1 = _mm512_mask_add_epi64(_acc_plus0, 0x7f, _mm512_mullo_epi64(_acc_plus0, _cur_mult), _cur_plus);

		_acc_mult0 = _mm512_mask_mullo_epi64(_acc_mult0, 0x80, _acc_mult0, _cur_mult);
		_acc_plus0 = _mm512_mask_add_epi64(_acc_plus0, 0x80, _mm512_mullo_epi64(_acc_plus0, _cur_mult), _cur_plus);

		_cur_plus = _mm512_add_epi64(_mm512_mullo_epi64(_cur_mult, _cur_plus), _cur_plus);
		_cur_mult = _mm512_mullo_epi64(_cur_mult, _cur_mult);

		_acc_mult1 = _mm512_mask_mullo_epi64(_acc_mult1, 0x80, _acc_mult1, _cur_mult);
		_acc_plus1 = _mm512_mask_add_epi64(_acc_plus1, 0x80, _mm512_mullo_epi64(_acc_plus1, _cur_mult), _cur_plus);
	}

	PCG_ALWAYS_INLINE void _preadvance_twisted(const __m512i _pcg_mult, const __m512i _inc, __m512i& _acc_mult0, __m512i& _acc_plus0, __m512i& _acc_mult1, __m512i& _acc_plus1)
	{
		__m512i _cur_mult = _pcg_mult;
		__m512i _cur_plus = _inc;

		_acc_mult0 = _mm512_set1_epi64(1ull);
		_acc_plus0 = _mm512_setzero_si512();

		_acc_mult1 = _cur_mult;
		_acc_plus1 = _cur_plus;

		_cur_plus = _mm512_add_epi64(_mm512_mullo_epi64(_cur_mult, _cur_plus), _cur_plus);
		_cur_mult = _mm512_mullo_epi64(_cur_mult, _cur_mult);

		_acc_mult0 = _mm512_mask_mullo_epi64(_acc_mult0, 0x55, _acc_mult0, _cur_mult);
		_acc_plus0 = _mm512_mask_add_epi64(_acc_plus0, 0x55, _mm512_mullo_epi64(_acc_plus0, _cur_mult), _cur_plus);

		_acc_mult1 = _mm512_mask_mullo_epi64(_acc_mult1, 0xaa, _acc_mult1, _cur_mult);
		_acc_plus1 = _mm512_mask_add_epi64(_acc_plus1, 0xaa, _mm512_mullo_epi64(_acc_plus1, _cur_mult), _cur_plus);

		_cur_plus = _mm512_add_epi64(_mm512_mullo_epi64(_cur_mult, _cur_plus), _cur_plus);
		_cur_mult = _mm512_mullo_epi64(_cur_mult, _cur_mult);

		_acc_mult0 = _mm512_mask_mullo_epi64(_acc_mult0, 0x66, _acc_mult0, _cur_mult);
		_acc_plus0 = _mm512_mask_add_epi64(_acc_plus0, 0x66, _mm512_mullo_epi64(_acc_plus0, _cur_mult), _cur_plus);

		_acc_mult1 = _mm512_mask_mullo_epi64(_acc_mult1, 0xcc, _acc_mult1, _cur_mult);
		_acc_plus1 = _mm512_mask_add_epi64(_acc_plus1, 0xcc, _mm512_mullo_epi64(_acc_plus1, _cur_mult), _cur_plus);

		_cur_plus = _mm512_add_epi64(_mm512_mullo_epi64(_cur_mult, _cur_plus), _cur_plus);
		_cur_mult = _mm512_mullo_epi64(_cur_mult, _cur_mult);

		_acc_mult0 = _mm512_mask_mullo_epi64(_acc_mult0, 0x78, _acc_mult0, _cur_mult);
		_acc_plus0 = _mm512_mask_add_epi64(_acc_plus0, 0x78, _mm512_mullo_epi64(_acc_plus0, _cur_mult), _cur_plus);

		_acc_mult1 = _mm512_mask_mullo_epi64(_acc_mult1, 0xf0, _acc_mult1, _cur_mult);
		_acc_plus1 = _mm512_mask_add_epi64(_acc_plus1, 0xf0, _mm512_mullo_epi64(_acc_plus1, _cur_mult), _cur_plus);

		_cur_plus = _mm512_add_epi64(_mm512_mullo_epi64(_cur_mult, _cur_plus), _cur_plus);
		_cur_mult = _mm512_mullo_epi64(_cur_mult, _cur_mult);

		_acc_mult0 = _mm512_mask_mullo_epi64(_acc_mult0, 0x80, _acc_mult0, _cur_mult);
		_acc_plus0 = _mm512_mask_add_epi64(_acc_plus0, 0x80, _mm512_mullo_epi64(_acc_plus0, _cur_mult), _cur_plus);
	}

	PCG_ALWAYS_INLINE __m512 _make_float(__m512 _low, __m512 _range, __m512i _rand01)
	{
		const __m512i _ldexpf32 = _mm512_set1_epi32(0x70000000);
		const __m512i _signmask32 = _mm512_set1_epi32(0x7fffffff);

		__m512 _frand01 = _mm512_cvtepu32_ps(_rand01);

		__m512 _grand01 = _mm512_castsi512_ps(
			_mm512_and_epi32(
				_mm512_add_epi32(
					_mm512_castps_si512(_frand01),
					_ldexpf32), _signmask32));

		__m512 _hrand01 = _mm512_fmadd_ps(_grand01, _range, _low);

		return _hrand01;
	}

	PCG_ALWAYS_INLINE __m512d _make_double(__m512d _low, __m512d _range, __m512i _rand)
	{
		const __m512i _ldexp64 = _mm512_set1_epi64(0x7c00000000000000ull);
		const __m512i _signmask64 = _mm512_set1_epi64(0x7fffffffffffffffull);

		__m512d _frand = _mm512_cvtepu64_pd(_rand);

		__m512d _grand = _mm512_castsi512_pd(
			_mm512_and_epi64(
				_mm512_add_epi64(
					_mm512_castpd_si512(_frand),
					_ldexp64), _signmask64));

		__m512d _hrand = _mm512_fmadd_pd(_grand, _range, _low);

		return _hrand;
	}

	PCG_ALWAYS_INLINE __m512i _xsh_rr(const __m512i _state0, const __m512i _state1)
	{
		const __m512i _lo = _mm512_ziplo_epi32(
			_mm512_srli_epi64(_state0, 27),
			_mm512_srli_epi64(_state1, 27));

		const __m512i _hi = _mm512_ziphi_epi32(
			_state0,
			_state1);
	
		const __m512i _r = _mm512_rorv_epi32(
			_mm512_xor_epi32(
					_mm512_srli_epi32(_hi, 13),
					_lo),
			_mm512_srli_epi32(_hi, 27));
		
		return _r;
	}

	template<>
	void fill(pcg_engines::setseq_xsh_rr_64_32& rng, float* ptr, size_t size, float low, float high)
	{
		const float range = high - low;

		const size_t size16 = size & ~0x0f;

		const __m512i _tail = _mm512_set1_epi64(7ull);
		const __m512i _body = _mm512_set_epi64(14, 13, 12, 11, 10, 9, 8, 7);

		const __m512 _range = _mm512_set1_ps(range);
		const __m512 _low = _mm512_set1_ps(low);


		uint64_t multiplier_, increment_, state_;
		scan_state(rng, multiplier_, increment_, state_);

		const __m512i _mult = _mm512_set1_epi64(multiplier_);
		const __m512i _inc = _mm512_set1_epi64(increment_);
		__m512i _state = _mm512_set1_epi64(state_);

		__m512i _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1;

		_preadvance_twisted(_mult, _inc, _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1);

		for (uint64_t i = 0; i < size16; i += 16)
		{
			__m512i _state00 = _mm512_add_epi64(_mm512_mullo_epi64(_state, _acc_mult0), _acc_plus0);

			__m512i _state01 = _mm512_add_epi64(_mm512_mullo_epi64(_state, _acc_mult1), _acc_plus1);

			__m512i _advance = _mm512_permutexvar_epi64(_tail, _state00);

			_state00 = _mm512_permutex2var_epi64(_state, _body, _state00);

			_state = _advance;

			__m512i _rand01 = _xsh_rr(_state00, _state01);

			 __m512 _frand01 = _make_float(_low, _range, _rand01);

			_mm512_storeu_ps(ptr + i, _frand01);
		}

		rng.advance(size16);

		for (size_t i = size16; i < size; i++) {
			uint32_t val = rng();

			ptr[i] = ldexpf(static_cast<float>(val), -32) * range + low;
		}
	}

	template<>
	void fill(pcg_engines::setseq_xsh_rr_64_32& rng0, pcg_engines::setseq_xsh_rr_64_32& rng1, double* ptr, size_t size, double low, double high)
	{
		const double range = high - low;

		const size_t size16 = size & ~0x0f;

		const __m512i _tail = _mm512_set1_epi64(7ull);
		const __m512i _body = _mm512_set_epi64(14, 13, 12, 11, 10, 9, 8, 7);

		const __m512d _range = _mm512_set1_pd(range);
		const __m512d _low = _mm512_set1_pd(low);

		__m512i _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01;

		uint64_t multiplier0_, increment0_, state0_;
		scan_state(rng0, multiplier0_, increment0_, state0_);

		const __m512i _mult0 = _mm512_set1_epi64(multiplier0_);
		const __m512i _inc0 = _mm512_set1_epi64(increment0_);
		__m512i _state0 = _mm512_set1_epi64(state0_);

		_preadvance_straight(_mult0, _inc0, _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01);

		__m512i _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11;

		uint64_t multiplier1_, increment1_, state1_;
		scan_state(rng1, multiplier1_, increment1_, state1_);

		const __m512i _mult1 = _mm512_set1_epi64(multiplier1_);
		const __m512i _inc1 = _mm512_set1_epi64(increment1_);
		__m512i _state1 = _mm512_set1_epi64(state1_);

		_preadvance_straight(_mult1, _inc1, _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11);

		for (uint64_t i = 0; i < size16; i += 16)
		{
			__m512i _state00 = _mm512_add_epi64(_mm512_mullo_epi64(_state0, _acc_mult00), _acc_plus00);

			__m512i _state01 = _mm512_add_epi64(_mm512_mullo_epi64(_state0, _acc_mult01), _acc_plus01);

			__m512i _state10 = _mm512_add_epi64(_mm512_mullo_epi64(_state1, _acc_mult10), _acc_plus10);

			__m512i _state11 = _mm512_add_epi64(_mm512_mullo_epi64(_state1, _acc_mult11), _acc_plus11);

			__m512i _advance0 = _mm512_permutexvar_epi64(_tail, _state01);

			__m512i _advance1 = _mm512_permutexvar_epi64(_tail, _state11);

			_state01 = _mm512_permutex2var_epi64(_state00, _body, _state01);

			_state00 = _mm512_permutex2var_epi64(_state0, _body, _state00);

			_state11 = _mm512_permutex2var_epi64(_state10, _body, _state11);

			_state10 = _mm512_permutex2var_epi64(_state1, _body, _state10);

			_state0 = _advance0;

			_state1 = _advance1;

			__m512i _rand0 = _mm512_shuffle_epi32(
				_xsh_rr(_state00, _state01),
				_MM_PERM_ENUM::_MM_PERM_DBCA);

			__m512i _rand1 = _mm512_shuffle_epi32(
				_xsh_rr(_state10, _state11),
				_MM_PERM_ENUM::_MM_PERM_DBCA);
			 
			__m512d _frand0 = _make_double(_low, _range,
				_mm512_unpacklo_epi32(_rand0, _rand1));

			__m512d _frand1 = _make_double(_low, _range,
				_mm512_unpackhi_epi32(_rand0, _rand1));

			_mm512_storeu_pd(ptr + i, _frand0);

			_mm512_storeu_pd(ptr + i + 8, _frand1);
		}

		rng0.advance(size16);
		rng1.advance(size16);

		for (size_t i = size16; i < size; i++)
		{
			uint64_t val = (static_cast<uint64_t>(rng1()) << 32) | rng0();

			ptr[i] = ldexp(static_cast<double>(val), -64) * range + low;
		}
	}

	PCG_ALWAYS_INLINE __m512i _xsh_rs(const __m512i _state0, const __m512i _state1)
	{
		const __m512i _rand0 = _mm512_srlv_epi64(
			_mm512_xor_epi64(
				_mm512_srli_epi64(_state0, 44),
				_mm512_srli_epi64(_state0, 22)),
			_mm512_srli_epi64(_state0, 61));

		const __m512i _rand1 = _mm512_srlv_epi64(
			_mm512_xor_epi64(
				_mm512_srli_epi64(_state1, 44),
				_mm512_srli_epi64(_state1, 22)),
			_mm512_srli_epi64(_state1, 61));

		return _mm512_ziplo_epi32(_rand0, _rand1);
	}

	template<>
	void fill(pcg_engines::setseq_xsh_rs_64_32& rng, float* ptr, size_t size, float low, float high)
	{
		const float range = high - low;

		const size_t size16 = size & ~0x0f;

		const __m512i _tail = _mm512_set1_epi64(7ull);
		const __m512i _body = _mm512_set_epi64(14, 13, 12, 11, 10, 9, 8, 7);

		const __m512 _range = _mm512_set1_ps(range);
		const __m512 _low = _mm512_set1_ps(low);

		uint64_t multiplier_, increment_, state_;
		scan_state(rng, multiplier_, increment_, state_);

		const __m512i _mult = _mm512_set1_epi64(multiplier_);
		const __m512i _inc = _mm512_set1_epi64(increment_);
		__m512i _state = _mm512_set1_epi64(state_);

		__m512i _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1;

		_preadvance_twisted(_mult, _inc, _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1);

		for (uint64_t i = 0; i < size16; i += 16)
		{
			__m512i _state00 = _mm512_add_epi64(_mm512_mullo_epi64(_state, _acc_mult0), _acc_plus0);

			__m512i _state01 = _mm512_add_epi64(_mm512_mullo_epi64(_state, _acc_mult1), _acc_plus1);

			__m512i _advance = _mm512_permutexvar_epi64(_tail, _state00);

			_state00 = _mm512_permutex2var_epi64(_state, _body, _state00);

			_state = _advance;

			__m512i _rand = _xsh_rs(_state00, _state01);

			__m512 _frand = _make_float(_low, _range, _rand);

			_mm512_storeu_ps(ptr + i, _frand);
		}

		rng.advance(size16);

		for (size_t i = size16; i < size; i++) {
			uint32_t val = rng();

			ptr[i] = ldexpf(static_cast<float>(val), -32) * range + low;
		}
	}

	template<>
	void fill(pcg_engines::setseq_xsh_rs_64_32& rng0, pcg_engines::setseq_xsh_rs_64_32& rng1, double* ptr, size_t size, double low, double high)
	{
		const double range = high - low;

		const size_t size16 = size & ~0x0f;

		const __m512i _tail = _mm512_set1_epi64(7ull);
		const __m512i _body = _mm512_set_epi64(14, 13, 12, 11, 10, 9, 8, 7);
 
		const __m512d _range = _mm512_set1_pd(range);
		const __m512d _low = _mm512_set1_pd(low);

		__m512i _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01;

		uint64_t multiplier0_, increment0_, state0_;
		scan_state(rng0, multiplier0_, increment0_, state0_);

		const __m512i _mult0 = _mm512_set1_epi64(multiplier0_);
		const __m512i _inc0 = _mm512_set1_epi64(increment0_);
		__m512i _state0 = _mm512_set1_epi64(state0_);

		_preadvance_straight(_mult0, _inc0, _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01);

		__m512i _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11;

		uint64_t multiplier1_, increment1_, state1_;
		scan_state(rng1, multiplier1_, increment1_, state1_);

		const __m512i _mult1 = _mm512_set1_epi64(multiplier1_);
		const __m512i _inc1 = _mm512_set1_epi64(increment1_);
		__m512i _state1 = _mm512_set1_epi64(state1_);

		_preadvance_straight(_mult1, _inc1, _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11);

		for (uint64_t i = 0; i < size16; i += 16)
		{
			__m512i _state00 = _mm512_add_epi64(_mm512_mullo_epi64(_state0, _acc_mult00), _acc_plus00);

			__m512i _state01 = _mm512_add_epi64(_mm512_mullo_epi64(_state0, _acc_mult01), _acc_plus01);

			__m512i _state10 = _mm512_add_epi64(_mm512_mullo_epi64(_state1, _acc_mult10), _acc_plus10);

			__m512i _state11 = _mm512_add_epi64(_mm512_mullo_epi64(_state1, _acc_mult11), _acc_plus11);

			__m512i _advance0 = _mm512_permutexvar_epi64(_tail, _state01);

			__m512i _advance1 = _mm512_permutexvar_epi64(_tail, _state11);

			_state01 = _mm512_permutex2var_epi64(_state00, _body, _state01);

			_state00 = _mm512_permutex2var_epi64(_state0, _body, _state00);

			_state11 = _mm512_permutex2var_epi64(_state10, _body, _state11);

			_state10 = _mm512_permutex2var_epi64(_state1, _body, _state10);

			_state0 = _advance0;

			_state1 = _advance1;

			__m512i _rand0 = _mm512_shuffle_epi32(
				_xsh_rs(_state00, _state01),
				_MM_PERM_ENUM::_MM_PERM_DBCA);

			__m512i _rand1 = _mm512_shuffle_epi32(
				_xsh_rs(_state10, _state11),
				_MM_PERM_ENUM::_MM_PERM_DBCA);

			__m512d _frand0 = _make_double(_low, _range,
				_mm512_unpacklo_epi32(_rand0, _rand1));

			__m512d _frand1 = _make_double(_low, _range,
				_mm512_unpackhi_epi32(_rand0, _rand1));

			_mm512_storeu_pd(ptr + i, _frand0);

			_mm512_storeu_pd(ptr + i + 8, _frand1);
		}

		rng0.advance(size16);
		rng1.advance(size16);

		for (size_t i = size16; i < size; i++)
		{
			uint64_t val = (static_cast<uint64_t>(rng1()) << 32) | rng0();

			ptr[i] = ldexp(static_cast<double>(val), -64) * range + low;
		}
	}

	PCG_ALWAYS_INLINE __m512i _rxs_m(const __m512i _state0, const __m512i _state1)
	{
		const __m512i _multiplier = _mm512_set1_epi64(pcg_detail::mcg_multiplier<uint64_t>::multiplier());

		const __m512i _rand0 = _mm512_mullo_epi64(
					_mm512_xor_epi64(
						_mm512_srlv_epi64(
							_mm512_srli_epi64(_state0, 4),
							_mm512_srli_epi64(_state0, 60)),
						_state0),
					_multiplier);

		const __m512i _rand1 = _mm512_mullo_epi64(
					_mm512_xor_epi64(
						_mm512_srlv_epi64(
							_mm512_srli_epi64(_state1, 4),
							_mm512_srli_epi64(_state1, 60)),
						_state1),
					_multiplier);

		return _mm512_ziphi_epi32(_rand0, _rand1);
	}

	template<>
	void fill(pcg_engines::setseq_rxs_m_64_32& rng, float* ptr, size_t size, float low, float high)
	{
		const float range = high - low;

		const size_t size16 = size & ~0x0f;

		const __m512i _tail = _mm512_set1_epi64(7ull);
		const __m512i _body = _mm512_set_epi64(14, 13, 12, 11, 10, 9, 8, 7);

		const __m512 _range = _mm512_set1_ps(range);
		const __m512 _low = _mm512_set1_ps(low);


		uint64_t multiplier_, increment_, state_;
		scan_state(rng, multiplier_, increment_, state_);

		const __m512i _mult = _mm512_set1_epi64(multiplier_);
		const __m512i _inc = _mm512_set1_epi64(increment_);
		__m512i _state = _mm512_set1_epi64(state_);

		__m512i _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1;

		_preadvance_twisted(_mult, _inc, _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1);

		for (uint64_t i = 0; i < size16; i += 16)
		{
			__m512i _state00 = _mm512_add_epi64(_mm512_mullo_epi64(_state, _acc_mult0), _acc_plus0);

			__m512i _state01 = _mm512_add_epi64(_mm512_mullo_epi64(_state, _acc_mult1), _acc_plus1);

			__m512i _advance = _mm512_permutexvar_epi64(_tail, _state00);

			_state00 = _mm512_permutex2var_epi64(_state, _body, _state00);

			_state = _advance;

			__m512i _rand = _rxs_m(_state00, _state01);

			__m512 _frand = _make_float(_low, _range, _rand);

			_mm512_storeu_ps(ptr + i, _frand);
		}

		rng.advance(size16);

		for (size_t i = size16; i < size; i++) {
			uint32_t val = rng();

			ptr[i] = ldexpf(static_cast<float>(val), -32) * range + low;
		}
	}

	template<>
	void fill(pcg_engines::setseq_rxs_m_64_32& rng0, pcg_engines::setseq_rxs_m_64_32& rng1, double* ptr, size_t size, double low, double high)
	{
		const double range = high - low;

		const size_t size16 = size & ~0x0f;

		const __m512i _tail = _mm512_set1_epi64(7ull);
		const __m512i _body = _mm512_set_epi64(14, 13, 12, 11, 10, 9, 8, 7);

		const __m512d _range = _mm512_set1_pd(range);
		const __m512d _low = _mm512_set1_pd(low);

		__m512i _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01;

		uint64_t multiplier0_, increment0_, state0_;
		scan_state(rng0, multiplier0_, increment0_, state0_);

		const __m512i _mult0 = _mm512_set1_epi64(multiplier0_);
		const __m512i _inc0 = _mm512_set1_epi64(increment0_);
		__m512i _state0 = _mm512_set1_epi64(state0_);

		_preadvance_straight(_mult0, _inc0, _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01);

		__m512i _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11;

		uint64_t multiplier1_, increment1_, state1_;
		scan_state(rng1, multiplier1_, increment1_, state1_);

		const __m512i _mult1 = _mm512_set1_epi64(multiplier1_);
		const __m512i _inc1 = _mm512_set1_epi64(increment1_);
		__m512i _state1 = _mm512_set1_epi64(state1_);

		_preadvance_straight(_mult1, _inc1, _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11);

		for (uint64_t i = 0; i < size16; i += 16)
		{
			__m512i _state00 = _mm512_add_epi64(_mm512_mullo_epi64(_state0, _acc_mult00), _acc_plus00);

			__m512i _state01 = _mm512_add_epi64(_mm512_mullo_epi64(_state0, _acc_mult01), _acc_plus01);

			__m512i _state10 = _mm512_add_epi64(_mm512_mullo_epi64(_state1, _acc_mult10), _acc_plus10);

			__m512i _state11 = _mm512_add_epi64(_mm512_mullo_epi64(_state1, _acc_mult11), _acc_plus11);

			__m512i _advance0 = _mm512_permutexvar_epi64(_tail, _state01);

			__m512i _advance1 = _mm512_permutexvar_epi64(_tail, _state11);

			_state01 = _mm512_permutex2var_epi64(_state00, _body, _state01);

			_state00 = _mm512_permutex2var_epi64(_state0, _body, _state00);

			_state11 = _mm512_permutex2var_epi64(_state10, _body, _state11);

			_state10 = _mm512_permutex2var_epi64(_state1, _body, _state10);

			_state0 = _advance0;

			_state1 = _advance1;

			__m512i _rand0 = _mm512_shuffle_epi32(
				_rxs_m(_state00, _state01),
				_MM_PERM_ENUM::_MM_PERM_DBCA);

			__m512i _rand1 = _mm512_shuffle_epi32(
				_rxs_m(_state10, _state11),
				_MM_PERM_ENUM::_MM_PERM_DBCA);

			__m512d _frand0 = _make_double(_low, _range,
				_mm512_unpacklo_epi32(_rand0, _rand1));

			__m512d _frand1 = _make_double(_low, _range,
				_mm512_unpackhi_epi32(_rand0, _rand1));

			_mm512_storeu_pd(ptr + i, _frand0);

			_mm512_storeu_pd(ptr + i + 8, _frand1);
		}

		rng0.advance(size16);
		rng1.advance(size16);

		for (size_t i = size16; i < size; i++)
		{
			uint64_t val = (static_cast<uint64_t>(rng1()) << 32) | rng0();

			ptr[i] = ldexp(static_cast<double>(val), -64) * range + low;
		}
	}

	PCG_ALWAYS_INLINE __m512i _dxsm(const __m512i _state0, const __m512i _state1)
	{
		const __m512i _one = _mm512_set1_epi32(1);
		const __m512i _multiplier = _mm512_set1_epi32(static_cast<uint32_t>(pcg_detail::cheap_multiplier<uint64_t>::multiplier()));

		const __m512i _s0 = _mm512_shuffle_epi32(_state0, _MM_PERM_ENUM::_MM_PERM_DBCA);
		const __m512i _s1 = _mm512_shuffle_epi32(_state1, _MM_PERM_ENUM::_MM_PERM_DBCA);

		__m512i _hi = _mm512_unpackhi_epi32(_s0, _s1);

		__m512i _lo = _mm512_or_epi32(
			_mm512_unpacklo_epi32(_s0, _s1),
			_one);

		_hi = _mm512_xor_epi32(_hi, _mm512_srli_epi32(_hi, 16));

		_hi = _mm512_mullo_epi32(
			_hi,
			_multiplier);

		_hi = _mm512_xor_epi32(_hi, _mm512_srli_epi32(_hi, 24));

		_hi = _mm512_mullo_epi32(_hi, _lo);

		return _hi;
	}

	template<>
	void fill(pcg_engines::setseq_dxsm_64_32& rng, float* ptr, size_t size, float low, float high)
	{
		const float range = high - low;

		const size_t size16 = size & ~0x0f;

		const __m512i _tail = _mm512_set1_epi64(7ull);
		const __m512i _body = _mm512_set_epi64(14, 13, 12, 11, 10, 9, 8, 7);

		const __m512 _range = _mm512_set1_ps(range);
		const __m512 _low = _mm512_set1_ps(low);


		uint64_t multiplier_, increment_, state_;
		scan_state(rng, multiplier_, increment_, state_);

		const __m512i _mult = _mm512_set1_epi64(multiplier_);
		const __m512i _inc = _mm512_set1_epi64(increment_);
		__m512i _state = _mm512_set1_epi64(state_);

		__m512i _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1;

		_preadvance_twisted(_mult, _inc, _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1);

		for (uint64_t i = 0; i < size16; i += 16)
		{
			__m512i _state00 = _mm512_add_epi64(_mm512_mullo_epi64(_state, _acc_mult0), _acc_plus0);

			__m512i _state01 = _mm512_add_epi64(_mm512_mullo_epi64(_state, _acc_mult1), _acc_plus1);
			
			__m512i _advance = _mm512_permutexvar_epi64(_tail, _state00);

			_state00 = _mm512_permutex2var_epi64(_state, _body, _state00);

			_state = _advance;

			__m512i _rand = _dxsm(_state00, _state01);

			__m512 _frand = _make_float(_low, _range, _rand);

			_mm512_storeu_ps(ptr + i, _frand);
		}

		rng.advance(size16);

		for (size_t i = size16; i < size; i++) {
			uint32_t val = rng();

			ptr[i] = ldexpf(static_cast<float>(val), -32) * range + low;
		}
	}

	template<>
	void fill(pcg_engines::setseq_dxsm_64_32& rng0, pcg_engines::setseq_dxsm_64_32& rng1, double* ptr, size_t size, double low, double high)
	{
		const double range = high - low;

		const size_t size16 = size & ~0x0f;

		const __m512i _tail = _mm512_set1_epi64(7ull);
		const __m512i _body = _mm512_set_epi64(14, 13, 12, 11, 10, 9, 8, 7);

		const __m512d _range = _mm512_set1_pd(range);
		const __m512d _low = _mm512_set1_pd(low);

		__m512i _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01;

		uint64_t multiplier0_, increment0_, state0_;
		scan_state(rng0, multiplier0_, increment0_, state0_);

		const __m512i _mult0 = _mm512_set1_epi64(multiplier0_);
		const __m512i _inc0 = _mm512_set1_epi64(increment0_);
		__m512i _state0 = _mm512_set1_epi64(state0_);

		_preadvance_straight(_mult0, _inc0, _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01);

		__m512i _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11;

		uint64_t multiplier1_, increment1_, state1_;
		scan_state(rng1, multiplier1_, increment1_, state1_);

		const __m512i _mult1 = _mm512_set1_epi64(multiplier1_);
		const __m512i _inc1 = _mm512_set1_epi64(increment1_);
		__m512i _state1 = _mm512_set1_epi64(state1_);

		_preadvance_straight(_mult1, _inc1, _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11);

		for (uint64_t i = 0; i < size16; i += 16)
		{
			__m512i _state00 = _mm512_add_epi64(_mm512_mullo_epi64(_state0, _acc_mult00), _acc_plus00);

			__m512i _state01 = _mm512_add_epi64(_mm512_mullo_epi64(_state0, _acc_mult01), _acc_plus01);

			__m512i _state10 = _mm512_add_epi64(_mm512_mullo_epi64(_state1, _acc_mult10), _acc_plus10);

			__m512i _state11 = _mm512_add_epi64(_mm512_mullo_epi64(_state1, _acc_mult11), _acc_plus11);

			__m512i _advance0 = _mm512_permutexvar_epi64(_tail, _state01);

			__m512i _advance1 = _mm512_permutexvar_epi64(_tail, _state11);

			_state01 = _mm512_permutex2var_epi64(_state00, _body, _state01);

			_state00 = _mm512_permutex2var_epi64(_state0, _body, _state00);

			_state11 = _mm512_permutex2var_epi64(_state10, _body, _state11);

			_state10 = _mm512_permutex2var_epi64(_state1, _body, _state10);

			_state0 = _advance0;

			_state1 = _advance1;

			__m512i _rand0 = _mm512_shuffle_epi32(
				_dxsm(_state00, _state01),
				_MM_PERM_ENUM::_MM_PERM_DBCA);

			__m512i _rand1 = _mm512_shuffle_epi32(
				_dxsm(_state10, _state11),
				_MM_PERM_ENUM::_MM_PERM_DBCA);

			__m512d _frand0 = _make_double(_low, _range,
				_mm512_unpacklo_epi32(_rand0, _rand1));

			__m512d _frand1 = _make_double(_low, _range,
				_mm512_unpackhi_epi32(_rand0, _rand1));

			_mm512_storeu_pd(ptr + i, _frand0);

			_mm512_storeu_pd(ptr + i + 8, _frand1);
		}

		rng0.advance(size16);
		rng1.advance(size16);

		for (size_t i = size16; i < size; i++)
		{
			uint64_t val = (static_cast<uint64_t>(rng1()) << 32) | rng0();

			ptr[i] = ldexp(static_cast<double>(val), -64) * range + low;
		}
	}

	PCG_ALWAYS_INLINE __m512i _xsl_rr(const __m512i _state0, const __m512i _state1)
	{
		const __m512i _s0 = _mm512_shuffle_epi32(_state0, _MM_PERM_ENUM::_MM_PERM_DBCA);
		const __m512i _s1 = _mm512_shuffle_epi32(_state1, _MM_PERM_ENUM::_MM_PERM_DBCA);

		const __m512i _lo = _mm512_unpacklo_epi32(_s0, _s1);

		const __m512i _hi = _mm512_unpackhi_epi32(_s0, _s1);

		return _mm512_rorv_epi32(
			_mm512_xor_epi32(_hi, _lo),
			_mm512_srli_epi32(_hi, 27));
	}

	template<>
	void fill(pcg_engines::setseq_xsl_rr_64_32& rng, float* ptr, size_t size, float low, float high)
	{
		const float range = high - low;

		const size_t size16 = size & ~0x0f;

		const __m512i _tail = _mm512_set1_epi64(7ull);
		const __m512i _body = _mm512_set_epi64(14, 13, 12, 11, 10, 9, 8, 7);

		const __m512 _range = _mm512_set1_ps(range);
		const __m512 _low = _mm512_set1_ps(low);


		uint64_t multiplier_, increment_, state_;
		scan_state(rng, multiplier_, increment_, state_);

		const __m512i _mult = _mm512_set1_epi64(multiplier_);
		const __m512i _inc = _mm512_set1_epi64(increment_);
		__m512i _state = _mm512_set1_epi64(state_);

		__m512i _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1;

		_preadvance_twisted(_mult, _inc, _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1);

		for (uint64_t i = 0; i < size16; i += 16)
		{
			__m512i _state00 = _mm512_add_epi64(_mm512_mullo_epi64(_state, _acc_mult0), _acc_plus0);

			__m512i _state01 = _mm512_add_epi64(_mm512_mullo_epi64(_state, _acc_mult1), _acc_plus1);

			__m512i _advance = _mm512_permutexvar_epi64(_tail, _state00);

			_state00 = _mm512_permutex2var_epi64(_state, _body, _state00);

			_state = _advance;

			__m512i _rand = _xsl_rr(_state00, _state01);

			__m512 _frand = _make_float(_low, _range, _rand);

			_mm512_storeu_ps(ptr + i, _frand);
		}

		rng.advance(size16);

		for (size_t i = size16; i < size; i++) {
			uint32_t val = rng();

			ptr[i] = ldexpf(static_cast<float>(val), -32) * range + low;
		}
	}

	template<>
	void fill(pcg_engines::setseq_xsl_rr_64_32& rng0, pcg_engines::setseq_xsl_rr_64_32& rng1, double* ptr, size_t size, double low, double high)
	{
		const double range = high - low;

		const size_t size16 = size & ~0x0f;

		const __m512i _tail = _mm512_set1_epi64(7ull);
		const __m512i _body = _mm512_set_epi64(14, 13, 12, 11, 10, 9, 8, 7);

		const __m512d _range = _mm512_set1_pd(range);
		const __m512d _low = _mm512_set1_pd(low);

		__m512i _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01;

		uint64_t multiplier0_, increment0_, state0_;
		scan_state(rng0, multiplier0_, increment0_, state0_);

		const __m512i _mult0 = _mm512_set1_epi64(multiplier0_);
		const __m512i _inc0 = _mm512_set1_epi64(increment0_);
		__m512i _state0 = _mm512_set1_epi64(state0_);

		_preadvance_straight(_mult0, _inc0, _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01);

		__m512i _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11;

		uint64_t multiplier1_, increment1_, state1_;
		scan_state(rng1, multiplier1_, increment1_, state1_);

		const __m512i _mult1 = _mm512_set1_epi64(multiplier1_);
		const __m512i _inc1 = _mm512_set1_epi64(increment1_);
		__m512i _state1 = _mm512_set1_epi64(state1_);

		_preadvance_straight(_mult1, _inc1, _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11);

		for (uint64_t i = 0; i < size16; i += 16)
		{
			__m512i _state00 = _mm512_add_epi64(_mm512_mullo_epi64(_state0, _acc_mult00), _acc_plus00);

			__m512i _state01 = _mm512_add_epi64(_mm512_mullo_epi64(_state0, _acc_mult01), _acc_plus01);

			__m512i _state10 = _mm512_add_epi64(_mm512_mullo_epi64(_state1, _acc_mult10), _acc_plus10);

			__m512i _state11 = _mm512_add_epi64(_mm512_mullo_epi64(_state1, _acc_mult11), _acc_plus11);

			__m512i _advance0 = _mm512_permutexvar_epi64(_tail, _state01);

			__m512i _advance1 = _mm512_permutexvar_epi64(_tail, _state11);

			_state01 = _mm512_permutex2var_epi64(_state00, _body, _state01);

			_state00 = _mm512_permutex2var_epi64(_state0, _body, _state00);

			_state11 = _mm512_permutex2var_epi64(_state10, _body, _state11);

			_state10 = _mm512_permutex2var_epi64(_state1, _body, _state10);

			_state0 = _advance0;

			_state1 = _advance1;

			__m512i _rand0 = _mm512_shuffle_epi32(
				_xsl_rr(_state00, _state01),
				_MM_PERM_ENUM::_MM_PERM_DBCA);

			__m512i _rand1 = _mm512_shuffle_epi32(
				_xsl_rr(_state10, _state11),
				_MM_PERM_ENUM::_MM_PERM_DBCA);

			__m512d _frand0 = _make_double(_low, _range,
				_mm512_unpacklo_epi32(_rand0, _rand1));

			__m512d _frand1 = _make_double(_low, _range,
				_mm512_unpackhi_epi32(_rand0, _rand1));

			_mm512_storeu_pd(ptr + i, _frand0);

			_mm512_storeu_pd(ptr + i + 8, _frand1);
		}

		rng0.advance(size16);
		rng1.advance(size16);

		for (size_t i = size16; i < size; i++)
		{
			uint64_t val = (static_cast<uint64_t>(rng1()) << 32) | rng0();

			ptr[i] = ldexp(static_cast<double>(val), -64) * range + low;
		}
	}
}
#define PCG_FILL_AVX512_HPP_INCLUDED 1
#endif // PCG_FILL_AVX512_HPP_INCLUDED