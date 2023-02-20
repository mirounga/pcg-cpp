/*
 * SVE2 Vectorized PCG Random Number Generation for C++
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

#ifndef PCG_FILL_SVE2_HPP_INCLUDED

#include <arm_sve.h>
#include <iostream>

namespace pcg_fill {

	template <typename CharT, typename Traits>
	std::basic_ostream<CharT,Traits>&
	operator<<(std::basic_ostream<CharT,Traits>& out, svuint64_t _val)
	{
		const svbool_t all_b64 = svptrue_b64();

		svuint64_t _idx = svindex_u64(0ull, 1ull);

		for(uint64_t i = 0; i< svcntd(); i++) {
			uint64_t active = svlastb_u64(svcmpeq_n_u64(all_b64, _idx, i), _val);
			out << active << "\t";
		}
		return out;
	}

	template <typename CharT, typename Traits>
	std::basic_ostream<CharT,Traits>&
	operator<<(std::basic_ostream<CharT,Traits>& out, svuint32_t _val)
	{
		const svbool_t all_b64 = svptrue_b64();

		svuint32_t _idx = svindex_u32(0ul, 1ul);

		for(uint32_t i = 0; i< svcntw(); i++) {
			uint64_t active = svlastb_u32(svcmpeq_n_u32(all_b64, _idx, i), _val);
			out << active << "\t";
		}
		return out;
	}

	template <typename CharT, typename Traits>
	std::basic_ostream<CharT,Traits>&
	operator<<(std::basic_ostream<CharT,Traits>& out, svfloat64_t _val)
	{
		const svbool_t all_b64 = svptrue_b64();

		svuint64_t _idx = svindex_u64(0ull, 1ull);

		for(uint64_t i = 0; i< svcntd(); i++) {
			double active = svlastb_f64(svcmpeq_n_u64(all_b64, _idx, i), _val);
			out << active << "\t";
		}
		return out;
	}

	template <typename CharT, typename Traits>
	std::basic_ostream<CharT,Traits>&
	operator<<(std::basic_ostream<CharT,Traits>& out, svfloat32_t _val)
	{
		const svbool_t all_b64 = svptrue_b64();

		svuint32_t _idx = svindex_u32(0ul, 1ul);

		for(uint32_t i = 0; i< svcntw(); i++) {
			float active = svlastb_f32(svcmpeq_n_u32(all_b64, _idx, i), _val);
			out << active << "\t";
		}
		return out;
	}

	PCG_ALWAYS_INLINE void _preadvance_straight(const svuint64_t _pcg_mult, const svuint64_t _inc, svuint64_t& _acc_mult0, svuint64_t& _acc_plus0, svuint64_t& _acc_mult1, svuint64_t& _acc_plus1)
	{
		const svbool_t all_b64 = svptrue_b64();

		svuint64_t _cur_mult = _pcg_mult;
		svuint64_t _cur_plus = _inc;

		_acc_mult0 = _acc_mult1 = svdup_u64(1ull);
		_acc_plus0 = _acc_plus1 = svdup_u64(0ull);
		
		for (svuint64_t _delta0 = svindex_u64(1ull, 1ull), _delta1 = svqadd_n_u64(_delta0, svcntd());
			svptest_any(all_b64, svcmpne_n_u64(all_b64, _delta1, 0ull));
			_delta0 = svlsr_n_u64_x(all_b64, _delta0, 1ull),
			_delta1 = svlsr_n_u64_x(all_b64, _delta1, 1ull))
		{
			svbool_t pg0 = svcmpne_n_u64(all_b64,
				svlsl_n_u64_x(all_b64, _delta0, 63ull), 
				0ull);

			svbool_t pg1 = svcmpne_n_u64(all_b64,
				svlsl_n_u64_x(all_b64, _delta1, 63ull), 
				0ull);

			_acc_mult0 = svmul_u64_m(pg0, _acc_mult0, _cur_mult);
			_acc_plus0 = svmad_u64_m(pg0, _acc_plus0, _cur_mult, _cur_plus);

			_acc_mult1 = svmul_u64_m(pg1, _acc_mult1, _cur_mult);
			_acc_plus1 = svmad_u64_m(pg1, _acc_plus1, _cur_mult, _cur_plus);

			_cur_plus = svmad_u64_z(all_b64, _cur_mult, _cur_plus, _cur_plus);
			_cur_mult = svmul_u64_z(all_b64, _cur_mult, _cur_mult);
		}
	}

	PCG_ALWAYS_INLINE void _preadvance_twisted(const svuint64_t _pcg_mult, const svuint64_t _inc, svuint64_t& _acc_mult0, svuint64_t& _acc_plus0, svuint64_t& _acc_mult1, svuint64_t& _acc_plus1)
	{
		const svbool_t all_b64 = svptrue_b64();

		svuint64_t _cur_mult = _pcg_mult;
		svuint64_t _cur_plus = _inc;

		_acc_mult0 = _acc_mult1 = svdup_u64(1ull);
		_acc_plus0 = _acc_plus1 = svdup_u64(0ull);

		for (svuint64_t _delta0 = svindex_u64(2ull, 2ull), _delta1 = svindex_u64(1ull, 2ull);
			svptest_any(all_b64, svcmpne_n_u64(all_b64, _delta0, 0ull)); 
			_delta0 = svlsr_n_u64_x(all_b64, _delta0, 1ull),
			_delta1 = svlsr_n_u64_x(all_b64, _delta1, 1ull)) 
		{
			svbool_t pg0 = svcmpne_n_u64(all_b64,
				svlsl_n_u64_x(all_b64, _delta0, 63ull), 
				0ull);

			svbool_t pg1 = svcmpne_n_u64(all_b64,
				svlsl_n_u64_x(all_b64, _delta1, 63ull), 
				0ull);

			_acc_mult0 = svmul_u64_m(pg0, _acc_mult0, _cur_mult);
			_acc_plus0 = svmad_u64_m(pg0, _acc_plus0, _cur_mult, _cur_plus);

			_acc_mult1 = svmul_u64_m(pg1, _acc_mult1, _cur_mult);
			_acc_plus1 = svmad_u64_m(pg1, _acc_plus1, _cur_mult, _cur_plus);

			_cur_plus = svmad_u64_z(all_b64, _cur_mult, _cur_plus, _cur_plus);
			_cur_mult = svmul_u64_z(all_b64, _cur_mult, _cur_mult);
		}
	}

	PCG_ALWAYS_INLINE svfloat32_t _make_float(svfloat32_t _low, svfloat32_t _range, svuint32_t _rand)
	{
		const svbool_t all_b32 = svptrue_b32();

		const svuint32_t _ldexpf32 = svdup_u32(0x70000000);
		const svuint32_t _signmask32 = svdup_u32(0x7fffffff);

		svfloat32_t _frand = svcvt_f32_u32_z(all_b32, _rand);

		svfloat32_t _grand = svreinterpret_f32_u32(
			svand_u32_z(all_b32,
				svadd_u32_z(all_b32,
					svreinterpret_u32_f32(_frand),
					_ldexpf32), _signmask32));

		svfloat32_t _hrand = svmad_f32_z(all_b32, _grand, _range, _low);

		return _hrand;
	}

	PCG_ALWAYS_INLINE svfloat64_t _make_double(svfloat64_t _low, svfloat64_t _range, svuint64_t _rand)
	{
		const svbool_t all_b64 = svptrue_b64();

		const svuint64_t _ldexp64 = svdup_u64(0x7c00000000000000ull);
		const svuint64_t _signmask64 = svdup_u64(0x7fffffffffffffffull);

		svfloat64_t _frand = svcvt_f64_u64_z(all_b64, _rand);

		svfloat64_t _grand = svreinterpret_f64_u64(
			svand_u64_z(all_b64,
				svadd_u64_z(all_b64,
					svreinterpret_u64_f64(_frand),
					_ldexp64), _signmask64));

		svfloat64_t _hrand = svmad_f64_z(all_b64, _grand, _range, _low);

		return _hrand;
	}

	PCG_ALWAYS_INLINE svuint32_t _svror_u32_z(svbool_t pg, const svuint32_t _a, const svuint32_t _b)
	{
		return svorr_u32_z(pg,
			svlsr_u32_z(pg, _a, _b),
			svlsl_u32_z(pg, 
				_a, 
				svsubr_n_u32_z(pg, _b, 32u)));
	}

	PCG_ALWAYS_INLINE svuint32_t _xsh_rr(const svuint64_t _state0, const svuint64_t _state1)
	{
		const svbool_t all_b32 = svptrue_b32();
		const svbool_t all_b64 = svptrue_b64();

		const svuint32_t _lo = svtrn1_u32(
			svreinterpret_u32_u64(svlsr_n_u64_z(all_b64, _state0, 27ull)),
			svreinterpret_u32_u64(svlsr_n_u64_z(all_b64, _state1, 27ull)));

		const svuint32_t _hi = svtrn2_u32(
			svreinterpret_u32_u64(_state0),
			svreinterpret_u32_u64(_state1));
	
		const svuint32_t _r = _svror_u32_z(all_b32,
			sveor_u32_z(all_b32,
				svlsr_n_u32_z(all_b32, _hi, 13u),
					_lo),
			svlsr_n_u32_z(all_b32, _hi, 27u));
		
		return _r;
	}

	template<>
	void fill(pcg_engines::setseq_xsh_rr_64_32& rng, float* ptr, size_t size, float low, float high)
	{
		const float range = high - low;

		const svfloat32_t _range = svdup_n_f32(range);
		const svfloat32_t _low = svdup_n_f32(low);

		uint64_t multiplier_, increment_, state_;
		scan_state(rng, multiplier_, increment_, state_);

		const svuint64_t _mult = svdup_n_u64(multiplier_);
		const svuint64_t _inc = svdup_n_u64(increment_);
		svuint64_t _state = svdup_n_u64(state_);

		svuint64_t _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1;

		_preadvance_twisted(_mult, _inc, _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1);

		const svbool_t all_b32 = svptrue_b32();
		const svbool_t all_b64 = svptrue_b64();

		const uint64_t tail = svcntd() - 1ull;
		const svbool_t pg_tail = svcmpeq_n_u64(all_b64, svindex_u64(0ull, 1ull), tail);
		
		svbool_t writemask;
		for (size_t i = 0; svptest_first(all_b32, writemask = svwhilelt_b32(i, size)); i += svcntw())
		{
			svuint64_t _state00 = svmad_u64_z(all_b64, _state, _acc_mult0, _acc_plus0);
			svuint64_t _state01 = svmad_u64_z(all_b64, _state, _acc_mult1, _acc_plus1);

			svuint64_t _advance = svdup_lane_u64(_state00, tail);
			_state00 = svsplice_u64(pg_tail, _state, _state00);
			_state = _advance;

			svuint32_t _rand = _xsh_rr(_state00, _state01);

			svfloat32_t _frand = _make_float(_low, _range, _rand);

			svst1_f32(writemask, ptr + i, _frand);
		}

		rng.advance(size);
	}

	template<>
	void fill(pcg_engines::setseq_xsh_rr_64_32& rng0, pcg_engines::setseq_xsh_rr_64_32& rng1, double* ptr, size_t size, double low, double high)
	{
		const double range = high - low;

		const svfloat64_t _range = svdup_n_f64(range);
		const svfloat64_t _low = svdup_n_f64(low);

		svuint64_t _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01;

		uint64_t multiplier0_, increment0_, state0_;
		scan_state(rng0, multiplier0_, increment0_, state0_);

		const svuint64_t _mult0 = svdup_n_u64(multiplier0_);
		const svuint64_t _inc0 = svdup_n_u64(increment0_);
		svuint64_t _state0 = svdup_n_u64(state0_);

		_preadvance_straight(_mult0, _inc0, _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01);

		svuint64_t _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11;

		uint64_t multiplier1_, increment1_, state1_;
		scan_state(rng1, multiplier1_, increment1_, state1_);

		const svuint64_t _mult1 = svdup_n_u64(multiplier1_);
		const svuint64_t _inc1 = svdup_n_u64(increment1_);
		svuint64_t _state1 = svdup_n_u64(state1_);

		_preadvance_straight(_mult1, _inc1, _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11);

		const svbool_t all_b64 = svptrue_b64();

		const uint64_t tail = svcntd() - 1ull;
		const svbool_t pg_tail = svcmpeq_n_u64(all_b64, svindex_u64(0ull, 1ull), tail);
		
		svbool_t writemask;
		for (size_t i = 0; svptest_first(all_b64, writemask = svwhilelt_b64(i, size)); i += svcntd())
		{
			svuint64_t _state00 = svmad_u64_z(all_b64, _state0, _acc_mult00, _acc_plus00);
			svuint64_t _state01 = svmad_u64_z(all_b64, _state0, _acc_mult01, _acc_plus01);

			svuint64_t _state10 = svmad_u64_z(all_b64, _state1, _acc_mult10, _acc_plus10);
			svuint64_t _state11 = svmad_u64_z(all_b64, _state1, _acc_mult11, _acc_plus11);

			svuint64_t _advance0 = svdup_lane_u64(_state01, tail);
			_state01 = svsplice_u64(pg_tail, _state00, _state01);
			_state00 = svsplice_u64(pg_tail, _state0, _state00);
			_state0 = _advance0;

			svuint64_t _advance1 = svdup_lane_u64(_state11, tail);
			_state11 = svsplice_u64(pg_tail, _state10, _state11);
			_state10 = svsplice_u64(pg_tail, _state1, _state10);
			_state1 = _advance1;

			svuint32_t _rand0 = _xsh_rr(_state00, _state01);

			svuint32_t _rand1 = _xsh_rr(_state10, _state11);

			svfloat64_t _frand0 = _make_double(_low, _range,
				svreinterpret_u64_u32(svtrn1_u32(_rand0, _rand1)));

			svfloat64_t _frand1 = _make_double(_low, _range,
				svreinterpret_u64_u32(svtrn2_u32(_rand0, _rand1)));

			svst1_f64(writemask, ptr + i, _frand0);

			i += svcntd();
			writemask = svwhilelt_b64(i, size);

			svst1_f64(writemask, ptr + i, _frand1);
		}

		rng0.advance(size);
		rng1.advance(size);
	}

	PCG_ALWAYS_INLINE svuint32_t _xsh_rs(const svuint64_t _state0, const svuint64_t _state1)
	{
		const svbool_t all_b64 = svptrue_b64();

		const svuint64_t _rand0 = svlsr_u64_z(all_b64,
			sveor_u64_z(all_b64,
				svlsr_n_u64_z(all_b64, _state0, 44ull),
				svlsr_n_u64_z(all_b64, _state0, 22ull)),
			svlsr_n_u64_z(all_b64, _state0, 61ull));

		const svuint64_t _rand1 = svlsr_u64_z(all_b64,
			sveor_u64_z(all_b64,
				svlsr_n_u64_z(all_b64, _state1, 44ull),
				svlsr_n_u64_z(all_b64, _state1, 22ull)),
			svlsr_n_u64_z(all_b64, _state1, 61ull));

		return svtrn1_u32(
			svreinterpret_u32_u64(_rand0), 
			svreinterpret_u32_u64(_rand1));
	}

	template<>
	void fill(pcg_engines::setseq_xsh_rs_64_32& rng, float* ptr, size_t size, float low, float high)
	{
		const float range = high - low;

		const svfloat32_t _range = svdup_n_f32(range);
		const svfloat32_t _low = svdup_n_f32(low);

		uint64_t multiplier_, increment_, state_;
		scan_state(rng, multiplier_, increment_, state_);

		const svuint64_t _mult = svdup_n_u64(multiplier_);
		const svuint64_t _inc = svdup_n_u64(increment_);
		svuint64_t _state = svdup_n_u64(state_);

		svuint64_t _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1;

		_preadvance_twisted(_mult, _inc, _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1);

		const svbool_t all_b32 = svptrue_b32();
		const svbool_t all_b64 = svptrue_b64();

		const uint64_t tail = svcntd() - 1ull;
		const svbool_t pg_tail = svcmpeq_n_u64(all_b64, svindex_u64(0ull, 1ull), tail);

		svbool_t writemask;
		for (size_t i = 0; svptest_first(all_b32, writemask = svwhilelt_b32(i, size)); i += svcntw())
		{
			svuint64_t _state00 = svmad_u64_z(all_b64, _state, _acc_mult0, _acc_plus0);
			svuint64_t _state01 = svmad_u64_z(all_b64, _state, _acc_mult1, _acc_plus1);

			svuint64_t _advance = svdup_lane_u64(_state00, tail);
			_state00 = svsplice_u64(pg_tail, _state, _state00);
			_state = _advance;

			svuint32_t _rand = _xsh_rs(_state00, _state01);

			svfloat32_t _frand = _make_float(_low, _range, _rand);

			svst1_f32(writemask, ptr + i, _frand);
		}

		rng.advance(size);
	}

	template<>
	void fill(pcg_engines::setseq_xsh_rs_64_32& rng0, pcg_engines::setseq_xsh_rs_64_32& rng1, double* ptr, size_t size, double low, double high)
	{
		const double range = high - low;

		const svfloat64_t _range = svdup_n_f64(range);
		const svfloat64_t _low = svdup_n_f64(low);

		svuint64_t _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01;

		uint64_t multiplier0_, increment0_, state0_;
		scan_state(rng0, multiplier0_, increment0_, state0_);

		const svuint64_t _mult0 = svdup_n_u64(multiplier0_);
		const svuint64_t _inc0 = svdup_n_u64(increment0_);
		svuint64_t _state0 = svdup_n_u64(state0_);

		_preadvance_straight(_mult0, _inc0, _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01);

		svuint64_t _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11;

		uint64_t multiplier1_, increment1_, state1_;
		scan_state(rng1, multiplier1_, increment1_, state1_);

		const svuint64_t _mult1 = svdup_n_u64(multiplier1_);
		const svuint64_t _inc1 = svdup_n_u64(increment1_);
		svuint64_t _state1 = svdup_n_u64(state1_);

		_preadvance_straight(_mult1, _inc1, _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11);

		const svbool_t all_b64 = svptrue_b64();

		const uint64_t tail = svcntd() - 1ull;
		const svbool_t pg_tail = svcmpeq_n_u64(all_b64, svindex_u64(0ull, 1ull), tail);
		
		svbool_t writemask;
		for (size_t i = 0; svptest_first(all_b64, writemask = svwhilelt_b64(i, size)); i += svcntd())
		{
			svuint64_t _state00 = svmad_u64_z(all_b64, _state0, _acc_mult00, _acc_plus00);
			svuint64_t _state01 = svmad_u64_z(all_b64, _state0, _acc_mult01, _acc_plus01);

			svuint64_t _state10 = svmad_u64_z(all_b64, _state1, _acc_mult10, _acc_plus10);
			svuint64_t _state11 = svmad_u64_z(all_b64, _state1, _acc_mult11, _acc_plus11);

			svuint64_t _advance0 = svdup_lane_u64(_state01, tail);
			_state01 = svsplice_u64(pg_tail, _state00, _state01);
			_state00 = svsplice_u64(pg_tail, _state0, _state00);
			_state0 = _advance0;

			svuint64_t _advance1 = svdup_lane_u64(_state11, tail);
			_state11 = svsplice_u64(pg_tail, _state10, _state11);
			_state10 = svsplice_u64(pg_tail, _state1, _state10);
			_state1 = _advance1;

			svuint32_t _rand0 = _xsh_rs(_state00, _state01);

			svuint32_t _rand1 = _xsh_rs(_state10, _state11);

			svfloat64_t _frand0 = _make_double(_low, _range,
				svreinterpret_u64_u32(svtrn1_u32(_rand0, _rand1)));

			svfloat64_t _frand1 = _make_double(_low, _range,
				svreinterpret_u64_u32(svtrn2_u32(_rand0, _rand1)));

			svst1_f64(writemask, ptr + i, _frand0);

			i += svcntd();
			writemask = svwhilelt_b64(i, size);

			svst1_f64(writemask, ptr + i, _frand1);
		}

		rng0.advance(size);
		rng1.advance(size);
	}

	PCG_ALWAYS_INLINE svuint32_t _rxs_m(const svuint64_t _state0, const svuint64_t _state1)
	{
		const svbool_t all_b64 = svptrue_b64();

		const svuint64_t _multiplier = svdup_n_u64(pcg_detail::mcg_multiplier<uint64_t>::multiplier());

		const svuint64_t _rand0 = svmul_u64_z(all_b64,
			sveor_u64_z(all_b64,
				svlsr_u64_z(all_b64,
					svlsr_n_u64_z(all_b64, _state0, 4ull),
					svlsr_n_u64_z(all_b64, _state0, 60ull)),
				_state0),
			_multiplier);

		const svuint64_t _rand1 = svmul_u64_z(all_b64,
			sveor_u64_z(all_b64,
				svlsr_u64_z(all_b64,
					svlsr_n_u64_z(all_b64, _state1, 4ull),
					svlsr_n_u64_z(all_b64, _state1, 60ull)),
				_state1),
			_multiplier);

		return svtrn2_u32(
			svreinterpret_u32_u64(_rand0),
			svreinterpret_u32_u64(_rand1));
	}

	template<>
	void fill(pcg_engines::setseq_rxs_m_64_32& rng, float* ptr, size_t size, float low, float high)
	{
		const float range = high - low;

		const svfloat32_t _range = svdup_n_f32(range);
		const svfloat32_t _low = svdup_n_f32(low);

		uint64_t multiplier_, increment_, state_;
		scan_state(rng, multiplier_, increment_, state_);

		const svuint64_t _mult = svdup_n_u64(multiplier_);
		const svuint64_t _inc = svdup_n_u64(increment_);
		svuint64_t _state = svdup_n_u64(state_);

		svuint64_t _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1;

		_preadvance_twisted(_mult, _inc, _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1);

		const svbool_t all_b32 = svptrue_b32();
		const svbool_t all_b64 = svptrue_b64();

		const uint64_t tail = svcntd() - 1ull;
		const svbool_t pg_tail = svcmpeq_n_u64(all_b64, svindex_u64(0ull, 1ull), tail);

		svbool_t writemask;
		for (size_t i = 0; svptest_first(all_b32, writemask = svwhilelt_b32(i, size)); i += svcntw())
		{
			svuint64_t _state00 = svmad_u64_z(all_b64, _state, _acc_mult0, _acc_plus0);
			svuint64_t _state01 = svmad_u64_z(all_b64, _state, _acc_mult1, _acc_plus1);

			svuint64_t _advance = svdup_lane_u64(_state00, tail);
			_state00 = svsplice_u64(pg_tail, _state, _state00);
			_state = _advance;

			svuint32_t _rand = _rxs_m(_state00, _state01);

			svfloat32_t _frand = _make_float(_low, _range, _rand);

			svst1_f32(writemask, ptr + i, _frand);
		}

		rng.advance(size);
	}

	template<>
	void fill(pcg_engines::setseq_rxs_m_64_32& rng0, pcg_engines::setseq_rxs_m_64_32& rng1, double* ptr, size_t size, double low, double high)
	{
		const double range = high - low;

		const svfloat64_t _range = svdup_n_f64(range);
		const svfloat64_t _low = svdup_n_f64(low);

		svuint64_t _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01;

		uint64_t multiplier0_, increment0_, state0_;
		scan_state(rng0, multiplier0_, increment0_, state0_);

		const svuint64_t _mult0 = svdup_n_u64(multiplier0_);
		const svuint64_t _inc0 = svdup_n_u64(increment0_);
		svuint64_t _state0 = svdup_n_u64(state0_);

		_preadvance_straight(_mult0, _inc0, _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01);

		svuint64_t _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11;

		uint64_t multiplier1_, increment1_, state1_;
		scan_state(rng1, multiplier1_, increment1_, state1_);

		const svuint64_t _mult1 = svdup_n_u64(multiplier1_);
		const svuint64_t _inc1 = svdup_n_u64(increment1_);
		svuint64_t _state1 = svdup_n_u64(state1_);

		_preadvance_straight(_mult1, _inc1, _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11);

		const svbool_t all_b64 = svptrue_b64();

		const uint64_t tail = svcntd() - 1ull;
		const svbool_t pg_tail = svcmpeq_n_u64(all_b64, svindex_u64(0ull, 1ull), tail);

		svbool_t writemask;
		for (size_t i = 0; svptest_first(all_b64, writemask = svwhilelt_b64(i, size)); i += svcntd())
		{
			svuint64_t _state00 = svmad_u64_z(all_b64, _state0, _acc_mult00, _acc_plus00);
			svuint64_t _state01 = svmad_u64_z(all_b64, _state0, _acc_mult01, _acc_plus01);

			svuint64_t _state10 = svmad_u64_z(all_b64, _state1, _acc_mult10, _acc_plus10);
			svuint64_t _state11 = svmad_u64_z(all_b64, _state1, _acc_mult11, _acc_plus11);

			svuint64_t _advance0 = svdup_lane_u64(_state01, tail);
			_state01 = svsplice_u64(pg_tail, _state00, _state01);
			_state00 = svsplice_u64(pg_tail, _state0, _state00);
			_state0 = _advance0;

			svuint64_t _advance1 = svdup_lane_u64(_state11, tail);
			_state11 = svsplice_u64(pg_tail, _state10, _state11);
			_state10 = svsplice_u64(pg_tail, _state1, _state10);
			_state1 = _advance1;

			svuint32_t _rand0 = _rxs_m(_state00, _state01);

			svuint32_t _rand1 = _rxs_m(_state10, _state11);

			svfloat64_t _frand0 = _make_double(_low, _range,
				svreinterpret_u64_u32(svtrn1_u32(_rand0, _rand1)));

			svfloat64_t _frand1 = _make_double(_low, _range,
				svreinterpret_u64_u32(svtrn2_u32(_rand0, _rand1)));

			svst1_f64(writemask, ptr + i, _frand0);

			i += svcntd();
			writemask = svwhilelt_b64(i, size);

			svst1_f64(writemask, ptr + i, _frand1);
		}

		rng0.advance(size);
		rng1.advance(size);
	}

	PCG_ALWAYS_INLINE svuint32_t _dxsm(const svuint64_t _state0, const svuint64_t _state1)
	{
		const svbool_t all_b32 = svptrue_b32();

		const svuint32_t _multiplier = svdup_n_u32(static_cast<uint32_t>(pcg_detail::cheap_multiplier<uint64_t>::multiplier()));

		const svuint32_t _lo = svorr_n_u32_z(all_b32,
			svtrn1_u32(
				svreinterpret_u32_u64(_state0),
				svreinterpret_u32_u64(_state1)),
			1u);

		svuint32_t _hi = svtrn2_u32(
			svreinterpret_u32_u64(_state0),
			svreinterpret_u32_u64(_state1));

		_hi = sveor_u32_z(all_b32, 
			svlsr_n_u32_z(all_b32, _hi, 16u), 
			_hi);

		_hi = svmul_u32_z(all_b32,
			_hi,
			_multiplier);

		_hi = sveor_u32_z(all_b32,
			svlsr_n_u32_z(all_b32, _hi, 24u),
			_hi);

		_hi = svmul_u32_z(all_b32, _hi, _lo);

		return _hi;
	}

	template<>
	void fill(pcg_engines::setseq_dxsm_64_32& rng, float* ptr, size_t size, float low, float high)
	{
		const float range = high - low;

		const svfloat32_t _range = svdup_n_f32(range);
		const svfloat32_t _low = svdup_n_f32(low);

		uint64_t multiplier_, increment_, state_;
		scan_state(rng, multiplier_, increment_, state_);

		const svuint64_t _mult = svdup_n_u64(multiplier_);
		const svuint64_t _inc = svdup_n_u64(increment_);
		svuint64_t _state = svdup_n_u64(state_);

		svuint64_t _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1;

		_preadvance_twisted(_mult, _inc, _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1);

		const svbool_t all_b32 = svptrue_b32();
		const svbool_t all_b64 = svptrue_b64();

		const uint64_t tail = svcntd() - 1ull;
		const svbool_t pg_tail = svcmpeq_n_u64(all_b64, svindex_u64(0ull, 1ull), tail);

		svbool_t writemask;
		for (size_t i = 0; svptest_first(all_b32, writemask = svwhilelt_b32(i, size)); i += svcntw())
		{
			svuint64_t _state00 = svmad_u64_z(all_b64, _state, _acc_mult0, _acc_plus0);
			svuint64_t _state01 = svmad_u64_z(all_b64, _state, _acc_mult1, _acc_plus1);

			svuint64_t _advance = svdup_lane_u64(_state00, tail);
			_state00 = svsplice_u64(pg_tail, _state, _state00);
			_state = _advance;

			svuint32_t _rand = _dxsm(_state00, _state01);

			svfloat32_t _frand = _make_float(_low, _range, _rand);

			svst1_f32(writemask, ptr + i, _frand);
		}

		rng.advance(size);
	}

	template<>
	void fill(pcg_engines::setseq_dxsm_64_32& rng0, pcg_engines::setseq_dxsm_64_32& rng1, double* ptr, size_t size, double low, double high)
	{
		const double range = high - low;

		const svfloat64_t _range = svdup_n_f64(range);
		const svfloat64_t _low = svdup_n_f64(low);

		svuint64_t _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01;

		uint64_t multiplier0_, increment0_, state0_;
		scan_state(rng0, multiplier0_, increment0_, state0_);

		const svuint64_t _mult0 = svdup_n_u64(multiplier0_);
		const svuint64_t _inc0 = svdup_n_u64(increment0_);
		svuint64_t _state0 = svdup_n_u64(state0_);

		_preadvance_straight(_mult0, _inc0, _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01);

		svuint64_t _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11;

		uint64_t multiplier1_, increment1_, state1_;
		scan_state(rng1, multiplier1_, increment1_, state1_);

		const svuint64_t _mult1 = svdup_n_u64(multiplier1_);
		const svuint64_t _inc1 = svdup_n_u64(increment1_);
		svuint64_t _state1 = svdup_n_u64(state1_);

		_preadvance_straight(_mult1, _inc1, _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11);

		const svbool_t all_b64 = svptrue_b64();

		const uint64_t tail = svcntd() - 1ull;
		const svbool_t pg_tail = svcmpeq_n_u64(all_b64, svindex_u64(0ull, 1ull), tail);

		svbool_t writemask;
		for (size_t i = 0; svptest_first(all_b64, writemask = svwhilelt_b64(i, size)); i += svcntd())
		{
			svuint64_t _state00 = svmad_u64_z(all_b64, _state0, _acc_mult00, _acc_plus00);
			svuint64_t _state01 = svmad_u64_z(all_b64, _state0, _acc_mult01, _acc_plus01);

			svuint64_t _state10 = svmad_u64_z(all_b64, _state1, _acc_mult10, _acc_plus10);
			svuint64_t _state11 = svmad_u64_z(all_b64, _state1, _acc_mult11, _acc_plus11);

			svuint64_t _advance0 = svdup_lane_u64(_state01, tail);
			_state01 = svsplice_u64(pg_tail, _state00, _state01);
			_state00 = svsplice_u64(pg_tail, _state0, _state00);
			_state0 = _advance0;

			svuint64_t _advance1 = svdup_lane_u64(_state11, tail);
			_state11 = svsplice_u64(pg_tail, _state10, _state11);
			_state10 = svsplice_u64(pg_tail, _state1, _state10);
			_state1 = _advance1;

			svuint32_t _rand0 = _dxsm(_state00, _state01);

			svuint32_t _rand1 = _dxsm(_state10, _state11);

			svfloat64_t _frand0 = _make_double(_low, _range,
				svreinterpret_u64_u32(svtrn1_u32(_rand0, _rand1)));

			svfloat64_t _frand1 = _make_double(_low, _range,
				svreinterpret_u64_u32(svtrn2_u32(_rand0, _rand1)));

			svst1_f64(writemask, ptr + i, _frand0);

			i += svcntd();
			writemask = svwhilelt_b64(i, size);

			svst1_f64(writemask, ptr + i, _frand1);
		}

		rng0.advance(size);
		rng1.advance(size);
	}

	PCG_ALWAYS_INLINE svuint32_t _xsl_rr(const svuint64_t _state0, const svuint64_t _state1)
	{
		const svbool_t all_b32 = svptrue_b32();

		const svuint32_t _lo = svtrn1_u32(
			svreinterpret_u32_u64(_state0),
			svreinterpret_u32_u64(_state1));

		const svuint32_t _hi = svtrn2_u32(
			svreinterpret_u32_u64(_state0),
			svreinterpret_u32_u64(_state1));

		return _svror_u32_z(all_b32,
			sveor_u32_z(all_b32, _hi, _lo),
			svlsr_n_u32_z(all_b32, _hi, 27));
	}

	template<>
	void fill(pcg_engines::setseq_xsl_rr_64_32& rng, float* ptr, size_t size, float low, float high)
	{
		const float range = high - low;

		const svfloat32_t _range = svdup_n_f32(range);
		const svfloat32_t _low = svdup_n_f32(low);

		uint64_t multiplier_, increment_, state_;
		scan_state(rng, multiplier_, increment_, state_);

		const svuint64_t _mult = svdup_n_u64(multiplier_);
		const svuint64_t _inc = svdup_n_u64(increment_);
		svuint64_t _state = svdup_n_u64(state_);

		svuint64_t _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1;

		_preadvance_twisted(_mult, _inc, _acc_mult0, _acc_plus0, _acc_mult1, _acc_plus1);

		const svbool_t all_b32 = svptrue_b32();
		const svbool_t all_b64 = svptrue_b64();

		const uint64_t tail = svcntd() - 1ull;
		const svbool_t pg_tail = svcmpeq_n_u64(all_b64, svindex_u64(0ull, 1ull), tail);

		svbool_t writemask;
		for (size_t i = 0; svptest_first(all_b32, writemask = svwhilelt_b32(i, size)); i += svcntw())
		{
			svuint64_t _state00 = svmad_u64_z(all_b64, _state, _acc_mult0, _acc_plus0);
			svuint64_t _state01 = svmad_u64_z(all_b64, _state, _acc_mult1, _acc_plus1);

			svuint64_t _advance = svdup_lane_u64(_state00, tail);
			_state00 = svsplice_u64(pg_tail, _state, _state00);
			_state = _advance;

			svuint32_t _rand = _xsl_rr(_state00, _state01);

			svfloat32_t _frand = _make_float(_low, _range, _rand);

			svst1_f32(writemask, ptr + i, _frand);
		}

		rng.advance(size);
	}

	template<>
	void fill(pcg_engines::setseq_xsl_rr_64_32& rng0, pcg_engines::setseq_xsl_rr_64_32& rng1, double* ptr, size_t size, double low, double high)
	{
		const double range = high - low;

		const svfloat64_t _range = svdup_n_f64(range);
		const svfloat64_t _low = svdup_n_f64(low);

		svuint64_t _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01;

		uint64_t multiplier0_, increment0_, state0_;
		scan_state(rng0, multiplier0_, increment0_, state0_);

		const svuint64_t _mult0 = svdup_n_u64(multiplier0_);
		const svuint64_t _inc0 = svdup_n_u64(increment0_);
		svuint64_t _state0 = svdup_n_u64(state0_);

		_preadvance_straight(_mult0, _inc0, _acc_mult00, _acc_plus00, _acc_mult01, _acc_plus01);

		svuint64_t _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11;

		uint64_t multiplier1_, increment1_, state1_;
		scan_state(rng1, multiplier1_, increment1_, state1_);

		const svuint64_t _mult1 = svdup_n_u64(multiplier1_);
		const svuint64_t _inc1 = svdup_n_u64(increment1_);
		svuint64_t _state1 = svdup_n_u64(state1_);

		_preadvance_straight(_mult1, _inc1, _acc_mult10, _acc_plus10, _acc_mult11, _acc_plus11);

		const svbool_t all_b64 = svptrue_b64();

		const uint64_t tail = svcntd() - 1ull;
		const svbool_t pg_tail = svcmpeq_n_u64(all_b64, svindex_u64(0ull, 1ull), tail);

		svbool_t writemask;
		for (size_t i = 0; svptest_first(all_b64, writemask = svwhilelt_b64(i, size)); i += svcntd())
		{
			svuint64_t _state00 = svmad_u64_z(all_b64, _state0, _acc_mult00, _acc_plus00);
			svuint64_t _state01 = svmad_u64_z(all_b64, _state0, _acc_mult01, _acc_plus01);

			svuint64_t _state10 = svmad_u64_z(all_b64, _state1, _acc_mult10, _acc_plus10);
			svuint64_t _state11 = svmad_u64_z(all_b64, _state1, _acc_mult11, _acc_plus11);

			svuint64_t _advance0 = svdup_lane_u64(_state01, tail);
			_state01 = svsplice_u64(pg_tail, _state00, _state01);
			_state00 = svsplice_u64(pg_tail, _state0, _state00);
			_state0 = _advance0;

			svuint64_t _advance1 = svdup_lane_u64(_state11, tail);
			_state11 = svsplice_u64(pg_tail, _state10, _state11);
			_state10 = svsplice_u64(pg_tail, _state1, _state10);
			_state1 = _advance1;

			svuint32_t _rand0 = _xsl_rr(_state00, _state01);

			svuint32_t _rand1 = _xsl_rr(_state10, _state11);

			svfloat64_t _frand0 = _make_double(_low, _range,
				svreinterpret_u64_u32(svtrn1_u32(_rand0, _rand1)));

			svfloat64_t _frand1 = _make_double(_low, _range,
				svreinterpret_u64_u32(svtrn2_u32(_rand0, _rand1)));

			svst1_f64(writemask, ptr + i, _frand0);

			i += svcntd();
			writemask = svwhilelt_b64(i, size);

			svst1_f64(writemask, ptr + i, _frand1);
		}

		rng0.advance(size);
		rng1.advance(size);
	}
}

#define PCG_FILL_SVE2_HPP_INCLUDED 1
#endif // PCG_FILL_SVE2_HPP_INCLUDED
