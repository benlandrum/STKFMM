#ifndef INCLUDE_RPYCUSTOMKERNEL_H_
#define INCLUDE_RPYCUSTOMKERNEL_H_

#include <cmath>
#include <cstdlib>
#include <vector>

#include "stkfmm_helpers.hpp"

namespace pvfmm {

/**********************************************************
 *                                                        *
 *     RPY velocity kernel, source: 4, target: 3          *
 *           fx,fy,fz,a -> ux,uy,uz                       *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void rpy_u_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                   Matrix<Real_t> &trg_value) {

    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    Vec_t FACV = set_intrin<Vec_t, Real_t>(1.0 / (8 * const_pi<Real_t>()));
    Vec_t nwtn_factor = set_intrin<Vec_t, Real_t>(1.0 / nwtn_scal);
    Vec_t three = set_intrin<Vec_t, Real_t>(3.0);

    const Vec_t one_over_three = set_intrin<Vec_t, Real_t>(static_cast<Real_t>(1.0 / 3.0));

    const size_t src_cnt_ = src_coord.Dim(1);
    const size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>();
            Vec_t vy = zero_intrin<Vec_t>();
            Vec_t vz = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = tx - bcast_intrin<Vec_t>(&src_coord[0][s]);
                const Vec_t dy = ty - bcast_intrin<Vec_t>(&src_coord[1][s]);
                const Vec_t dz = tz - bcast_intrin<Vec_t>(&src_coord[2][s]);

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                const Vec_t a = bcast_intrin<Vec_t>(&src_value[3][s]);

                const Vec_t a2_over_three = mul_intrin(mul_intrin(a, a), one_over_three);
                const Vec_t r2 = add_intrin(add_intrin(mul_intrin(dx, dx), mul_intrin(dy, dy)), mul_intrin(dz, dz));

                const Vec_t rinv = mul_intrin(rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2), nwtn_factor);
                const Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                const Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);
                const Vec_t fdotr = add_intrin(add_intrin(mul_intrin(fx, dx), mul_intrin(fy, dy)), mul_intrin(fz, dz));

                vx = add_intrin(vx, mul_intrin(add_intrin(mul_intrin(r2, fx), mul_intrin(dx, fdotr)), rinv3));
                vy = add_intrin(vy, mul_intrin(add_intrin(mul_intrin(r2, fy), mul_intrin(dy, fdotr)), rinv3));
                vz = add_intrin(vz, mul_intrin(add_intrin(mul_intrin(r2, fz), mul_intrin(dz, fdotr)), rinv3));
                const Vec_t three_fdotr_rinv5 = mul_intrin(mul_intrin(three, fdotr), rinv5);

                vx = add_intrin(vx, mul_intrin(a2_over_three,
                                               sub_intrin(mul_intrin(fx, rinv3), mul_intrin(three_fdotr_rinv5, dx))));
                vy = add_intrin(vy, mul_intrin(a2_over_three,
                                               sub_intrin(mul_intrin(fy, rinv3), mul_intrin(three_fdotr_rinv5, dy))));
                vz = add_intrin(vz, mul_intrin(a2_over_three,
                                               sub_intrin(mul_intrin(fz, rinv3), mul_intrin(three_fdotr_rinv5, dz))));
            }

            vx = add_intrin(mul_intrin(vx, FACV), load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, FACV), load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, FACV), load_intrin<Vec_t>(&trg_value[2][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
        }
    }
}

/**********************************************************
 *                                                        *
 * RPY Force,a Vel,lapVel kernel, source: 4, target: 6    *
 *       fx,fy,fz,a -> ux,uy,uz,lapux,lapuy,lapuz         *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void rpy_ulapu_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                       Matrix<Real_t> &trg_value) {

    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Vec_t FACV = set_intrin<Vec_t, Real_t>(1.0 / (8 * const_pi<Real_t>()));
    const Vec_t nwtn_factor = set_intrin<Vec_t, Real_t>(1.0 / nwtn_scal);
    const Vec_t one_over_three = set_intrin<Vec_t, Real_t>(1.0 / 3.0);
    const Vec_t two = set_intrin<Vec_t, Real_t>(2.0);
    const Vec_t three = set_intrin<Vec_t, Real_t>(3.0);
    const size_t src_cnt_ = src_coord.Dim(1);
    const size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>();
            Vec_t vy = zero_intrin<Vec_t>();
            Vec_t vz = zero_intrin<Vec_t>();
            Vec_t lapvx = zero_intrin<Vec_t>();
            Vec_t lapvy = zero_intrin<Vec_t>();
            Vec_t lapvz = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = tx - bcast_intrin<Vec_t>(&src_coord[0][s]);
                const Vec_t dy = ty - bcast_intrin<Vec_t>(&src_coord[1][s]);
                const Vec_t dz = tz - bcast_intrin<Vec_t>(&src_coord[2][s]);

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                const Vec_t a = bcast_intrin<Vec_t>(&src_value[3][s]);

                const Vec_t a2_over_three = mul_intrin(mul_intrin(a, a), one_over_three);
                const Vec_t r2 = add_intrin(add_intrin(mul_intrin(dx, dx), mul_intrin(dy, dy)), mul_intrin(dz, dz));

                const Vec_t rinv = mul_intrin(rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2), nwtn_factor);
                const Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                const Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);
                const Vec_t fdotr = add_intrin(add_intrin(mul_intrin(fx, dx), mul_intrin(fy, dy)), mul_intrin(fz, dz));

                const Vec_t three_fdotr_rinv5 = mul_intrin(mul_intrin(three, fdotr), rinv5);
                const Vec_t cx = sub_intrin(mul_intrin(fx, rinv3), mul_intrin(three_fdotr_rinv5, dx));
                const Vec_t cy = sub_intrin(mul_intrin(fy, rinv3), mul_intrin(three_fdotr_rinv5, dy));
                const Vec_t cz = sub_intrin(mul_intrin(fz, rinv3), mul_intrin(three_fdotr_rinv5, dz));

                const Vec_t fdotr_rinv3 = mul_intrin(fdotr, rinv3);
                vx = add_intrin(vx, add_intrin(mul_intrin(fx, rinv),
                                               add_intrin(mul_intrin(dx, fdotr_rinv3), mul_intrin(a2_over_three, cx))));
                vy = add_intrin(vy, add_intrin(mul_intrin(fy, rinv),
                                               add_intrin(mul_intrin(dy, fdotr_rinv3), mul_intrin(a2_over_three, cy))));
                vz = add_intrin(vz, add_intrin(mul_intrin(fz, rinv),
                                               add_intrin(mul_intrin(dz, fdotr_rinv3), mul_intrin(a2_over_three, cz))));

                lapvx = add_intrin(lapvx, mul_intrin(two, cx));
                lapvy = add_intrin(lapvy, mul_intrin(two, cy));
                lapvz = add_intrin(lapvz, mul_intrin(two, cz));
            }

            vx = add_intrin(mul_intrin(vx, FACV), load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, FACV), load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, FACV), load_intrin<Vec_t>(&trg_value[2][t]));
            lapvx = add_intrin(mul_intrin(lapvx, FACV), load_intrin<Vec_t>(&trg_value[3][t]));
            lapvy = add_intrin(mul_intrin(lapvy, FACV), load_intrin<Vec_t>(&trg_value[4][t]));
            lapvz = add_intrin(mul_intrin(lapvz, FACV), load_intrin<Vec_t>(&trg_value[5][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
            store_intrin(&trg_value[3][t], lapvx);
            store_intrin(&trg_value[4][t], lapvy);
            store_intrin(&trg_value[5][t], lapvz);
        }
    }
}

/***********************************************************
 *                                                         *
 * Regularized RPY sphere to sphere, source: 4, target: 4  *
 *       fx,fy,fz,a -> ux,uy,uz,a                          *
 ***********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void rpy_reg_sphere_to_sphere_uKernel(Matrix<Real_t> &src_coord,
				      Matrix<Real_t> &src_value,
				      Matrix<Real_t> &trg_coord,
				      Matrix<Real_t> &trg_value) {

    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Vec_t FACV = set_intrin<Vec_t, Real_t>(1.0 / (8 * const_pi<Real_t>()));
    const Vec_t nwtn_factor = set_intrin<Vec_t, Real_t>(1.0 / nwtn_scal);
    const Vec_t one_over_two = set_intrin<Vec_t, Real_t>(1.0 / 2.0);
    const Vec_t one_over_three = set_intrin<Vec_t, Real_t>(1.0 / 3.0);
    const Vec_t one_over_six = set_intrin<Vec_t, Real_t>(1.0 / 6.0);
    const Vec_t one_over_thirtytwo = set_intrin<Vec_t, Real_t>(1.0 / 32.0);
    const Vec_t four_over_three = set_intrin<Vec_t, Real_t>(4.0 / 3.0);
    const Vec_t two = set_intrin<Vec_t, Real_t>(2.0);
    const Vec_t three = set_intrin<Vec_t, Real_t>(3.0);
    const size_t src_cnt_ = src_coord.Dim(1);
    const size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);
	    const Vec_t b  = load_intrin<Vec_t>(&trg_coord[3][t]);
	    const Vec_t b2 = mul_intrin(b, b);

            Vec_t vx = zero_intrin<Vec_t>();
            Vec_t vy = zero_intrin<Vec_t>();
            Vec_t vz = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = tx - bcast_intrin<Vec_t>(&src_coord[0][s]);
                const Vec_t dy = ty - bcast_intrin<Vec_t>(&src_coord[1][s]);
                const Vec_t dz = tz - bcast_intrin<Vec_t>(&src_coord[2][s]);
		const Vec_t r2 = add_intrin(add_intrin(mul_intrin(dx, dx), mul_intrin(dy, dy)), mul_intrin(dz, dz));
		const Vec_t rinv = mul_intrin(rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2), nwtn_factor);
                const Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                const Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
		const Vec_t fdotr = add_intrin(add_intrin(mul_intrin(fx, dx), mul_intrin(fy, dy)), mul_intrin(fz, dz));
		const Vec_t fdotr_rinv3 = mul_intrin(fdotr, rinv3);
		const Vec_t fdotr_rinv5 = mul_intrin(fdotr, rinv5);

                const Vec_t a           = bcast_intrin<Vec_t>(&src_value[3][s]); ////
		const Vec_t a2          = mul_intrin(a, a); ////
		const Vec_t a_add_b     = add_intrin(a, b);
		const Vec_t a_add_b_2   = mul_intrin(a_add_b, a_add_b);
		const Vec_t a_sub_b     = sub_intrin(a, b);
		const Vec_t a_sub_b_2   = mul_intrin(a_sub_b, a_sub_b);

		// Scenario: r <= |a - b|
		// Set unconditionally; override later.
		{
		  const Vec_t max_a2_b2 = max_intrin(a2, b2);
		  const Vec_t radinv = mul_intrin(rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(max_a2_b2), nwtn_factor);
		  vx = mul_intrin(four_over_three,
				  mul_intrin(radinv, fx));
		  vy = mul_intrin(four_over_three,
				  mul_intrin(radinv, fy));
		  vz = mul_intrin(four_over_three,
				  mul_intrin(radinv, fz));
		}

		// Scenario: |a - b| < r <= a + b
		// Set if |a - b| < r; override later.
		{
		  // TODO: Why not just a division intrin?
		  // TODO: Revisit these operations for extra multiplications (ifactor).
		  const Vec_t a2b2       = mul_intrin(a2, b2);
		  const Vec_t abinv      = mul_intrin(rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(a2b2), nwtn_factor);
		  const Vec_t front      = mul_intrin(four_over_three, abinv);
		  const Vec_t isub_sqrt  = add_intrin(a_sub_b_2,
						      mul_intrin(three, r2));
		  Vec_t ifactor          = sub_intrin(mul_intrin(one_over_two, a_add_b),
						      mul_intrin(mul_intrin(one_over_thirtytwo, rinv3),
							 mul_intrin(isub_sqrt, isub_sqrt)));
		  ifactor                = mul_intrin(ifactor, front);

		  Vec_t vx_trial = mul_intrin(ifactor, fx);
		  Vec_t vy_trial = mul_intrin(ifactor, fy);
		  Vec_t vz_trial = mul_intrin(ifactor, fz);

		  const Vec_t xxsub_sqrt = sub_intrin(a_sub_b_2, r2);
		  Vec_t xxfactor = mul_intrin(three_over_thirtytwo,
					      mul_intrin(xxsub_sqrt, xxsub_sqrt));
		  xxfactor = mul_intrin(xxfactor, fdotr_rinv5);
		  vx_trial = add_intrin(vx_trial, mul_intrin(dx, xxfactor));
		  vy_trial = add_intrin(vy_trial, mul_intrin(dy, xxfactor));
		  vz_trial = add_intrin(vz_trial, mul_intrin(dz, xxfactor));

		  const Vec_t sub_lt_r = cmplt_intrin(a_sub_b_2, r2);
		  vx = blend_intrin(vx, vx_trial, sub_lt_r);
		  vy = blend_intrin(vy, vy_trial, sub_lt_r);
		  vz = blend_intrin(vz, vz_trial, sub_lt_r);
		}

		// Scenario: a + b < r
		// Set if a + b < r;
		{
		  const Vec_t a2_add_b2 = add_intrin(a2, b2);
		  Vec_t vx_trial = add_intrin(mul_intrin(fx, rinv),
					 mul_intrin(mul_intrin(one_over_three, a2_add_a2),
						    mul_intrin(fx, rinv3)));
		  Vec_t vy_trial = add_intrin(mul_intrin(fy, rinv),
					 mul_intrin(mul_intrin(one_over_three, a2_add_a2),
						    mul_intrin(fy, rinv3)));
		  Vec_t vz_trial = add_intrin(mul_intrin(fz, rinv),
					 mul_intrin(mul_intrin(one_over_three, a2_add_a2),
						    mul_intrin(fz, rinv3)));
		  vx_trial = add_intrin(vx_trial, sub_intrin(mul_intrin(dx, fdotr_rinv3),
							     mul_intrin(a2_add_b2,
									mul_intrin(dx, fdotr_rinv5))));
		  vy_trial = add_intrin(vy_trial, sub_intrin(mul_intrin(dy, fdotr_rinv3),
							     mul_intrin(a2_add_b2,
									mul_intrin(dy, fdotr_rinv5))));
		  vz_trial = add_intrin(vz_trial, sub_intrin(mul_intrin(dz, fdotr_rinv3),
							     mul_intrin(a2_add_b2,
									mul_intrin(dz, fdotr_rinv5))));
		  const Vec_t add_lt_r = cmplt_intrin(a_add_b_2, r2);
		  vx = blend_intrin(vx, vx_trial, add_lt_r);
		  vy = blend_intrin(vy, vy_trial, add_lt_r);
		  vz = blend_intrin(vz, vz_trial, add_lt_r);
		}
            }

            vx = add_intrin(mul_intrin(vx, FACV), load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, FACV), load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, FACV), load_intrin<Vec_t>(&trg_value[2][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
        }
    }
}

/**********************************************************
 *                                                        *
 * Stokes Force Vel,lapVel kernel,source: 3, target: 6    *
 *       fx,fy,fz -> ux,uy,uz,lapux,lapuy,lapuz           *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stk_ulapu_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                       Matrix<Real_t> &trg_value) {

    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);
    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Vec_t FACV = set_intrin<Vec_t, Real_t>(1.0 / (8 * const_pi<Real_t>()));
    Vec_t nwtn_factor = set_intrin<Vec_t, Real_t>(1.0 / nwtn_scal);
    Vec_t two = set_intrin<Vec_t, Real_t>(2.0);
    Vec_t three = set_intrin<Vec_t, Real_t>(3.0);

    const size_t src_cnt_ = src_coord.Dim(1);
    const size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>();
            Vec_t vy = zero_intrin<Vec_t>();
            Vec_t vz = zero_intrin<Vec_t>();
            Vec_t lapvx = zero_intrin<Vec_t>();
            Vec_t lapvy = zero_intrin<Vec_t>();
            Vec_t lapvz = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = tx - bcast_intrin<Vec_t>(&src_coord[0][s]);
                const Vec_t dy = ty - bcast_intrin<Vec_t>(&src_coord[1][s]);
                const Vec_t dz = tz - bcast_intrin<Vec_t>(&src_coord[2][s]);

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);

                const Vec_t r2 = add_intrin(add_intrin(mul_intrin(dx, dx), mul_intrin(dy, dy)), mul_intrin(dz, dz));

                const Vec_t rinv = mul_intrin(rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2), nwtn_factor);
                const Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                const Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);
                const Vec_t fdotr = add_intrin(add_intrin(mul_intrin(fx, dx), mul_intrin(fy, dy)), mul_intrin(fz, dz));

                const Vec_t three_fdotr_rinv5 = mul_intrin(mul_intrin(three, fdotr), rinv5);
                const Vec_t cx = sub_intrin(mul_intrin(fx, rinv3), mul_intrin(three_fdotr_rinv5, dx));
                const Vec_t cy = sub_intrin(mul_intrin(fy, rinv3), mul_intrin(three_fdotr_rinv5, dy));
                const Vec_t cz = sub_intrin(mul_intrin(fz, rinv3), mul_intrin(three_fdotr_rinv5, dz));

                const Vec_t fdotr_rinv3 = mul_intrin(fdotr, rinv3);
                vx = add_intrin(vx, add_intrin(mul_intrin(fx, rinv), mul_intrin(dx, fdotr_rinv3)));
                vy = add_intrin(vy, add_intrin(mul_intrin(fy, rinv), mul_intrin(dy, fdotr_rinv3)));
                vz = add_intrin(vz, add_intrin(mul_intrin(fz, rinv), mul_intrin(dz, fdotr_rinv3)));

                lapvx = add_intrin(lapvx, mul_intrin(two, cx));
                lapvy = add_intrin(lapvy, mul_intrin(two, cy));
                lapvz = add_intrin(lapvz, mul_intrin(two, cz));
            }

            vx = add_intrin(mul_intrin(vx, FACV), load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, FACV), load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, FACV), load_intrin<Vec_t>(&trg_value[2][t]));
            lapvx = add_intrin(mul_intrin(lapvx, FACV), load_intrin<Vec_t>(&trg_value[3][t]));
            lapvy = add_intrin(mul_intrin(lapvy, FACV), load_intrin<Vec_t>(&trg_value[4][t]));
            lapvz = add_intrin(mul_intrin(lapvz, FACV), load_intrin<Vec_t>(&trg_value[5][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
            store_intrin(&trg_value[3][t], lapvx);
            store_intrin(&trg_value[4][t], lapvy);
            store_intrin(&trg_value[5][t], lapvz);
        }
    }
}

/***********************************************************
 *                                                         *
 * Regularized RPY surface-to-sphere kernel                *
 * Just Stokes with a Faxen operator, source: 3, target: 4 *
 *       fx,fy,fz -> ux,uy,uz,a                            *
 ***********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void rpy_reg_surface_to_sphere_uKernel(Matrix<Real_t> &src_coord,
				       Matrix<Real_t> &src_value,
				       Matrix<Real_t> &trg_coord,
				       Matrix<Real_t> &trg_value) {

    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);
    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Vec_t FACV = set_intrin<Vec_t, Real_t>(1.0 / (8 * const_pi<Real_t>()));
    const Vec_t nwtn_factor = set_intrin<Vec_t, Real_t>(1.0 / nwtn_scal);
    const Vec_t two = set_intrin<Vec_t, Real_t>(2.0);
    const Vec_t three = set_intrin<Vec_t, Real_t>(3.0);
    const Vec_t one_over_six = set_intrin<Vec_t, Real_t>(static_cast<Real_t>(1.0 / 6.0));

    const size_t src_cnt_ = src_coord.Dim(1);
    const size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>();
            Vec_t vy = zero_intrin<Vec_t>();
            Vec_t vz = zero_intrin<Vec_t>();

	    // These are laplacians scaled with a^2/6.
	    Vec_t lapvx = zero_intrin<Vec_t>();
            Vec_t lapvy = zero_intrin<Vec_t>();
            Vec_t lapvz = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = tx - bcast_intrin<Vec_t>(&src_coord[0][s]);
                const Vec_t dy = ty - bcast_intrin<Vec_t>(&src_coord[1][s]);
                const Vec_t dz = tz - bcast_intrin<Vec_t>(&src_coord[2][s]);

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);

                const Vec_t r2 = add_intrin(add_intrin(mul_intrin(dx, dx), mul_intrin(dy, dy)), mul_intrin(dz, dz));

                const Vec_t rinv = mul_intrin(rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2), nwtn_factor);
                const Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                const Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);
                const Vec_t fdotr = add_intrin(add_intrin(mul_intrin(fx, dx), mul_intrin(fy, dy)), mul_intrin(fz, dz));

                const Vec_t three_fdotr_rinv5 = mul_intrin(mul_intrin(three, fdotr), rinv5);
                const Vec_t cx = sub_intrin(mul_intrin(fx, rinv3), mul_intrin(three_fdotr_rinv5, dx));
                const Vec_t cy = sub_intrin(mul_intrin(fy, rinv3), mul_intrin(three_fdotr_rinv5, dy));
                const Vec_t cz = sub_intrin(mul_intrin(fz, rinv3), mul_intrin(three_fdotr_rinv5, dz));

                const Vec_t fdotr_rinv3 = mul_intrin(fdotr, rinv3);
                vx = add_intrin(vx, add_intrin(mul_intrin(fx, rinv), mul_intrin(dx, fdotr_rinv3)));
                vy = add_intrin(vy, add_intrin(mul_intrin(fy, rinv), mul_intrin(dy, fdotr_rinv3)));
                vz = add_intrin(vz, add_intrin(mul_intrin(fz, rinv), mul_intrin(dz, fdotr_rinv3)));
                lapvx = add_intrin(lapvx, mul_intrin(two, cx));
                lapvy = add_intrin(lapvy, mul_intrin(two, cy));
                lapvz = add_intrin(lapvz, mul_intrin(two, cz));
            }

            vx = add_intrin(mul_intrin(vx, FACV), load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, FACV), load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, FACV), load_intrin<Vec_t>(&trg_value[2][t]));

	    // Add the Laplacian part of the velocity.
	    // Need to scale by the target's squared radius.
	    const Vec_t b = bcast_intrin<Vec_t>(&trg_value[3][t]);
	    const Vec_t b2_over_six = mul_intrin(mul_intrin(b, b), one_over_six);

	    vx = add_intrin(mul_intrin(mul_intrin(lapvx, FACV), b2_over_six), load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(mul_intrin(lapvy, FACV), b2_over_six), load_intrin<Vec_t>(&trg_value[1][t]));
	    vz = add_intrin(mul_intrin(mul_intrin(lapvz, FACV), b2_over_six), load_intrin<Vec_t>(&trg_value[2][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
        }
    }
}

/**********************************************************
 *                                                        *
 *   Laplace quadrupole kernel,source: 4, target: 4       *
 *       fx,fy,fz,b -> phi,gradphix,gradphiy,gradphiz     *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void laplace_phigradphi_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                                Matrix<Real_t> &trg_value) {

    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);
    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Vec_t FACV = set_intrin<Vec_t, Real_t>(1.0 / (4.0 * const_pi<Real_t>()));
    Vec_t nwtn_factor = set_intrin<Vec_t, Real_t>(1.0 / nwtn_scal);
    Vec_t two = set_intrin<Vec_t, Real_t>(2.0);
    Vec_t three = set_intrin<Vec_t, Real_t>(3.0);

    const size_t src_cnt_ = src_coord.Dim(1);
    const size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t phi = zero_intrin<Vec_t>();
            Vec_t gradphix = zero_intrin<Vec_t>();
            Vec_t gradphiy = zero_intrin<Vec_t>();
            Vec_t gradphiz = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = tx - bcast_intrin<Vec_t>(&src_coord[0][s]);
                const Vec_t dy = ty - bcast_intrin<Vec_t>(&src_coord[1][s]);
                const Vec_t dz = tz - bcast_intrin<Vec_t>(&src_coord[2][s]);
                const Vec_t dx2 = mul_intrin(dx, dx);
                const Vec_t dy2 = mul_intrin(dy, dy);
                const Vec_t dz2 = mul_intrin(dz, dz);
                const Vec_t dxdydz = mul_intrin(mul_intrin(dx, dy), dz);

                Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                Vec_t b2 = bcast_intrin<Vec_t>(&src_value[3][s]);
                // All terms scale as b2 * f, so premultiply
                b2 = mul_intrin(b2, b2);
                fx = mul_intrin(b2, fx);
                fy = mul_intrin(b2, fy);
                fz = mul_intrin(b2, fz);

                const Vec_t r2 = add_intrin(add_intrin(mul_intrin(dx, dx), mul_intrin(dy, dy)), mul_intrin(dz, dz));

                const Vec_t rinv = mul_intrin(rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2), nwtn_factor);
                const Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                const Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);
                const Vec_t three_rinv5 = mul_intrin(three, rinv5);

                // TODO: verify and optimize Laplace Quadrupole interaction
                Vec_t Gzx = mul_intrin(mul_intrin(dx, dz), rinv3);
                Vec_t Gzy = mul_intrin(mul_intrin(dy, dz), rinv3);
                Vec_t Gxx_Gyy = add_intrin(add_intrin(rinv, rinv), mul_intrin(add_intrin(dx2, dy2), rinv3));

                phi = add_intrin(
                    phi, add_intrin(add_intrin(mul_intrin(Gzx, fx), mul_intrin(Gzy, fy)), mul_intrin(Gxx_Gyy, fz)));

                gradphix = add_intrin(
                    gradphix,
                    mul_intrin(sub_intrin(mul_intrin(dz, rinv3), mul_intrin(mul_intrin(dx2, dz), three_rinv5)), fx));
                gradphix = sub_intrin(gradphix, mul_intrin(mul_intrin(dxdydz, three_rinv5), fy));
                gradphix = sub_intrin(gradphix, mul_intrin(mul_intrin(three_rinv5, fz),
                                                           add_intrin(mul_intrin(dx, dx2), mul_intrin(dx, dy2))));

                gradphiy = sub_intrin(gradphiy, mul_intrin(mul_intrin(dxdydz, three_rinv5), fx));
                gradphiy = add_intrin(
                    gradphiy,
                    mul_intrin(sub_intrin(mul_intrin(dz, rinv3), mul_intrin(mul_intrin(dy2, dz), three_rinv5)), fy));
                gradphiy = sub_intrin(gradphiy, mul_intrin(mul_intrin(three_rinv5, fz),
                                                           add_intrin(mul_intrin(dy, dy2), mul_intrin(dy, dx2))));

                gradphiz = add_intrin(
                    gradphiz,
                    mul_intrin(sub_intrin(mul_intrin(dx, rinv3), mul_intrin(mul_intrin(dz2, dx), three_rinv5)), fx));
                gradphiz = add_intrin(
                    gradphiz,
                    mul_intrin(sub_intrin(mul_intrin(dy, rinv3), mul_intrin(mul_intrin(dz2, dy), three_rinv5)), fy));
                gradphiz = sub_intrin(
                    gradphiz, mul_intrin(add_intrin(mul_intrin(mul_intrin(two, dz), rinv3),
                                                    mul_intrin(three_rinv5,
                                                               add_intrin(mul_intrin(dx2, dz), mul_intrin(dy2, dz)))),
                                         fz));
            }

            phi = add_intrin(mul_intrin(phi, FACV), load_intrin<Vec_t>(&trg_value[0][t]));
            gradphix = add_intrin(mul_intrin(gradphix, FACV), load_intrin<Vec_t>(&trg_value[1][t]));
            gradphiy = add_intrin(mul_intrin(gradphiy, FACV), load_intrin<Vec_t>(&trg_value[2][t]));
            gradphiz = add_intrin(mul_intrin(gradphiz, FACV), load_intrin<Vec_t>(&trg_value[3][t]));

            store_intrin(&trg_value[0][t], phi);
            store_intrin(&trg_value[1][t], gradphix);
            store_intrin(&trg_value[2][t], gradphiy);
            store_intrin(&trg_value[3][t], gradphiz);
        }
    }
}

GEN_KERNEL(rpy_u, rpy_u_uKernel, 4, 3)
GEN_KERNEL(rpy_ulapu, rpy_ulapu_uKernel, 4, 6)
GEN_KERNEL(rpy_reg_sphere_to_sphere, rpy_reg_sphere_to_sphere_uKernel, 4, 4)
GEN_KERNEL(stk_ulapu, stk_ulapu_uKernel, 3, 6)
GEN_KERNEL(rpy_reg_surf_to_sphere, rpy_reg_surf_to_sphere_uKernel, 3, 4)
// GEN_KERNEL(laplace_phigradphi, laplace_phigradphi_uKernel, 4, 4)

template <class T>
struct RPYKernel {
    inline static const Kernel<T> &ulapu(); //   3+1->6
  private:
    static constexpr int NEWTON_ITE = sizeof(T) / 4;
};

// 1 newton for float, 2 newton for double
// the string for stk_ker must be exactly the same as in kernel.txx of pvfmm
template <class T>
inline const Kernel<T> &RPYKernel<T>::ulapu() {

    static Kernel<T> g_ker = StokesKernel<T>::velocity();
    static Kernel<T> gr_ker = BuildKernel<T, rpy_u<T, NEWTON_ITE>>("rpy_u", 3, std::pair<int, int>(4, 3));

    static Kernel<T> glapg_ker = BuildKernel<T, stk_ulapu<T, NEWTON_ITE>>("stk_ulapu", 3, std::pair<int, int>(3, 6));

    static Kernel<T> grlapgr_ker = BuildKernel<T, rpy_ulapu<T, NEWTON_ITE>>("rpy_ulapu", 3, std::pair<int, int>(4, 6),
                                                                            &gr_ker,    // k_s2m
                                                                            &gr_ker,    // k_s2l
                                                                            NULL,       // k_s2t
                                                                            &g_ker,     // k_m2m
                                                                            &g_ker,     // k_m2l
                                                                            &glapg_ker, // k_m2t
                                                                            &g_ker,     // k_l2l
                                                                            &glapg_ker, // k_l2t
                                                                            NULL);
    return grlapgr_ker;
}

// 1 newton for float, 2 newton for double
// the string for stk_ker must be exactly the same as in kernel.txx of pvfmm
template <class T>
inline const Kernel<T> &RPYKernel<T>::regularized() {
    static Kernel<T> surface_to_surface = StokesKernel<T>::velocity();
    static Kernel<T> sphere_to_surface = BuildKernel<T, rpy_u<T, NEWTON_ITE>>("rpy_u", 3, std::pair<int, int>(4, 3));
    static Kernel<T> surface_to_sphere = BuildKernel<T, rpy_reg_surface_to_sphere<T, NEWTON_ITE>>("rpy_reg_surface_to_sphere", 3, std::pair<int, int>(3, 4));
    static Kernel<T> full_kernel = BuildKernel<T, rpy_reg_sphere_to_sphere<T, NEWTON_ITE>>(
        "rpy_reg_sphere_to_sphere", 3, std::pair<int, int>(4, 4),
	&sphere_to_surface,   // k_s2m
	&sphere_to_surface,   // k_s2l
	NULL,                 // k_s2t
	&surface_to_surface,  // k_m2m
	&surface_to_surface,  // k_m2l
	&surface_to_sphere,   // k_m2t
	&surface_to_surface,  // k_l2l
	&surface_to_sphere,   // k_l2t
	NULL);
    return full_kernel;
}
} // namespace pvfmm

#endif
