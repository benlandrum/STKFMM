/**
 * @file StokesSingleLayerKernel.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief Stokes single layer kernels
 * @version 0.1
 * @date 2019-12-23
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef STOKESSINGLELAYER_HPP_
#define STOKESSINGLELAYER_HPP_

#include <cmath>
#include <cstdlib>
#include <vector>

#include "stkfmm_helpers.hpp"

namespace pvfmm {

/*********************************************************
 *                                                        *
 *     Stokes P Vel kernel, source: 4, target: 4          *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER = 0>
void stokes_pvel_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                         Matrix<Real_t> &trg_value) {

    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV = 1.0 / (8 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);
    const Vec_t facp = set_intrin<Vec_t, Real_t>(2 * FACV);
    // p = (1/4pi) rk Fk /r^3
    // const Vec_t n23 = set_intrin<Vec_t, Real_t>(static_cast<Real_t>(-2.0 / 3.0));

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t p = zero_intrin<Vec_t>();  // pressure
            Vec_t vx = zero_intrin<Vec_t>(); // vx
            Vec_t vy = zero_intrin<Vec_t>(); // vy
            Vec_t vz = zero_intrin<Vec_t>(); // vz

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                const Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                const Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                const Vec_t tr = bcast_intrin<Vec_t>(&src_value[3][s]); // trace of doublet

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);

                Vec_t commonCoeff = mul_intrin(fx, dx);
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fy, dy));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fz, dz));

                p = add_intrin(p, mul_intrin(rinv3, commonCoeff));
                commonCoeff = sub_intrin(commonCoeff, tr);
                vx = add_intrin(vx, mul_intrin(add_intrin(mul_intrin(r2, fx), mul_intrin(dx, commonCoeff)), rinv3));
                vy = add_intrin(vy, mul_intrin(add_intrin(mul_intrin(r2, fy), mul_intrin(dy, commonCoeff)), rinv3));
                vz = add_intrin(vz, mul_intrin(add_intrin(mul_intrin(r2, fz), mul_intrin(dz, commonCoeff)), rinv3));
            }

            p = add_intrin(mul_intrin(p, facp), load_intrin<Vec_t>(&trg_value[0][t]));
            vx = add_intrin(mul_intrin(vx, facv), load_intrin<Vec_t>(&trg_value[1][t]));
            vy = add_intrin(mul_intrin(vy, facv), load_intrin<Vec_t>(&trg_value[2][t]));
            vz = add_intrin(mul_intrin(vz, facv), load_intrin<Vec_t>(&trg_value[3][t]));

            store_intrin(&trg_value[0][t], p);
            store_intrin(&trg_value[1][t], vx);
            store_intrin(&trg_value[2][t], vy);
            store_intrin(&trg_value[3][t], vz);
        }
    }
}

GEN_KERNEL(stokes_pvel, stokes_pvel_uKernel, 4, 4, 0)

/*********************************************************
 *                                                        *
 *   Stokes P Vel Grad kernel, source: 4, target: 1+3+3+9 *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stokes_pvelgrad_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                             Matrix<Real_t> &trg_value) {
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV = 1.0 / (8 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);
    const Vec_t facp = set_intrin<Vec_t, Real_t>(2 * FACV);

    const Real_t FACV5 = 1.0 / (8 * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Vec_t facv5 = set_intrin<Vec_t, Real_t>(FACV5);
    const Vec_t facp5 = set_intrin<Vec_t, Real_t>(2 * FACV5);
    const Vec_t nthree = set_intrin<Vec_t, Real_t>(-3.0);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t p = zero_intrin<Vec_t>();       // pressure
            Vec_t vx = zero_intrin<Vec_t>();      // vx
            Vec_t vy = zero_intrin<Vec_t>();      // vy
            Vec_t vz = zero_intrin<Vec_t>();      // vz
            Vec_t pgxSum = zero_intrin<Vec_t>();  // p grad x
            Vec_t pgySum = zero_intrin<Vec_t>();  // p grad y
            Vec_t pgzSum = zero_intrin<Vec_t>();  // p grad z
            Vec_t vxgxSum = zero_intrin<Vec_t>(); // vx grad
            Vec_t vxgySum = zero_intrin<Vec_t>(); //
            Vec_t vxgzSum = zero_intrin<Vec_t>(); //
            Vec_t vygxSum = zero_intrin<Vec_t>(); // vy grad
            Vec_t vygySum = zero_intrin<Vec_t>(); //
            Vec_t vygzSum = zero_intrin<Vec_t>(); //
            Vec_t vzgxSum = zero_intrin<Vec_t>(); // vz grad
            Vec_t vzgySum = zero_intrin<Vec_t>(); //
            Vec_t vzgzSum = zero_intrin<Vec_t>(); //

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                const Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                const Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                const Vec_t tr = bcast_intrin<Vec_t>(&src_value[3][s]); // trace of doublet

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);

                Vec_t commonCoeff = mul_intrin(fx, dx);
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fy, dy));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fz, dz));

                p = add_intrin(p, mul_intrin(rinv3, commonCoeff));
                commonCoeff = sub_intrin(commonCoeff, tr);
                vx = add_intrin(vx, mul_intrin(add_intrin(mul_intrin(r2, fx), mul_intrin(dx, commonCoeff)), rinv3));
                vy = add_intrin(vy, mul_intrin(add_intrin(mul_intrin(r2, fy), mul_intrin(dy, commonCoeff)), rinv3));
                vz = add_intrin(vz, mul_intrin(add_intrin(mul_intrin(r2, fz), mul_intrin(dz, commonCoeff)), rinv3));
                Vec_t px = zero_intrin<Vec_t>(); // p grad x
                Vec_t py = zero_intrin<Vec_t>(); // p grad y
                Vec_t pz = zero_intrin<Vec_t>(); // p grad z

                Vec_t vxx = zero_intrin<Vec_t>(); // vx grad
                Vec_t vxy = zero_intrin<Vec_t>(); //
                Vec_t vxz = zero_intrin<Vec_t>(); //

                Vec_t vyx = zero_intrin<Vec_t>(); // vy grad
                Vec_t vyy = zero_intrin<Vec_t>(); //
                Vec_t vyz = zero_intrin<Vec_t>(); //

                Vec_t vzx = zero_intrin<Vec_t>(); // vz grad
                Vec_t vzy = zero_intrin<Vec_t>(); //
                Vec_t vzz = zero_intrin<Vec_t>(); //

                // qij = r^2 \delta_{ij} - 3 ri rj, symmetric
                Vec_t qxx = add_intrin(r2, mul_intrin(nthree, mul_intrin(dx, dx)));
                Vec_t qxy = mul_intrin(nthree, mul_intrin(dx, dy));
                Vec_t qxz = mul_intrin(nthree, mul_intrin(dx, dz));
                Vec_t qyy = add_intrin(r2, mul_intrin(nthree, mul_intrin(dy, dy)));
                Vec_t qyz = mul_intrin(nthree, mul_intrin(dy, dz));
                Vec_t qzz = add_intrin(r2, mul_intrin(nthree, mul_intrin(dz, dz)));

                // px dp/dx, etc
                commonCoeff = add_intrin(commonCoeff, tr);
                px = add_intrin(mul_intrin(r2, fx), mul_intrin(mul_intrin(nthree, dx), commonCoeff));
                py = add_intrin(mul_intrin(r2, fy), mul_intrin(mul_intrin(nthree, dy), commonCoeff));
                pz = add_intrin(mul_intrin(r2, fz), mul_intrin(mul_intrin(nthree, dz), commonCoeff));

                // vxx = dvx/dx , etc
                commonCoeff = sub_intrin(commonCoeff, tr);
                vxx = mul_intrin(qxx, commonCoeff);
                vxy = add_intrin(mul_intrin(qxy, commonCoeff),
                                 mul_intrin(r2, sub_intrin(mul_intrin(dx, fy), mul_intrin(dy, fx))));
                vxz = add_intrin(mul_intrin(qxz, commonCoeff),
                                 mul_intrin(r2, sub_intrin(mul_intrin(dx, fz), mul_intrin(dz, fx))));

                vyx = add_intrin(mul_intrin(qxy, commonCoeff),
                                 mul_intrin(r2, sub_intrin(mul_intrin(dy, fx), mul_intrin(dx, fy))));
                vyy = mul_intrin(qyy, commonCoeff);
                vyz = add_intrin(mul_intrin(qyz, commonCoeff),
                                 mul_intrin(r2, sub_intrin(mul_intrin(dy, fz), mul_intrin(dz, fy))));

                vzx = add_intrin(mul_intrin(qxz, commonCoeff),
                                 mul_intrin(r2, sub_intrin(mul_intrin(dz, fx), mul_intrin(dx, fz))));
                vzy = add_intrin(mul_intrin(qyz, commonCoeff),
                                 mul_intrin(r2, sub_intrin(mul_intrin(dz, fy), mul_intrin(dy, fz))));
                vzz = mul_intrin(qzz, commonCoeff);

                // calculate grad
                pgxSum = add_intrin(pgxSum, mul_intrin(px, rinv5));
                pgySum = add_intrin(pgySum, mul_intrin(py, rinv5));
                pgzSum = add_intrin(pgzSum, mul_intrin(pz, rinv5));

                vxgxSum = add_intrin(vxgxSum, mul_intrin(vxx, rinv5));
                vxgySum = add_intrin(vxgySum, mul_intrin(vxy, rinv5));
                vxgzSum = add_intrin(vxgzSum, mul_intrin(vxz, rinv5));

                vygxSum = add_intrin(vygxSum, mul_intrin(vyx, rinv5));
                vygySum = add_intrin(vygySum, mul_intrin(vyy, rinv5));
                vygzSum = add_intrin(vygzSum, mul_intrin(vyz, rinv5));

                vzgxSum = add_intrin(vzgxSum, mul_intrin(vzx, rinv5));
                vzgySum = add_intrin(vzgySum, mul_intrin(vzy, rinv5));
                vzgzSum = add_intrin(vzgzSum, mul_intrin(vzz, rinv5));
            }

            p = add_intrin(mul_intrin(p, facp), load_intrin<Vec_t>(&trg_value[0][t]));
            vx = add_intrin(mul_intrin(vx, facv), load_intrin<Vec_t>(&trg_value[1][t]));
            vy = add_intrin(mul_intrin(vy, facv), load_intrin<Vec_t>(&trg_value[2][t]));
            vz = add_intrin(mul_intrin(vz, facv), load_intrin<Vec_t>(&trg_value[3][t]));

            pgxSum = add_intrin(mul_intrin(pgxSum, facp5), load_intrin<Vec_t>(&trg_value[4][t]));
            pgySum = add_intrin(mul_intrin(pgySum, facp5), load_intrin<Vec_t>(&trg_value[5][t]));
            pgzSum = add_intrin(mul_intrin(pgzSum, facp5), load_intrin<Vec_t>(&trg_value[6][t]));

            vxgxSum = add_intrin(mul_intrin(vxgxSum, facv5), load_intrin<Vec_t>(&trg_value[7][t]));
            vxgySum = add_intrin(mul_intrin(vxgySum, facv5), load_intrin<Vec_t>(&trg_value[8][t]));
            vxgzSum = add_intrin(mul_intrin(vxgzSum, facv5), load_intrin<Vec_t>(&trg_value[9][t]));

            vygxSum = add_intrin(mul_intrin(vygxSum, facv5), load_intrin<Vec_t>(&trg_value[10][t]));
            vygySum = add_intrin(mul_intrin(vygySum, facv5), load_intrin<Vec_t>(&trg_value[11][t]));
            vygzSum = add_intrin(mul_intrin(vygzSum, facv5), load_intrin<Vec_t>(&trg_value[12][t]));

            vzgxSum = add_intrin(mul_intrin(vzgxSum, facv5), load_intrin<Vec_t>(&trg_value[13][t]));
            vzgySum = add_intrin(mul_intrin(vzgySum, facv5), load_intrin<Vec_t>(&trg_value[14][t]));
            vzgzSum = add_intrin(mul_intrin(vzgzSum, facv5), load_intrin<Vec_t>(&trg_value[15][t]));

            store_intrin(&trg_value[0][t], p);
            store_intrin(&trg_value[1][t], vx);
            store_intrin(&trg_value[2][t], vy);
            store_intrin(&trg_value[3][t], vz);
            store_intrin(&trg_value[4][t], pgxSum);
            store_intrin(&trg_value[5][t], pgySum);
            store_intrin(&trg_value[6][t], pgzSum);
            store_intrin(&trg_value[7][t], vxgxSum);
            store_intrin(&trg_value[8][t], vxgySum);
            store_intrin(&trg_value[9][t], vxgzSum);
            store_intrin(&trg_value[10][t], vygxSum);
            store_intrin(&trg_value[11][t], vygySum);
            store_intrin(&trg_value[12][t], vygzSum);
            store_intrin(&trg_value[13][t], vzgxSum);
            store_intrin(&trg_value[14][t], vzgySum);
            store_intrin(&trg_value[15][t], vzgzSum);
        }
    }
}

GEN_KERNEL(stokes_pvelgrad, stokes_pvelgrad_uKernel, 4, 16, 0)

/*********************************************************
 *                                                        *
 *      Stokes traction kernel, source: 4, target: 9      *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stokes_traction_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                             Matrix<Real_t> &trg_value) {
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t OOEP = -3.0 / (4 * const_pi<Real_t>());
    Vec_t inv_nwtn_scal5 = set_intrin<Vec_t, Real_t>(1.0 / (nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal));
    Vec_t invthree = set_intrin<Vec_t, Real_t>(static_cast<Real_t>(1.0 / 3.0));

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);
    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t tv0 = zero_intrin<Vec_t>();
            Vec_t tv1 = zero_intrin<Vec_t>();
            Vec_t tv2 = zero_intrin<Vec_t>();
            Vec_t tv3 = zero_intrin<Vec_t>();
            Vec_t tv4 = zero_intrin<Vec_t>();
            Vec_t tv5 = zero_intrin<Vec_t>();
            Vec_t tv6 = zero_intrin<Vec_t>();
            Vec_t tv7 = zero_intrin<Vec_t>();
            Vec_t tv8 = zero_intrin<Vec_t>();

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                Vec_t sv0 = bcast_intrin<Vec_t>(&src_value[0][s]);
                Vec_t sv1 = bcast_intrin<Vec_t>(&src_value[1][s]);
                Vec_t sv2 = bcast_intrin<Vec_t>(&src_value[2][s]);
                Vec_t tr = bcast_intrin<Vec_t>(&src_value[3][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                Vec_t rinv2 = mul_intrin(rinv, rinv);
                Vec_t rinv4 = mul_intrin(rinv2, rinv2);

                Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv4), inv_nwtn_scal5);

                Vec_t commonCoeff = mul_intrin(sv0, dx);
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv1, dy));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(sv2, dz));
                commonCoeff = sub_intrin(commonCoeff, tr);
                Vec_t diag = mul_intrin(mul_intrin(mul_intrin(tr, r2), rinv5), invthree);

                tv0 = add_intrin(tv0, mul_intrin(rinv5, mul_intrin(mul_intrin(dx, dx), commonCoeff)));
                tv0 = add_intrin(tv0, diag);
                tv1 = add_intrin(tv1, mul_intrin(rinv5, mul_intrin(mul_intrin(dx, dy), commonCoeff)));
                tv2 = add_intrin(tv2, mul_intrin(rinv5, mul_intrin(mul_intrin(dx, dz), commonCoeff)));
                tv3 = add_intrin(tv3, mul_intrin(rinv5, mul_intrin(mul_intrin(dy, dx), commonCoeff)));
                tv4 = add_intrin(tv4, mul_intrin(rinv5, mul_intrin(mul_intrin(dy, dy), commonCoeff)));
                tv4 = add_intrin(tv4, diag);
                tv5 = add_intrin(tv5, mul_intrin(rinv5, mul_intrin(mul_intrin(dy, dz), commonCoeff)));
                tv6 = add_intrin(tv6, mul_intrin(rinv5, mul_intrin(mul_intrin(dz, dx), commonCoeff)));
                tv7 = add_intrin(tv7, mul_intrin(rinv5, mul_intrin(mul_intrin(dz, dy), commonCoeff)));
                tv8 = add_intrin(tv8, mul_intrin(rinv5, mul_intrin(mul_intrin(dz, dz), commonCoeff)));
                tv8 = add_intrin(tv8, diag);
            }
            Vec_t ooep = set_intrin<Vec_t, Real_t>(OOEP);

            tv0 = add_intrin(mul_intrin(tv0, ooep), load_intrin<Vec_t>(&trg_value[0][t]));
            tv1 = add_intrin(mul_intrin(tv1, ooep), load_intrin<Vec_t>(&trg_value[1][t]));
            tv2 = add_intrin(mul_intrin(tv2, ooep), load_intrin<Vec_t>(&trg_value[2][t]));
            tv3 = add_intrin(mul_intrin(tv3, ooep), load_intrin<Vec_t>(&trg_value[3][t]));
            tv4 = add_intrin(mul_intrin(tv4, ooep), load_intrin<Vec_t>(&trg_value[4][t]));
            tv5 = add_intrin(mul_intrin(tv5, ooep), load_intrin<Vec_t>(&trg_value[5][t]));
            tv6 = add_intrin(mul_intrin(tv6, ooep), load_intrin<Vec_t>(&trg_value[6][t]));
            tv7 = add_intrin(mul_intrin(tv7, ooep), load_intrin<Vec_t>(&trg_value[7][t]));
            tv8 = add_intrin(mul_intrin(tv8, ooep), load_intrin<Vec_t>(&trg_value[8][t]));

            store_intrin(&trg_value[0][t], tv0);
            store_intrin(&trg_value[1][t], tv1);
            store_intrin(&trg_value[2][t], tv2);
            store_intrin(&trg_value[3][t], tv3);
            store_intrin(&trg_value[4][t], tv4);
            store_intrin(&trg_value[5][t], tv5);
            store_intrin(&trg_value[6][t], tv6);
            store_intrin(&trg_value[7][t], tv7);
            store_intrin(&trg_value[8][t], tv8);
        }
    }
}

GEN_KERNEL(stokes_traction, stokes_traction_uKernel, 4, 9, 0)

/*********************************************************
 *                                                        *
 * Stokes P Vel Laplacian kernel, source: 4, target: 7      *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stokes_pvellaplacian_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                                  Matrix<Real_t> &trg_value) {
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV = 1.0 / (8 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Real_t FACLAP = 1.0 / (4 * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);
    const Vec_t facp = set_intrin<Vec_t, Real_t>(2 * FACV);
    const Vec_t faclap = set_intrin<Vec_t, Real_t>(FACLAP); // laplacian = 1/(4pi) (I/r^3-3rr/r^5)
    const Vec_t nthree = set_intrin<Vec_t, Real_t>(-3.0);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t p = zero_intrin<Vec_t>();     // pressure
            Vec_t vx = zero_intrin<Vec_t>();    // vx
            Vec_t vy = zero_intrin<Vec_t>();    // vy
            Vec_t vz = zero_intrin<Vec_t>();    // vz
            Vec_t vxlap = zero_intrin<Vec_t>(); // vx laplacian
            Vec_t vylap = zero_intrin<Vec_t>(); // vy
            Vec_t vzlap = zero_intrin<Vec_t>(); // vz

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                const Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                const Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                const Vec_t tr = bcast_intrin<Vec_t>(&src_value[3][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);

                Vec_t commonCoeff = mul_intrin(fx, dx);
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fy, dy));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fz, dz));

                p = add_intrin(p, mul_intrin(rinv3, commonCoeff));
                commonCoeff = sub_intrin(commonCoeff, tr);
                vx = add_intrin(vx, mul_intrin(add_intrin(mul_intrin(r2, fx), mul_intrin(dx, commonCoeff)), rinv3));
                vy = add_intrin(vy, mul_intrin(add_intrin(mul_intrin(r2, fy), mul_intrin(dy, commonCoeff)), rinv3));
                vz = add_intrin(vz, mul_intrin(add_intrin(mul_intrin(r2, fz), mul_intrin(dz, commonCoeff)), rinv3));

                commonCoeff = add_intrin(commonCoeff, tr);
                vxlap = add_intrin(vxlap, mul_intrin(mul_intrin(fx, r2), rinv5));
                vxlap = add_intrin(vxlap, mul_intrin(mul_intrin(mul_intrin(commonCoeff, nthree), rinv5), dx));
                vylap = add_intrin(vylap, mul_intrin(mul_intrin(fy, r2), rinv5));
                vylap = add_intrin(vylap, mul_intrin(mul_intrin(mul_intrin(commonCoeff, nthree), rinv5), dy));
                vzlap = add_intrin(vzlap, mul_intrin(mul_intrin(fz, r2), rinv5));
                vzlap = add_intrin(vzlap, mul_intrin(mul_intrin(mul_intrin(commonCoeff, nthree), rinv5), dz));
            }

            p = add_intrin(mul_intrin(p, facp), load_intrin<Vec_t>(&trg_value[0][t]));
            vx = add_intrin(mul_intrin(vx, facv), load_intrin<Vec_t>(&trg_value[1][t]));
            vy = add_intrin(mul_intrin(vy, facv), load_intrin<Vec_t>(&trg_value[2][t]));
            vz = add_intrin(mul_intrin(vz, facv), load_intrin<Vec_t>(&trg_value[3][t]));
            vxlap = add_intrin(mul_intrin(vxlap, faclap), load_intrin<Vec_t>(&trg_value[4][t]));
            vylap = add_intrin(mul_intrin(vylap, faclap), load_intrin<Vec_t>(&trg_value[5][t]));
            vzlap = add_intrin(mul_intrin(vzlap, faclap), load_intrin<Vec_t>(&trg_value[6][t]));

            store_intrin(&trg_value[0][t], p);
            store_intrin(&trg_value[1][t], vx);
            store_intrin(&trg_value[2][t], vy);
            store_intrin(&trg_value[3][t], vz);
            store_intrin(&trg_value[4][t], vxlap);
            store_intrin(&trg_value[5][t], vylap);
            store_intrin(&trg_value[6][t], vzlap);
        }
    }
}

GEN_KERNEL(stokes_pvellaplacian, stokes_pvellaplacian_uKernel, 4, 7, 0)

/*********************************************************
 *                                                       *
 * Stokes Vel Grad kernel, source: 3, target: 3+9        *
 *                                                       *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stokes_velgrad_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
			    Matrix<Real_t> &trg_value) {
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV = 1.0 / (8 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);

    const Real_t FACV5 = 1.0 / (8 * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Vec_t facv5 = set_intrin<Vec_t, Real_t>(FACV5);
    const Vec_t nthree = set_intrin<Vec_t, Real_t>(-3.0);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>();      // vx
            Vec_t vy = zero_intrin<Vec_t>();      // vy
            Vec_t vz = zero_intrin<Vec_t>();      // vz
            Vec_t vxgxSum = zero_intrin<Vec_t>(); // vx grad
            Vec_t vxgySum = zero_intrin<Vec_t>(); //
            Vec_t vxgzSum = zero_intrin<Vec_t>(); //
            Vec_t vygxSum = zero_intrin<Vec_t>(); // vy grad
            Vec_t vygySum = zero_intrin<Vec_t>(); //
            Vec_t vygzSum = zero_intrin<Vec_t>(); //
            Vec_t vzgxSum = zero_intrin<Vec_t>(); // vz grad
            Vec_t vzgySum = zero_intrin<Vec_t>(); //
            Vec_t vzgzSum = zero_intrin<Vec_t>(); //

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                const Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                const Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                const Vec_t tr = bcast_intrin<Vec_t>(&src_value[3][s]); // trace of doublet

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));

                Vec_t rinv = rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2);
                Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);

                Vec_t commonCoeff = mul_intrin(fx, dx);
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fy, dy));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fz, dz));

                vx = add_intrin(vx, mul_intrin(add_intrin(mul_intrin(r2, fx), mul_intrin(dx, commonCoeff)), rinv3));
                vy = add_intrin(vy, mul_intrin(add_intrin(mul_intrin(r2, fy), mul_intrin(dy, commonCoeff)), rinv3));
                vz = add_intrin(vz, mul_intrin(add_intrin(mul_intrin(r2, fz), mul_intrin(dz, commonCoeff)), rinv3));

                Vec_t vxx = zero_intrin<Vec_t>(); // vx grad
                Vec_t vxy = zero_intrin<Vec_t>(); //
                Vec_t vxz = zero_intrin<Vec_t>(); //

                Vec_t vyx = zero_intrin<Vec_t>(); // vy grad
                Vec_t vyy = zero_intrin<Vec_t>(); //
                Vec_t vyz = zero_intrin<Vec_t>(); //

                Vec_t vzx = zero_intrin<Vec_t>(); // vz grad
                Vec_t vzy = zero_intrin<Vec_t>(); //
                Vec_t vzz = zero_intrin<Vec_t>(); //

                // qij = r^2 \delta_{ij} - 3 ri rj, symmetric
                Vec_t qxx = add_intrin(r2, mul_intrin(nthree, mul_intrin(dx, dx)));
                Vec_t qxy = mul_intrin(nthree, mul_intrin(dx, dy));
                Vec_t qxz = mul_intrin(nthree, mul_intrin(dx, dz));
                Vec_t qyy = add_intrin(r2, mul_intrin(nthree, mul_intrin(dy, dy)));
                Vec_t qyz = mul_intrin(nthree, mul_intrin(dy, dz));
                Vec_t qzz = add_intrin(r2, mul_intrin(nthree, mul_intrin(dz, dz)));

                // vxx = dvx/dx , etc
                vxx = mul_intrin(qxx, commonCoeff);
                vxy = add_intrin(mul_intrin(qxy, commonCoeff),
                                 mul_intrin(r2, sub_intrin(mul_intrin(dx, fy), mul_intrin(dy, fx))));
                vxz = add_intrin(mul_intrin(qxz, commonCoeff),
                                 mul_intrin(r2, sub_intrin(mul_intrin(dx, fz), mul_intrin(dz, fx))));

                vyx = add_intrin(mul_intrin(qxy, commonCoeff),
                                 mul_intrin(r2, sub_intrin(mul_intrin(dy, fx), mul_intrin(dx, fy))));
                vyy = mul_intrin(qyy, commonCoeff);
                vyz = add_intrin(mul_intrin(qyz, commonCoeff),
                                 mul_intrin(r2, sub_intrin(mul_intrin(dy, fz), mul_intrin(dz, fy))));

                vzx = add_intrin(mul_intrin(qxz, commonCoeff),
                                 mul_intrin(r2, sub_intrin(mul_intrin(dz, fx), mul_intrin(dx, fz))));
                vzy = add_intrin(mul_intrin(qyz, commonCoeff),
                                 mul_intrin(r2, sub_intrin(mul_intrin(dz, fy), mul_intrin(dy, fz))));
                vzz = mul_intrin(qzz, commonCoeff);

                vxgxSum = add_intrin(vxgxSum, mul_intrin(vxx, rinv5));
                vxgySum = add_intrin(vxgySum, mul_intrin(vxy, rinv5));
                vxgzSum = add_intrin(vxgzSum, mul_intrin(vxz, rinv5));

                vygxSum = add_intrin(vygxSum, mul_intrin(vyx, rinv5));
                vygySum = add_intrin(vygySum, mul_intrin(vyy, rinv5));
                vygzSum = add_intrin(vygzSum, mul_intrin(vyz, rinv5));

                vzgxSum = add_intrin(vzgxSum, mul_intrin(vzx, rinv5));
                vzgySum = add_intrin(vzgySum, mul_intrin(vzy, rinv5));
                vzgzSum = add_intrin(vzgzSum, mul_intrin(vzz, rinv5));
            }

            vx = add_intrin(mul_intrin(vx, facv), load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, facv), load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, facv), load_intrin<Vec_t>(&trg_value[2][t]));

            vxgxSum = add_intrin(mul_intrin(vxgxSum, facv5), load_intrin<Vec_t>(&trg_value[3][t]));
            vxgySum = add_intrin(mul_intrin(vxgySum, facv5), load_intrin<Vec_t>(&trg_value[4][t]));
            vxgzSum = add_intrin(mul_intrin(vxgzSum, facv5), load_intrin<Vec_t>(&trg_value[5][t]));

            vygxSum = add_intrin(mul_intrin(vygxSum, facv5), load_intrin<Vec_t>(&trg_value[6][t]));
            vygySum = add_intrin(mul_intrin(vygySum, facv5), load_intrin<Vec_t>(&trg_value[7][t]));
            vygzSum = add_intrin(mul_intrin(vygzSum, facv5), load_intrin<Vec_t>(&trg_value[8][t]));

            vzgxSum = add_intrin(mul_intrin(vzgxSum, facv5), load_intrin<Vec_t>(&trg_value[9][t]));
            vzgySum = add_intrin(mul_intrin(vzgySum, facv5), load_intrin<Vec_t>(&trg_value[10][t]));
            vzgzSum = add_intrin(mul_intrin(vzgzSum, facv5), load_intrin<Vec_t>(&trg_value[11][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
            store_intrin(&trg_value[3][t], vxgxSum);
            store_intrin(&trg_value[4][t], vxgySum);
            store_intrin(&trg_value[5][t], vxgzSum);
            store_intrin(&trg_value[6][t], vygxSum);
            store_intrin(&trg_value[7][t], vygySum);
            store_intrin(&trg_value[8][t], vygzSum);
            store_intrin(&trg_value[9][t], vzgxSum);
            store_intrin(&trg_value[10][t], vzgySum);
            store_intrin(&trg_value[11][t], vzgzSum);
        }
    }
}

GEN_KERNEL(stokes_velgrad, stokes_velgrad_uKernel, 3, 12, 0)

} // namespace pvfmm
#endif // STOKESSINGLELAYERKERNEL_HPP
