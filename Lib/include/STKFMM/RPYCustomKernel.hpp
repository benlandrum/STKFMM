#ifndef INCLUDE_RPYCUSTOMKERNEL_H_
#define INCLUDE_RPYCUSTOMKERNEL_H_

#include <cmath>
#include <cstdlib>
#include <vector>

#include <pvfmm.hpp>

namespace pvfmm {

template <typename Vec_t, typename Real_t, int nwtn>
inline Vec_t rsqrt_wrapper(Vec_t r2) {
    switch (nwtn) {
    case 0:
        return rsqrt_intrin0<Vec_t, Real_t>(r2);
        break;
    case 1:
        return rsqrt_intrin1<Vec_t, Real_t>(r2);
        break;
    case 2:
        return rsqrt_intrin2<Vec_t, Real_t>(r2);
        break;
    case 3:
        return rsqrt_intrin3<Vec_t, Real_t>(r2);
        break;
    default:
        break;
    }
}

/**********************************************************
 *                                                        *
 *     Stokes P Vel kernel, source: 4, target: 4          *
 *                                                        *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void rpy_u_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value,
                   Matrix<Real_t> &trg_coord, Matrix<Real_t> &trg_value) {

#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    Vec_t FACV = set_intrin<Vec_t, Real_t>(1.0 / (8 * const_pi<Real_t>()));
    Vec_t nwtn_factor = set_intrin<Vec_t, Real_t>(1.0 / nwtn_scal);
    Vec_t three = set_intrin<Vec_t, Real_t>(3.0);

    const Vec_t one_over_three =
        set_intrin<Vec_t, Real_t>(static_cast<Real_t>(1.0 / 3.0));

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

                const Vec_t a2_over_three =
                    mul_intrin(mul_intrin(a, a), one_over_three);
                const Vec_t r2 = add_intrin(
                    add_intrin(mul_intrin(dx, dx), mul_intrin(dy, dy)),
                    mul_intrin(dz, dz));

                const Vec_t rinv = mul_intrin(
                    rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2), nwtn_factor);
                const Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                const Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);
                const Vec_t fdotr = add_intrin(
                    add_intrin(mul_intrin(fx, dx), mul_intrin(fy, dy)),
                    mul_intrin(fz, dz));

                vx =
                    add_intrin(vx, mul_intrin(add_intrin(mul_intrin(r2, fx),
                                                         mul_intrin(dx, fdotr)),
                                              rinv3));
                vy =
                    add_intrin(vy, mul_intrin(add_intrin(mul_intrin(r2, fy),
                                                         mul_intrin(dy, fdotr)),
                                              rinv3));
                vz =
                    add_intrin(vz, mul_intrin(add_intrin(mul_intrin(r2, fz),
                                                         mul_intrin(dz, fdotr)),
                                              rinv3));
                const Vec_t three_fdotr_rinv5 =
                    mul_intrin(mul_intrin(three, fdotr), rinv5);

                vx = add_intrin(
                    vx,
                    mul_intrin(a2_over_three,
                               sub_intrin(mul_intrin(fx, rinv3),
                                          mul_intrin(three_fdotr_rinv5, dx))));
                vy = add_intrin(
                    vy,
                    mul_intrin(a2_over_three,
                               sub_intrin(mul_intrin(fy, rinv3),
                                          mul_intrin(three_fdotr_rinv5, dy))));
                vz = add_intrin(
                    vz,
                    mul_intrin(a2_over_three,
                               sub_intrin(mul_intrin(fz, rinv3),
                                          mul_intrin(three_fdotr_rinv5, dz))));
            }

            vx = add_intrin(mul_intrin(vx, FACV),
                            load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, FACV),
                            load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, FACV),
                            load_intrin<Vec_t>(&trg_value[2][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
        }
    }
#undef SRC_BLK
}

/**********************************************************
 *                                                        *
 * RPY Force,a Vel kernel,source: 4, target: 6            *
 *       fx,fy,fz,a -> ux,uy,uz,lapux,lapuy,lapuz         *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void rpy_ulapu_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value,
                       Matrix<Real_t> &trg_coord, Matrix<Real_t> &trg_value) {

#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Vec_t FACV =
        set_intrin<Vec_t, Real_t>(1.0 / (8 * const_pi<Real_t>()));
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

                const Vec_t a2_over_three =
                    mul_intrin(mul_intrin(a, a), one_over_three);
                const Vec_t r2 = add_intrin(
                    add_intrin(mul_intrin(dx, dx), mul_intrin(dy, dy)),
                    mul_intrin(dz, dz));

                const Vec_t rinv = mul_intrin(
                    rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2), nwtn_factor);
                const Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                const Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);
                const Vec_t fdotr = add_intrin(
                    add_intrin(mul_intrin(fx, dx), mul_intrin(fy, dy)),
                    mul_intrin(fz, dz));

                const Vec_t three_fdotr_rinv5 =
                    mul_intrin(mul_intrin(three, fdotr), rinv5);
                const Vec_t cx = sub_intrin(mul_intrin(fx, rinv3),
                                            mul_intrin(three_fdotr_rinv5, dx));
                const Vec_t cy = sub_intrin(mul_intrin(fy, rinv3),
                                            mul_intrin(three_fdotr_rinv5, dy));
                const Vec_t cz = sub_intrin(mul_intrin(fz, rinv3),
                                            mul_intrin(three_fdotr_rinv5, dz));

                const Vec_t fdotr_rinv3 = mul_intrin(fdotr, rinv3);
                vx = add_intrin(
                    vx, add_intrin(mul_intrin(fx, rinv),
                                   add_intrin(mul_intrin(dx, fdotr_rinv3),
                                              mul_intrin(a2_over_three, cx))));
                vy = add_intrin(
                    vy, add_intrin(mul_intrin(fy, rinv),
                                   add_intrin(mul_intrin(dy, fdotr_rinv3),
                                              mul_intrin(a2_over_three, cy))));
                vz = add_intrin(
                    vz, add_intrin(mul_intrin(fz, rinv),
                                   add_intrin(mul_intrin(dz, fdotr_rinv3),
                                              mul_intrin(a2_over_three, cz))));

                lapvx = add_intrin(lapvx, mul_intrin(two, cx));
                lapvy = add_intrin(lapvy, mul_intrin(two, cy));
                lapvz = add_intrin(lapvz, mul_intrin(two, cz));
            }

            vx = add_intrin(mul_intrin(vx, FACV),
                            load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, FACV),
                            load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, FACV),
                            load_intrin<Vec_t>(&trg_value[2][t]));
            lapvx = add_intrin(mul_intrin(lapvx, FACV),
                               load_intrin<Vec_t>(&trg_value[3][t]));
            lapvy = add_intrin(mul_intrin(lapvy, FACV),
                               load_intrin<Vec_t>(&trg_value[4][t]));
            lapvz = add_intrin(mul_intrin(lapvz, FACV),
                               load_intrin<Vec_t>(&trg_value[5][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
            store_intrin(&trg_value[3][t], lapvx);
            store_intrin(&trg_value[4][t], lapvy);
            store_intrin(&trg_value[5][t], lapvz);
        }
    }
#undef SRC_BLK
}

/**********************************************************
 *                                                        *
 * Stokes Force Vel,lapVel kernel,source: 3, target: 6    *
 *       fx,fy,fz -> ux,uy,uz,lapux,lapuy,lapuz           *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, size_t NWTN_ITER>
void stk_ulapu_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value,
                       Matrix<Real_t> &trg_coord, Matrix<Real_t> &trg_value) {

#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);
    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Vec_t FACV =
        set_intrin<Vec_t, Real_t>(1.0 / (8 * const_pi<Real_t>()));
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

                const Vec_t r2 = add_intrin(
                    add_intrin(mul_intrin(dx, dx), mul_intrin(dy, dy)),
                    mul_intrin(dz, dz));

                const Vec_t rinv = mul_intrin(
                    rsqrt_wrapper<Vec_t, Real_t, NWTN_ITER>(r2), nwtn_factor);
                const Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);
                const Vec_t rinv5 = mul_intrin(mul_intrin(rinv, rinv), rinv3);
                const Vec_t fdotr = add_intrin(
                    add_intrin(mul_intrin(fx, dx), mul_intrin(fy, dy)),
                    mul_intrin(fz, dz));

                const Vec_t three_fdotr_rinv5 =
                    mul_intrin(mul_intrin(three, fdotr), rinv5);
                const Vec_t cx = sub_intrin(mul_intrin(fx, rinv3),
                                            mul_intrin(three_fdotr_rinv5, dx));
                const Vec_t cy = sub_intrin(mul_intrin(fy, rinv3),
                                            mul_intrin(three_fdotr_rinv5, dy));
                const Vec_t cz = sub_intrin(mul_intrin(fz, rinv3),
                                            mul_intrin(three_fdotr_rinv5, dz));

                const Vec_t fdotr_rinv3 = mul_intrin(fdotr, rinv3);
                vx = add_intrin(vx, add_intrin(mul_intrin(fx, rinv),
                                               mul_intrin(dx, fdotr_rinv3)));
                vy = add_intrin(vy, add_intrin(mul_intrin(fy, rinv),
                                               mul_intrin(dy, fdotr_rinv3)));
                vz = add_intrin(vz, add_intrin(mul_intrin(fz, rinv),
                                               mul_intrin(dz, fdotr_rinv3)));

                lapvx = add_intrin(lapvx, mul_intrin(two, cx));
                lapvy = add_intrin(lapvy, mul_intrin(two, cy));
                lapvz = add_intrin(lapvz, mul_intrin(two, cz));
            }

            vx = add_intrin(mul_intrin(vx, FACV),
                            load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, FACV),
                            load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, FACV),
                            load_intrin<Vec_t>(&trg_value[2][t]));
            lapvx = add_intrin(mul_intrin(lapvx, FACV),
                               load_intrin<Vec_t>(&trg_value[3][t]));
            lapvy = add_intrin(mul_intrin(lapvy, FACV),
                               load_intrin<Vec_t>(&trg_value[4][t]));
            lapvz = add_intrin(mul_intrin(lapvz, FACV),
                               load_intrin<Vec_t>(&trg_value[5][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
            store_intrin(&trg_value[3][t], lapvx);
            store_intrin(&trg_value[4][t], lapvy);
            store_intrin(&trg_value[5][t], lapvz);
        }
    }
#undef SRC_BLK
}

#if defined __MIC__
#define Vec_ts Real_t
#define Vec_td Real_t
#elif defined __AVX__
#define Vec_ts __m256
#define Vec_td __m256d
#elif defined __SSE3__
#define Vec_ts __m128
#define Vec_td __m128d
#else
#define Vec_ts Real_t
#define Vec_td Real_t
#endif

#define GEN_KERNEL_HELPER(MICROKERNEL, SRCDIM, TARDIM, VEC_T, REAL_T)          \
    generic_kernel<REAL_T, SRCDIM, TARDIM,                                     \
                   MICROKERNEL<REAL_T, VEC_T, newton_iter>>(                   \
        (REAL_T *)r_src, src_cnt, (REAL_T *)v_src, dof, (REAL_T *)r_trg,       \
        trg_cnt, (REAL_T *)v_trg, mem_mgr)

#define GEN_KERNEL(KERNEL, MICROKERNEL, SRCDIM, TARDIM)                        \
    template <class T, int newton_iter = 0>                                    \
    void KERNEL(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg,            \
                int trg_cnt, T *v_trg, mem::MemoryManager *mem_mgr) {          \
                                                                               \
        if (mem::TypeTraits<T>::ID() == mem::TypeTraits<float>::ID()) {        \
            typedef float Real_t;                                              \
            GEN_KERNEL_HELPER(MICROKERNEL, SRCDIM, TARDIM, Vec_ts, Real_t);    \
        } else if (mem::TypeTraits<T>::ID() ==                                 \
                   mem::TypeTraits<double>::ID()) {                            \
            typedef double Real_t;                                             \
            GEN_KERNEL_HELPER(MICROKERNEL, SRCDIM, TARDIM, Vec_td, Real_t);    \
        } else {                                                               \
            typedef T Real_t;                                                  \
            GEN_KERNEL_HELPER(MICROKERNEL, SRCDIM, TARDIM, Real_t, Real_t);    \
        }                                                                      \
    }

GEN_KERNEL(rpy_u, rpy_u_uKernel, 4, 3)
GEN_KERNEL(rpy_ulapu, rpy_ulapu_uKernel, 4, 6)
GEN_KERNEL(stk_ulapu, stk_ulapu_uKernel, 3, 6)

#undef Vec_ts
#undef Vec_td
#undef GEN_KERNEL
#undef GEN_KERNEL_HELPER

template <class T>
struct RPYTestKernel {
    inline static const Kernel<T> &ulapu(); //   3+1->6
  private:
    static constexpr int NEWTON_ITE = sizeof(T) / 4;
};

// 1 newton for float, 2 newton for double
// the string for stk_ker must be exactly the same as in kernel.txx of pvfmm
template <class T>
inline const Kernel<T> &RPYTestKernel<T>::ulapu() {

    static Kernel<T> g_ker = StokesKernel<T>::velocity();
    static Kernel<T> gr_ker = BuildKernel<T, rpy_u<T, NEWTON_ITE>>(
        "rpy_u", 3, std::pair<int, int>(4, 3));

    static Kernel<T> glapg_ker = BuildKernel<T, stk_ulapu<T, NEWTON_ITE>>(
        "stk_ulapu", 3, std::pair<int, int>(3, 6));
    // glapg_ker.surf_dim = 3;

    static Kernel<T> grlapgr_ker = BuildKernel<T, rpy_ulapu<T, NEWTON_ITE>>(
        "rpy_ulapu", 3, std::pair<int, int>(4, 6),
        &gr_ker,    // k_s2m
        &gr_ker,    // k_s2l
        NULL,       // k_s2t
        &g_ker,     // k_m2m
        &g_ker,     // k_m2l
        &glapg_ker, // k_m2t
        &g_ker,     // k_l2l
        &glapg_ker, // k_l2t
        NULL);
    // grlapgr_ker.surf_dim = 4;
    return grlapgr_ker;
}

} // namespace pvfmm

#endif
