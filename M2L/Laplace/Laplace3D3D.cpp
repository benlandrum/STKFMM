/*
 * LaplaceM2L.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: wyan
 */

#include "SVD_pvfmm.hpp"
#include <random>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

namespace Laplace3D3D {
  const bool NEUTRALIZE_FOR_M2C = true;
  //const bool NEUTRALIZE_M_EQUIVALENT = true;

  const double MADELUNG = 2.837297479;

/*******************************************
 *
 *    Laplace Potential Summation
 *
 *********************************************/
inline double freal(double xi, double r) { return std::erfc(xi * r) / r; }

inline double frealp(double xi, double r) {
    return -(2. * exp(-r * r * (xi * xi)) * xi) / (sqrt(M_PI) * r) - std::erfc(r * xi) / (r * r);
}

// xm: target, xn: source
inline double realSum(const double xi, const EVec3 &xn, const EVec3 &xm) {
    EVec3 rmn = xm - xn;
    double rnorm = rmn.norm();
    if (rnorm < eps) {
        return 0;
    }
    return freal(xi, rnorm);
}

inline double gKernelEwald(const EVec3 &xm, const EVec3 &xn) {
    const double xi = 2; // recommend for box=1 to get machine precision
    EVec3 target = xm;
    EVec3 source = xn;
    target[0] = target[0] - floor(target[0]); // periodic BC
    target[1] = target[1] - floor(target[1]);
    target[2] = target[2] - floor(target[2]);
    source[0] = source[0] - floor(source[0]);
    source[1] = source[1] - floor(source[1]);
    source[2] = source[2] - floor(source[2]);

    // real sum

    // bjlbjl this did not change
    int rLim = 4;
    //int rLim = 8;


    double Kreal = 0;
    for (int i = -rLim; i <= rLim; i++) {
        for (int j = -rLim; j <= rLim; j++) {
            for (int k = -rLim; k <= rLim; k++) {
                Kreal += realSum(xi, target, source - EVec3(i, j, k));
            }
        }
    }

    // wave sum
    //// bjlbjl this did not change
    int wLim = 4;
    //int wLim = 8;


    double Kwave = 0;
    EVec3 rmn = target - source;
    const double xi2 = xi * xi;
    const double rmnnorm = rmn.norm();
    for (int i = -wLim; i <= wLim; i++) {
        for (int j = -wLim; j <= wLim; j++) {
            for (int k = -wLim; k <= wLim; k++) {
                if (i == 0 && j == 0 && k == 0) {
                    continue;
                }
                EVec3 kvec = EVec3(i, j, k) * (2 * M_PI);
                double k2 = kvec.dot(kvec);
                Kwave += 4 * M_PI * cos(kvec.dot(rmn)) * exp(-k2 / (4 * xi2)) / k2;
            }
        }
    }

    double Kself = rmnnorm < 1e-10 ? -2 * xi / sqrt(M_PI) : 0;

    return (Kreal + Kwave + Kself - M_PI / xi2) / (4 * M_PI);
}

inline double gKernel(const EVec3 &target, const EVec3 &source) {
    EVec3 rst = target - source;
    double rnorm = rst.norm();
    return rnorm < eps ? 0 : 1 / (4 * M_PI * rnorm);
}

inline double gKernelNF(const EVec3 &target, const EVec3 &source, int N = DIRECTLAYER) {
    double gNF = 0;
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                gNF += gKernel(target, source + EVec3(i, j, k));
            }
        }
    }
    return gNF;
}

// Out of Direct Sum Layer, far field part
inline double gKernelFF(const EVec3 &target, const EVec3 &source) {
    double fEwald = gKernelEwald(target, source);
    fEwald -= gKernelNF(target, source);
    return fEwald;
}

/*******************************************
 *
 *    Laplace Grad Summation
 *
 *********************************************/

inline void realGradSum(double xi, const EVec3 &target, const EVec3 &source, EVec3 &v) {
    EVec3 rvec = target - source;
    double rnorm = rvec.norm();
    if (rnorm < eps) {
        v.setZero();
    } else {
        v = (frealp(xi, rnorm) / rnorm) * rvec;
    }
}

inline void gradkernel(const EVec3 &target, const EVec3 &source, EVec3 &answer) {
    // grad of Laplace potential
    EVec3 rst = target - source;
    double rnorm = rst.norm();
    if (rnorm < eps) {
        answer.setZero();
        return;
    }
    double rnorm3 = rnorm * rnorm * rnorm;
    answer = -rst / (rnorm3 * 4 * M_PI);
}

inline void gradEwald(const EVec3 &target_, const EVec3 &source_, EVec3 &answer) {
    // grad of Laplace potential, periodic of -r_k/r^3
    EVec3 target = target_;
    EVec3 source = source_;
    target[0] = target[0] - floor(target[0]); // periodic BC
    target[1] = target[1] - floor(target[1]);
    target[2] = target[2] - floor(target[2]);
    source[0] = source[0] - floor(source[0]);
    source[1] = source[1] - floor(source[1]);
    source[2] = source[2] - floor(source[2]);

    double xi = 0.54;

    // real sum
    int rLim = 10;
    EVec3 Kreal = EVec3::Zero();
    for (int i = -rLim; i < rLim + 1; i++) {
        for (int j = -rLim; j < rLim + 1; j++) {
            for (int k = -rLim; k < rLim + 1; k++) {
                EVec3 v = EVec3::Zero();
                realGradSum(xi, target, source + EVec3(i, j, k), v);
                Kreal += v;
            }
        }
    }

    // wave sum
    int wLim = 10;
    EVec3 rmn = target - source;
    double xi2 = xi * xi;
    EVec3 Kwave(0., 0., 0.);
    for (int i = -wLim; i < wLim + 1; i++) {
        for (int j = -wLim; j < wLim + 1; j++) {
            for (int k = -wLim; k < wLim + 1; k++) {
                if (i == 0 && j == 0 && k == 0)
                    continue;
                EVec3 kvec = EVec3(i, j, k) * (2 * M_PI);
                double k2 = kvec.dot(kvec);
                double knorm = kvec.norm();
                Kwave += -kvec * (sin(kvec.dot(rmn)) * exp(-k2 / (4 * xi2)) / k2);
            }
        }
    }

    answer = (Kreal + Kwave) / (4 * M_PI);
}

inline EVec4 ggradKernel(const EVec3 &target, const EVec3 &source) {
    EVec3 rst = target - source;
    EVec4 pgrad = EVec4::Zero();
    double rnorm = rst.norm();
    if (rnorm < eps) {
        pgrad.setZero();
    } else {
        pgrad[0] = 1 / rnorm;
        double rnorm3 = rnorm * rnorm * rnorm;
        pgrad.block<3, 1>(1, 0) = -rst / rnorm3;
    }
    return pgrad / (4 * M_PI);
}

inline EVec4 ggradKernelEwald(const EVec3 &target, const EVec3 &source) {
    EVec4 pgrad = EVec4::Zero();
    pgrad[0] = gKernelEwald(target, source);
    EVec3 grad;
    gradEwald(target, source, grad);
    pgrad[1] = grad[0];
    pgrad[2] = grad[1];
    pgrad[3] = grad[2];
    return pgrad;
}

inline EVec4 ggradKernelNF(const EVec3 &target, const EVec3 &source, int N = DIRECTLAYER) {
    EVec4 gNF = EVec4::Zero();
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                EVec4 gFree = ggradKernel(target, source + EVec3(i, j, k));
                gNF += gFree;
            }
        }
    }
    return gNF;
}

// Out of Direct Sum Layer, far field part
inline EVec4 ggradKernelFF(const EVec3 &target, const EVec3 &source) {
    EVec4 fEwald = ggradKernelEwald(target, source);
    fEwald -= ggradKernelNF(target, source);
    return fEwald;
}

// Go through x and y for constant z.
// Plot the field value to the output z coordinate.
template<typename Field>
void PlotField(Field&& callable, const std::string& name) {
  int pixels = 400;
  double field_cutoff = 1;

  double min_coord = -1;
  double max_coord = 2;

  EVec3 source(0.5, 0.5, 0.5);

  std::vector<std::vector<double>> x, y, z;
  for (int i=0; i<pixels; ++i) {
    std::vector<double> x_row, y_row, z_row;
    double x_coord = min_coord + (max_coord - min_coord) * i / pixels;
    for (int j=0; j<pixels; ++j) {
      double y_coord = min_coord + (max_coord - min_coord) * j / pixels;
      x_row.push_back(x_coord);
      y_row.push_back(y_coord);
      EVec3 target(x_coord, y_coord, 0.5);
      auto field = callable(target, source);
      if (field > field_cutoff || field < -field_cutoff)
	field = 0;
      z_row.push_back(field);
    }
    x.push_back(std::move(x_row));
    y.push_back(std::move(y_row));
    z.push_back(std::move(z_row));
  }

  /*
  int ncols = pixels, nrows = pixels;
  std::vector<float> z(ncols * nrows);
  for (int j=0; j<nrows; ++j) {
    double y_coord = min_coord + (max_coord - min_coord) * j / nrows;
    for (int i=0; i<ncols; ++i) {
      double x_coord = min_coord + (max_coord - min_coord) * i / ncols;
      EVec3 target(x_coord, y_coord, 0.5);
      auto field = callable(target, source);
      if (field > field_cutoff || field < -field_cutoff)
	field = 0;
      z[ncols * j + i] = field;
    }
  }
  */
  
  /*
  int colors = 1;
  PyObject* mat;
  plt::imshow(&z[0], nrows, ncols, colors, {}, &mat);
  plt::colorbar(mat);
  */

  plt::clf();
  plt::contour(x, y, z);
  plt::set_aspect_equal();
  plt::title(name);
  plt::save(name + std::string(".png"));
  plt::clf();
}

template<typename Surface>
void PlotField(const Surface& pointLEquiv,
	       const Surface& pointMEquiv,
	       const EVec& Msource,
	       const EVec& M2Lsource) {
  //PlotField([](const EVec3& target, const EVec3& source){ return gKernel(target, source); }, "point_free");
  PlotField([](const EVec3& target, const EVec3& source){ return gKernelEwald(target, source); }, "point_ewald");
  PlotField([](const EVec3& target, const EVec3& source){ return gKernelEwald(target, source) - gKernel(target, source); }, "point_ewald_diff_point_free");
  // auto equiv_value = [&](const EVec3& target,
  // 			 const EVec3& source,
  // 			 const auto& equiv_points,
  // 			 const auto& equiv_sources,
  // 			 auto& callable) {
  //   auto num_pts = equiv_sources.size();
  //   if (3 * num_pts != equiv_points.size())
  //     throw std::runtime_error(std::string("size mismatch ")
  // 			       + std::to_string(num_pts)
  // 			       + std::string(" ")
  // 			       + std::to_string(equiv_points.size()));

  //   double value = 0;
  //   for (int i=0; i<num_pts; ++i) {
  //     EVec3 center(0.5, 0.5, 0.5);
  //     EVec3 m_point(equiv_points[3 * i], equiv_points[3 * i + 1], equiv_points[3 * i + 2]);
  //     EVec3 delta = m_point - center;
  //     EVec3 m_source = source + delta;
  //     value += callable(target, m_source) * equiv_sources[i];
  //   }
  //   return value;
  // };

  // auto m_equiv_value = [&](const EVec3& target,
  // 			   const EVec3& source,
  // 			   auto& callable) {
  //   return equiv_value(target, source,
  // 		       pointMEquiv,
  // 		       Msource,
  // 		       callable);
  // };

  // auto l_equiv_value = [&](const EVec3& target,
  // 			   const EVec3& source,
  // 			   auto& callable) {
  //   return equiv_value(target, source,
  // 		       pointLEquiv,
  // 		       M2Lsource,
  // 		       callable);
  // };

  // auto free_space = [](const EVec3& target, const EVec3& source) {
  //   return gKernel(target, source);
  // };
  // auto ewald = [](const EVec3& target, const EVec3& source) {
  //   return gKernelEwald(target, source);
  // };

  // PlotField([&](const EVec3& target, const EVec3& source) { return m_equiv_value(target, source, free_space); }, "mequiv_free");
  // PlotField([&](const EVec3& target, const EVec3& source) { return m_equiv_value(target, source, ewald); }, "mequiv_ewald");
  // PlotField([&](const EVec3& target, const EVec3& source) { return m_equiv_value(target, source, free_space) - gKernel(target, source); }, "mequiv_free_diff_point_free");
  // PlotField([&](const EVec3& target, const EVec3& source) { return m_equiv_value(target, source, ewald) - m_equiv_value(target, source, free_space); }, "mequiv_ewald_diff_mequiv_free");
  // PlotField([&](const EVec3& target, const EVec3& source) { return m_equiv_value(target, source, ewald) - gKernelEwald(target, source); }, "mequiv_ewald_diff_point_ewald");

  // PlotField([&](const EVec3& target, const EVec3& source) {
  //   return m_equiv_value(target, source, free_space)
  //     + l_equiv_value(target, source, free_space)
  //     - m_equiv_value(target, source, ewald);
  // },
  //   "error_vs_ewald");
}

bool NumericalLaplacian(const EVec3& target,
			const EVec3& source,
			double delta,
			double& result) {
  result = 0;

  // Adding a doubling fudge factor.
  double fudge = 2e-10;

  if ((target-source).norm() < fudge) return false;

  result += -6 * gKernelEwald(target, source);
  for (auto& offset : {
      EVec3(-delta, 0, 0),
      EVec3(+delta, 0, 0),
      EVec3(0, -delta, 0),
      EVec3(0, +delta, 0),
      EVec3(0, 0, -delta),
      EVec3(0, 0, +delta)}) {
    auto new_target = target + offset;
    auto x = source - new_target;

    // Adding a doubling fudge factor.
    if (x.norm() < fudge) {
      result = 0;
      return false;
    }
    result += gKernelEwald(new_target, source);
  }
  result /= (delta * delta);
  return true;
}

// The max error is about 10^{-4} here.
// The Ewald solution is fine.
void TestNumericalLaplacian() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  EVec3 source(0.2, 0.1, 0.9);

  double expected(1.0);

  double max_error = 0.0;
  EVec3 max_error_point(0.0, 0.0, 0.0);
  int num_points = 10000;
  for (int i = 0; i < num_points; ++i) {
    auto x = dis(gen);
    auto y = dis(gen);
    auto z = dis(gen);
    EVec3 target(x, y, z);
    double delta = 0.00001;

    double value = 0;
    if (!NumericalLaplacian(target, source, delta, value)) continue;

    auto error = std::abs(value - expected);
    if (error > max_error) {
      max_error = error;
      max_error_point = target;
    }
  }
  std::cout << "Laplacian max error: " << max_error
	    << " at target " << max_error_point.transpose()
	    << " for source " << source.transpose()
	    << std::endl;
}

// This proves that translation variance isn't a problem.
void TestTranslationInvariance() {
  int n = 5;
  std::vector<double> errors(3 * n * n * n);
  int m = 0;
  for (auto& sourceToTarget : {EVec3(-0.1, 0.1, -0.5),
			       EVec3(0.8, 1.2, 3.4)}) {
    auto noOffset = gKernelEwald(sourceToTarget, EVec3::Zero());
    for (int i=0; i<n; ++i) {
      for (int j=0; j<n; ++j) {
	for (int k=0; k<n; ++k) {
	  EVec3 source(double(i)/n, double(j)/n, double(k)/n);
	  errors[m++] = std::abs(gKernelEwald(sourceToTarget + source, source) - noOffset);
	}
      }
    }
  }
  // This is 5.20417e-17.
  std::cout << "bjlbjl translation invariance max delta: "
	    << *std::max_element(errors.begin(), errors.end())
	    << std::endl;
}

void TestZeroAverage() {
  int count = 10000000;
  // 1000000: 0.000175398
  // 10000000:

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  double average = 0.0;

  EVec3 source(0.2, 0.1, 0.8);
  for (int i=0; i<count; ++i) {
    auto x = dis(gen);
    auto y = dis(gen);
    auto z = dis(gen);
    EVec3 target(x, y, z);
    auto value = gKernelEwald(target, source);
    average = i / (i+1.) * average + 1 / (i+1.) * value;
  }
  std::cout << "zero average? " << average << std::endl;
}

template<class Surface>
void PrintMomentStats(int equivN,
		      double diam,
		      const Surface& surfacePoints,
		      const EVec& surfaceCharges,
		      const std::string& name) {
  // The charge is nearly one, as expected.
  double sum_m_source = 0.0;
  EVec3 sum_m_dipole(0.0, 0.0, 0.0);
  EMat3 sum_degen_quad = EMat3::Zero();

  // https://en.wikipedia.org/wiki/Multipole_expansion#Multipole_expansion_of_a_potential_outside_an_electrostatic_charge_distribution
  for (int p = 0; p < equivN; ++p) {
    EVec3 Mpoint(surfacePoints[3 * p], surfacePoints[3 * p + 1], surfacePoints[3 * p + 2]);
    sum_m_source += surfaceCharges[p];
    EVec3 displacement = Mpoint - EVec3(0.5, 0.5, 0.5);
    sum_m_dipole += surfaceCharges[p] * displacement;
    sum_degen_quad += surfaceCharges[p] * (3 * displacement * displacement.transpose() - displacement.dot(displacement) * EMat3::Identity());
  }
  std::cout << "bjlbjl moments "
	    << name
	    << " mono "
	    << sum_m_source
	    << " scaled dipole " << sum_m_dipole.transpose() / sum_m_source
	    << " scaled quad \n" << sum_degen_quad / sum_m_source
	    << std::endl;
}

template<class Surface>
void TestOffset(const EVec3& offset,
		int equivN,
		const Surface& pointMEquiv,
		const EVec& Msource) {
  EVec3 center(0.5, 0.5, 0.5);
  auto target = center + offset;

  double sum_m_field = 0.0;
  for (int p = 0; p < equivN; ++p) {
    EVec3 Mpoint(pointMEquiv[3 * p], pointMEquiv[3 * p + 1], pointMEquiv[3 * p + 2]);
    sum_m_field += Msource[p] * gKernel(target, Mpoint);
  }

  auto direct = gKernel(target, center);

  std::cout << "bjlbjl offset "
	    << offset.transpose()
	    << " sum "
	    << sum_m_field
	    << " direct "
	    << direct
	    << " sum-direct "
	    << (sum_m_field - direct)
	    << std::endl;
}

template<class Surface>
void TestZeroOffSum(int equivN,
		    const Surface& pointMEquiv,
		    const EVec& Msource) {
  // This matches the DIRECTLAYER=0 (UNFM2T - UNFS2T): 0.132437.
  // sum-direct 0.132437.
  // This means we ARE computing the UNF correctly.
  TestOffset(EVec3::Zero(), equivN, pointMEquiv, Msource);
}

template<class Surface>
void TestOneOffSum(int equivN,
		   const Surface& pointMEquiv,
		   const EVec& Msource) {
  // This means that the difference HAS to be in the zero box.
  // We see consistent errors of about 10e-6 here.
  for (int sign : {-1, 1}) {
    for (int dir = 0; dir < 3; ++dir) {
      EVec3 offset(0.0, 0.0, 0.0);
      offset[dir] = sign;
      TestOffset(offset, equivN, pointMEquiv, Msource);
    }
  }
}

template<class Surface>
void TestPoints(int numSample,
		const std::vector<EVec3, Eigen::aligned_allocator<EVec3>>& samplePoints,
		const EVec3& sourcePoint,
		double sourceCharge,
		int equivN,
		const Surface& pointLEquiv,
		const Surface& pointMEquiv,
		const EVec& Msource,
		const EVec& M2Lsource) {
  std::vector<std::pair<int, double>> errors(numSample);

  for (int ii = 0; ii < numSample; ++ii) {
    auto& samplePoint = samplePoints[ii];

    EVec4 UFFS2T = ggradKernelFF(samplePoint, sourcePoint) * sourceCharge;
    EVec4 UNF = ggradKernelNF(samplePoint, sourcePoint) * sourceCharge;
    EVec4 UEwald = ggradKernelEwald(samplePoint, sourcePoint) * sourceCharge;

    EVec4 UFFL2T = EVec4::Zero();
    EVec4 UFFM2T = EVec4::Zero();

    EVec4 UNFM2T = EVec4::Zero();
    EVec4 UEwaldM2T = EVec4::Zero();

#pragma omp sections
    {
      // Loop over Le points with their equivalent charges.
      // Calculate their free-space sum at the target point.
      // THIS USES M2L.
#pragma omp section
      for (int p = 0; p < equivN; p++) {
	EVec3 Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1], pointLEquiv[3 * p + 2]);
	UFFL2T += ggradKernel(samplePoint, Lpoint) * M2Lsource[p];
      }
      // Loop over Me points with their equivalent charges.
      // Calculate their far-field sum at the target point.
      // THIS DOES NOT USE THE M2L, ONLY THE M-EQUIVALENT SOURCES.
#pragma omp section
      for (int p = 0; p < equivN; p++) {
	EVec3 Mpoint(pointMEquiv[3 * p], pointMEquiv[3 * p + 1], pointMEquiv[3 * p + 2]);
	UFFM2T += ggradKernelFF(samplePoint, Mpoint) * Msource[p];
	UNFM2T += ggradKernelNF(samplePoint, Mpoint) * Msource[p];
	UEwaldM2T += ggradKernelEwald(samplePoint, Mpoint) * Msource[p];
      }
    }

    errors[ii].first = ii;
    errors[ii].second = (UFFM2T - UFFS2T)[0];
    std::cout << "UFFM2T: " << UFFM2T[0]
	      << ", UFFS2T: " << UFFS2T[0]
	      << ", UEwaldM2T: " << UEwaldM2T[0]
	      << ", UEwaldS2T: " << UEwald[0]
	      << ", UNFM2T: " << UNFM2T[0]
	      << ", UNFS2T: " << UNF[0]
	      << ", (UFFM2T - UFFS2T): " << (UFFM2T - UFFS2T)[0]
	      << ", (UEwaldM2T - UEwaldS2T): " << UEwaldM2T[0] - UEwald[0]
	      << ", (UNFM2T - UNFS2T): " << UNFM2T[0] - UNF[0]
	      << std::endl;

    /*
      std::cout << "===============================================" << std::endl;
      if (NEUTRALIZE_FOR_M2C) {
      std::cout << "++++ NEUTRALIZE ----" << std::endl;
      }
      else {
      std::cout << "NO NEUTRALIZATION" << std::endl;
      }
      std::cout << "UFF S2T: " << UFFS2T.transpose() << std::endl;
      std::cout << "UFF M2T: " << UFFM2T.transpose() << std::endl;
      std::cout << "UFF L2T: " << UFFL2T.transpose() << std::endl;
      std::cout << "Error M2T: " << (UFFM2T - UFFS2T).transpose() << std::endl;
      std::cout << "Error L2T: " << (UFFL2T - UFFS2T).transpose() << std::endl;

      std::cout << "UNF: " << UNF.transpose() << std::endl;
      std::cout << "UEwald: " << UEwald.transpose() << std::endl;

      // This is the same thing as the Error L2T.
      std::cout << "Error vs. Ewald: " << (UNF + UFFL2T - UEwald).transpose() << std::endl;
    */
  }

  std::sort(errors.begin(), errors.end(),
	    [](auto& a, auto& b) {
	      if (abs(a.second) != abs(b.second))
		return abs(a.second) < abs(b.second);
	      return a.first < b.first;
	    });
  // Print the errors.
  // This is roughly constant for the lone charge: 6.202233e-02.
  // But it is slightly worse at the charge point itself.
  // Their neutralization doesn't improve things.

  // Also, the error is positive.
  // 1) Both the UFFM2T and the UFFS2T are negative, as expected for the far field.
  // 2) The S2T is more negative than the M2T.
  //    a) So the M2T has more positive sources?
  //       Or maybe the M2T has closer sources?
  for (int ii = 0; ii < numSample; ++ii) {
    auto& element = errors[ii];
    auto index = element.first;
    auto error = element.second;
    std::cout << "Error " << error << " | " << samplePoints[index].transpose() << std::endl;
  }
}

int main(int argc, char **argv) {
    Eigen::initParallel();
    Eigen::setNbThreads(1);
    constexpr int kdim[2] = {1, 1}; // target, source dimension

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    const int pEquiv = atoi(argv[1]);
    const int pCheck = atoi(argv[1]);

    TestTranslationInvariance();
    //TestZeroAverage();
    TestNumericalLaplacian();

    // For 6 per side.
    // 0.1: 0.00057
    // 0.2: 0.00225
    // 0.5: 0.01406
    // 1.0: 0.05625
    // 1.05 (scaleIn): 0.06201
    // 1.5: 0.1256131

    // For 12 per side.
    // 0.1: 0.0005
    // 0.2: 0.002248
    // 0.5: 0.014054
    // 1.0: 0.056214
    // 1.05 (scaleIn): 0.06198
    // 1.5: 0.1227277
    

    double scaleIn2(1.05);

    const double pCenterMEquiv[3] = {-(scaleIn2 - 1) / 2, -(scaleIn2 - 1) / 2, -(scaleIn2 - 1) / 2};
    const double pCenterMCheck[3] = {-(scaleOut - 1) / 2, -(scaleOut - 1) / 2, -(scaleOut - 1) / 2};

    const double pCenterLEquiv[3] = {-(scaleOut - 1) / 2, -(scaleOut - 1) / 2, -(scaleOut - 1) / 2};
    const double pCenterLCheck[3] = {-(scaleIn2 - 1) / 2, -(scaleIn2 - 1) / 2, -(scaleIn2 - 1) / 2};

    // Important: Both check _and_ equivalent are beyond the boundaries of the unit box.

    auto pointMEquiv = surface(pEquiv, (double *)&(pCenterMEquiv[0]), scaleIn2, 0);
    auto pointMCheck = surface(pCheck, (double *)&(pCenterMCheck[0]), scaleOut, 0);

    auto pointLCheck = surface(pCheck, (double *)&(pCenterLCheck[0]), scaleIn2, 0);
    auto pointLEquiv = surface(pEquiv, (double *)&(pCenterLEquiv[0]), scaleOut, 0);

    const int equivN = pointMEquiv.size() / 3;
    const int checkN = pointMCheck.size() / 3;

    EMat M2L(kdim[1] * equivN, kdim[1] * equivN); // M2L density
    EMat M2C(kdim[0] * checkN, kdim[1] * equivN); // M2C check surface

    EMat AL(kdim[0] * checkN, kdim[1] * equivN); // L den to L check
    EMat ALpinvU(AL.cols(), AL.rows());
    EMat ALpinvVT(AL.cols(), AL.rows());
#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        EVec3 Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1], pointLCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const EVec3 Lpoint(pointLEquiv[3 * l], pointLEquiv[3 * l + 1], pointLEquiv[3 * l + 2]);
            AL(k, l) = gKernel(Cpoint, Lpoint);
        }
    }
    pinv(AL, ALpinvU, ALpinvVT);

    // AL is check x equiv
    // AL-1 is equiv x check
    // I want to know if AL-1 * AL is the identity.
    //std::cout << "backwards error #1: " << (ALpinvU.transpose() * ALpinvVT.transpose()) * AL << std::endl;

#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const EVec3 Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1], pointMEquiv[3 * i + 2]);
	const EVec3 Npoint(0.5, 0.5, 0.5); // neutralizing

        EVec f(checkN);
        for (int k = 0; k < checkN; k++) {
            EVec3 Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1], pointLCheck[3 * k + 2]);
            f(k) = gKernelFF(Cpoint, Mpoint); // sum the images
	    if (NEUTRALIZE_FOR_M2C) {
	      f(k) -= gKernelFF(Cpoint, Npoint);
	    }
        }
        M2C.col(i) = f;
        M2L.col(i) = (ALpinvU.transpose() * (ALpinvVT.transpose() * f));
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "Precomputing time:" << duration / 1e6 << std::endl;

    saveEMat(M2L, "M2L_laplace_3D3D_p" + std::to_string(pEquiv));
    saveEMat(M2C, "M2C_laplace_3D3D_p" + std::to_string(pEquiv));

    ////////////////// TEST CODE

    // AM construction (inverted later)
    // --------------------------------

    // AM
    // (Me ->G Mc)

    EMat AM(kdim[0] * checkN, kdim[1] * equivN); // M den to M check
    EMat AMpinvU(AM.cols(), AM.rows());
    EMat AMpinvVT(AM.cols(), AM.rows());
#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        EVec3 Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const EVec3 Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1], pointMEquiv[3 * l + 2]);
            AM(k, l) = gKernel(Cpoint, Mpoint);
        }
    }
    // AM-1
    // (Mc ->G-1 Me)
    pinv(AM, AMpinvU, AMpinvVT);

    if (true) {
      std::cout << "TESTING A DIPOLE" << std::endl;
      // Test
      // NOTE: These sources DO add to zero!
      // It is a dipole about the center of the box.
      EVec3 center(0.6, 0.5, 0.5);
      std::vector<EVec3, Eigen::aligned_allocator<EVec3>> chargePoint(2);
      std::vector<double> chargeValue(2);
      chargePoint[0] = center + EVec3(0.1, 0, 0);
      chargeValue[0] = 1;
      chargePoint[1] = center + EVec3(-0.1, 0., 0.);
      chargeValue[1] = -1;

      // solve M
      EVec f(checkN);
      for (int k = 0; k < checkN; k++) {
        double temp = 0;
        EVec3 Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        for (size_t p = 0; p < chargePoint.size(); p++) {
	  temp = temp + gKernel(Cpoint, chargePoint[p]) * (chargeValue[p]);
        }
        f[k] = temp;
      }
      EVec Msource = (AMpinvU.transpose() * (AMpinvVT.transpose() * f));
      EVec M2Lsource = M2L * (Msource);

      //std::cout << "Msource: " << Msource.transpose() << std::endl;
      //std::cout << "M2Lsource: " << M2Lsource.transpose() << std::endl;

      // f goes from the source charge to the Mc.
      // AM * Msource = AM AM-1 f.
      // This is very small, as intended.
      //std::cout << "backwards error #2: " << f - AM * Msource << std::endl;

      // check total charge
      {
	double total_charge = 0.0;
        for (int i = 0; i < chargePoint.size(); i++) {
	  total_charge += chargeValue[i];
        }
	std::cout << "charge " << total_charge << std::endl;
      }

      // check dipole moment
      // This evaluates the dipole (as constructed) about its center.
      {
        EVec3 dipole = EVec3::Zero();
        for (int i = 0; i < chargePoint.size(); i++) {
	  dipole += chargeValue[i] * chargePoint[i];
        }
        std::cout << "charge dipole " << dipole.transpose() << std::endl;
      }

      {
	double total_charge = 0.0;
        for (int i = 0; i < equivN; i++) {
	  total_charge += Msource[i];
        }
	std::cout << "equivalent charge " << total_charge << std::endl;
      }

      // This gets the upward equivalent dipole from the equivalent surface.
      // This should match what is above and does not depend on periodicity.
      {
        EVec3 dipole = EVec3::Zero();
        for (int i = 0; i < equivN; i++) {
	  EVec3 Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1], pointMEquiv[3 * i + 2]);
	  dipole += Mpoint * Msource[i];
        }
        std::cout << "Mequiv dipole " << dipole.transpose() << std::endl;
      }

      // Loop over 5 random sample points.
      // Compute three quantities.
      // 1) The far-field source-to-target (source to sample point).
      // 2) The free L-equivalent to target.
      // 3) The far-field M-equivalent to target.
      for (int is = 0; is < 5; is++) {

        EVec3 samplePoint = EVec3::Random() * 0.2 + EVec3(0.5, 0.5, 0.5);

        EVec4 UFFL2T = EVec4::Zero();
        EVec4 UFFS2T = EVec4::Zero();
        EVec4 UFFM2T = EVec4::Zero();
	EVec4 UNF = EVec4::Zero();
	EVec4 UEwald = EVec4::Zero();

#pragma omp sections
        {
#pragma omp section
	  for (int p = 0; p < chargePoint.size(); p++) {
	    UFFS2T += ggradKernelFF(samplePoint, chargePoint[p]) * chargeValue[p];
	    UNF += ggradKernelNF(samplePoint, chargePoint[p]) * chargeValue[p];
	    UEwald += ggradKernelEwald(samplePoint, chargePoint[p]) * chargeValue[p];
	  }

#pragma omp section
	  for (int p = 0; p < equivN; p++) {
	    EVec3 Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1], pointLEquiv[3 * p + 2]);
	    UFFL2T += ggradKernel(samplePoint, Lpoint) * M2Lsource[p];
	  }

#pragma omp section
	  for (int p = 0; p < equivN; p++) {
	    EVec3 Mpoint(pointMEquiv[3 * p], pointMEquiv[3 * p + 1], pointMEquiv[3 * p + 2]);
	    UFFM2T += ggradKernelFF(samplePoint, Mpoint) * Msource[p];
	  }
	}
        std::cout << std::scientific << std::setprecision(10);
        std::cout << "-----------------------------------------------" << std::endl;
        std::cout << "samplePoint:" << samplePoint.transpose() << std::endl;
        std::cout << "UFF S2T: " << UFFS2T.transpose() << std::endl;
        std::cout << "UFF M2T: " << UFFM2T.transpose() << std::endl;
        std::cout << "UFF L2T: " << UFFL2T.transpose() << std::endl;
        std::cout << "Error M2T: " << (UFFM2T - UFFS2T).transpose() << std::endl;
        std::cout << "Error L2T: " << (UFFL2T - UFFS2T).transpose() << std::endl;

	std::cout << "UNF: " << UNF.transpose() << std::endl;
	std::cout << "UEwald: " << UEwald.transpose() << std::endl;

	// This is exactly what he's calculating above!
	std::cout << "Error vs. Ewald: " << (UNF + UFFL2T - UEwald).transpose() << std::endl;
      }
    }

    if (false) {
      // Test the far-field equivalence of M.
      // Loop from 1 to 5.
      for (int offset = 1; offset <= 5; ++offset) {
	EVec3 benPoint(0.5, 0.5, 0.5);
	double benCharge = 1;

	EVec3 samplePoint(0.5 + offset, 0.5, 0.5);

	// Determine the upward equivalent sources by matching at the check surface.
	EVec f(checkN);
	for (int k = 0; k < checkN; k++) {
	  EVec3 Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
	  f[k] = gKernel(Cpoint, benPoint) * benCharge;
	}
	EVec Msource = (AMpinvU.transpose() * (AMpinvVT.transpose() * f));

	double fromEquiv = 0;
	for (int p = 0; p < equivN; ++p) {
	  EVec3 Mpoint(pointMEquiv[3 * p], pointMEquiv[3 * p + 1], pointMEquiv[3 * p + 2]);
	  fromEquiv += (ggradKernel(samplePoint, Mpoint) * Msource[p])[0];
	}

	auto direct = (ggradKernel(samplePoint, benPoint) * benCharge)[0];

	std::cout << "offset: " << offset
		  << ", direct: " << direct
		  << ", equivalent: " << fromEquiv
		  << ", (equivalent-direct): " << fromEquiv - direct
		  << std::endl;
      }
    }

    if (true) {
      EVec3 benPoint(0.5, 0.5, 0.5);
      double benCharge = 1;

      // This is the madelung constant divided by 4 pi.
      // So their Ewald sum is correct.
      std::cout << "bjlbjl ewald center " << benCharge * gKernelEwald(benPoint, benPoint) << std::endl;

      // Determine the upward equivalent sources by matching at the check surface.
      EVec f(checkN);
      for (int k = 0; k < checkN; k++) {
        EVec3 Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
	f[k] = gKernel(Cpoint, benPoint) * benCharge;
      }
      EVec Msource = (AMpinvU.transpose() * (AMpinvVT.transpose() * f));
      PrintMomentStats(equivN, scaleIn2, pointMEquiv, Msource, "Msource");

      //TestOneOffSum(equivN, pointMEquiv, Msource);
      //TestZeroOffSum(equivN, pointMEquiv, Msource);

      // Get the L equivalent sources from the M equivalent sources.
      EVec M2Lsource = M2L * Msource;
      PrintMomentStats(equivN, scaleOut, pointLEquiv, M2Lsource, "M2Lsource");
      PlotField(pointLEquiv, pointMEquiv, Msource, M2Lsource);

      if (true) {
	std::cout << "Test on source" << std::endl;
	int numSample = 1;
	std::vector<EVec3, Eigen::aligned_allocator<EVec3>> samplePoints(numSample);
	samplePoints[0] = benPoint;
	TestPoints(numSample,
		   samplePoints,
		   benPoint,
		   benCharge,
		   equivN,
		   pointLEquiv,
		   pointMEquiv,
		   Msource,
		   M2Lsource);
      }

      if (false) {
	// This confirms that the discrepancy isn't from the on-source contribution.
	std::cout << "Test off source" << std::endl;
	int numSample = 1;
	std::vector<EVec3, Eigen::aligned_allocator<EVec3>> samplePoints(numSample);
	samplePoints[0] = benPoint + EVec3(0.01, 0.02, -0.01);
	TestPoints(numSample,
		   samplePoints,
		   benPoint,
		   benCharge,
		   equivN,
		   pointLEquiv,
		   pointMEquiv,
		   Msource,
		   M2Lsource);
      }

      if (false) {
	std::cout << "bjlbjl Raster over" << std::endl;
	int vertSample = 5;
	auto numSample = vertSample * vertSample * vertSample;
	std::vector<EVec3, Eigen::aligned_allocator<EVec3>> samplePoints(numSample);
	int nn = 0;
	for (int ii = 0; ii < vertSample; ++ii) {
	  double x = double(ii) / vertSample;
	  for (int jj = 0; jj < vertSample; ++jj) {
	    double y = double(jj) / vertSample;
	    for (int kk = 0; kk < vertSample; ++kk) {
	      double z = double(kk) / vertSample;
	      samplePoints[nn] = EVec3(x, y, z);
	      ++nn;
	    }
	  }
	}

	TestPoints(numSample,
		   samplePoints,
		   benPoint,
		   benCharge,
		   equivN,
		   pointLEquiv,
		   pointMEquiv,
		   Msource,
		   M2Lsource);
      }
    }

    std::cout << "DIRECTLAYER: " << DIRECTLAYER << std::endl;

    return 0;
}

} // namespace Laplace3D3D
