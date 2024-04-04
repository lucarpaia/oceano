/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2020 - 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Martin Kronbichler, 2020
 *         Luca Arpaia,        2023
 */
#ifndef STRONGSTABILITYRUNGEKUTTAINTEGRATOR_HPP
#define STRONGSTABILITYRUNGEKUTTAINTEGRATOR_HPP

#include <deal.II/base/timer.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/fe/fe_system.h>

#include <time_integrator/time_stepping.h>
/**
 * Namespace containing the time stepping methods.
 */

namespace TimeIntegrator
{

  using namespace dealii;

  using Number = double;

  enum StrongStabilityRungeKuttaScheme
  {
    stage_1_order_1, /* Forward Euler */
    stage_2_order_2, /* RK2 coded as SSP22 Gottilieb and Shu */
    stage_3_order_3, /* SSP33 Gottilieb and Shu */
  };



  // @sect3{Strong Stability Preserving explicit Runge--Kutta time integrators}

  // The next few lines implement the Strong Stability Preserving Runge--Kutta
  // methods. These methods have specific tableaux with coefficients
  // $\beta_i$ and $\alpha_i$ as shown in the introduction. As usual in Runge--Kutta
  // method, we can deduce time steps, $c_i = \sum_{j=1}^{i-2} b_i + a_{i-1}$
  // from those coefficients. The main advantage of this kind of scheme is the
  // fact that ... lrp
  //
  // In this tutorial programs, we concentrate on a few variants of
  // ... TODO!lrp
  //
  // We define a single class for the four integrators, distinguished by the
  // enum described above. To each scheme, we then fill the vectors for the
  // $b_i$ and $a_i$ to the given variables in the class.
  class StrongStabilityRungeKuttaIntegrator //: public RungeKuttaIntegrator
  {
  public:
    StrongStabilityRungeKuttaIntegrator(const StrongStabilityRungeKuttaScheme scheme);
    ~StrongStabilityRungeKuttaIntegrator(){};

    unsigned int n_stages() const
    {
      return bi.size();
    }

    template <typename VectorType, typename Operator>                                  
    void perform_time_step(const Operator          &pde_operator,
                           const double             current_time,
                           const double             time_step,
                           VectorType              &solution,
                           std::vector<VectorType> &vec_ri,
                           VectorType              &vec_ki) const;

  private:
    std::vector<std::vector<double>> ai;
    std::vector<double> bi;
    std::vector<double> ci;

  };



  StrongStabilityRungeKuttaIntegrator::StrongStabilityRungeKuttaIntegrator(
    const StrongStabilityRungeKuttaScheme scheme)
  {
    TimeSteppingOceano::runge_kutta_method_oceano ssprk;
    // First comes the three-stage scheme of order three by Kennedy et al.
    // (2000). While its stability region is significantly smaller than for
    // the other schemes, it only involves three stages, so it is very
    // competitive in terms of the work per stage.
    switch (scheme)
      {
        case stage_1_order_1:
          {
            ssprk = TimeSteppingOceano::FORWARD_EULER;
            break;
          }

          // The next scheme is a five-stage scheme of order four, again
          // defined in the paper by Kennedy et al. (2000).
        case stage_2_order_2: 
          {
            ssprk = TimeSteppingOceano::SSP_SECOND_ORDER;
            break;
          }

          // The next scheme is a five-stage scheme of order four, again
          // defined in the paper by Kennedy et al. (2000).
        case stage_3_order_3: 
          {
            ssprk = TimeSteppingOceano::SSP_THIRD_ORDER;
            break;
          }

        default:
          AssertThrow(false, ExcNotImplemented());
      }
    TimeSteppingOceano::StrongStabilityRungeKutta rk_integrator(ssprk);
    rk_integrator.get_coefficients(ai, bi, ci);
  }
  
  // The main function of the time integrator is to go through the stages,
  // evaluate the operator, prepare the $\mathbf{r}_i$ vector for the next
  // evaluation, and update the solution vector $\mathbf{w}$. We hand off
  // the work to the `pde_operator` involved in order to be able to merge
  // the vector operations of the Runge--Kutta setup with the evaluation of
  // the differential operator for better performance, so all we do here is
  // to delegate the vectors and coefficients.
  //
  // We separately call the operator for the first stage because we need
  // slightly modified arguments there: We evaluate the solution from
  // the old solution $\mathbf{w}^n$ rather than a $\mathbf r_i$ vector, so
  // the first argument is `solution`. We here let the stage vector
  // $\mathbf{r}_i$ also hold the temporary result of the evaluation, as it
  // is not used otherwise. For all subsequent stages, we use the vector
  // `vec_ki` as the second vector argument to store the result of the
  // operator evaluation. Finally, when we are at the last stage, we must
  // skip the computation of the vector $\mathbf{r}_{s+1}$ as there is no
  // coefficient $a_s$ available (nor will it be used).
  template <typename VectorType, typename Operator>
  void StrongStabilityRungeKuttaIntegrator::perform_time_step(
    const Operator          &pde_operator,
    const double             current_time,
    const double             time_step,
    VectorType              &solution,
    std::vector<VectorType> &vec_ri,
    VectorType              &vec_ki) const
  {
    AssertDimension(ci.size(), bi.size());

    pde_operator.perform_stage(0,
                               current_time,
                               bi[0] * time_step,
                               &ai[0][0],
                               solution,
                               vec_ki,
                               solution,
                               vec_ri);

    for (unsigned int stage = 1; stage < ci.size(); ++stage)
      {
        const double c_i = ci[stage];
        pde_operator.perform_stage(stage,
                                   current_time + c_i * time_step,
                                   bi[stage] * time_step,
                                   &ai[stage][0],
                                   vec_ri[stage],
                                   vec_ki,
                                   solution,
                                   vec_ri);
      }
  }

} // namespace TimeIntegrator

#endif //STRONGSTABILITYRUNGEKUTTAINTEGRATOR_HPP
