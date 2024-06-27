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
#ifndef ADDITIVERUNGEKUTTAINTEGRATOR_HPP
#define ADDITIVERUNGEKUTTAINTEGRATOR_HPP

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

  enum AdditiveRungeKuttaScheme
  {
    stage_3_order_2, /* Three stage second order explicit scheme (Giraldo et al., 2012) and TR-BDF2 */
  };



  // @sect3{Additive Runge--Kutta time integrators}

  // The next few lines implement an an Implicit-Explicit Additive Runge-Kutta
  // scheme for the momentum and an explicit Runge-Kutta scheme for the mass
  // and the tracer equations. Additive Runge-Kutta are useful for problems that
  // can be written as the sum of a stiff and non-stiff components. Then an
  // implicit and an explicit companion scheme are applied to the each component.
  // The stiff-part is to be designed into the PDE operator but we anticipate 
  // that for our `OceanOperator`, it will be the bottom friction. The main
  // advantage of  this kind of scheme is the enchanced stability.
  // Consider also that the mass equation and the tracers must be solved with
  // the same time-integrator for consistency reason (the so called "tracer
  // consistency with the mass-equation"). 
  // 
  // A first remark on the implementation. In a multi-stage Runge-Kutta scheme
  // we are obliged to time timestep both hydrodynamics and tracers variables
  // with a unique call to `perform_time_step`. To distinguish the two cases,
  // thus avoiding compiler warnings or fake loops when tracers are absent we
  // use preprocessor. Although this may worsen the code readibility, we believe it
  // is better then creating a derived class that overloads `perform_time_step`.
  //
  // The Additive Runge-Kutta method has two tableaux with the coefficients
  // $b_i$ and $a_i$, one for the explicit scheme and one for the implicit scheme.
  // The implicit coefficients are typically distinguished with a tilde.
  // As usual in Runge--Kutta method, we can deduce time steps, 
  // $c_i = \sum_{j=1}^{i-2} b_i + a_{i-1}$ from those coefficients. For the
  // implicit part we extract the diagonal matrix and we store in a separate
  // vector called `dtilde`.
  //
  // There exist second, third and fourth order ARK schemes.
  // We define them in single class, distinguished by the
  // enum described above. To each scheme, we then fill the vectors for the
  // $b_i$, $a_{ij}$ and $\tilde{b}_i$ and $\tilde{a}_{ij}$ to the given variables
  //  in the class.
  class AdditiveRungeKuttaIntegrator //: public RungeKuttaIntegrator
  {
  public:
    AdditiveRungeKuttaIntegrator(const AdditiveRungeKuttaScheme scheme);
    ~AdditiveRungeKuttaIntegrator(){};

    unsigned int n_stages() const
    {
      return bi.size();
    }

    template <typename VectorType, typename Operator>                                  
    void perform_time_step(Operator                &pde_operator,
                           const double             current_time,
                           const double             time_step,
                           VectorType              &solution_height,
                           VectorType              &solution_discharge,
                           VectorType              &solution_tracer,
                           VectorType              &vec_ri_height,
                           VectorType              &vec_ri_discharge,
                           VectorType              &vec_ri_tracer,
                           std::vector<VectorType> &vec_ki_height,
                           std::vector<VectorType> &vec_ki_discharge,
                           std::vector<VectorType> &vec_ki_tracer) const;

  private:
    std::vector<std::vector<double>> ai;
    std::vector<double> bi;
    std::vector<double> ci;

    std::vector<std::vector<double>> atildei;
    std::vector<double> btildei;
    std::vector<double> ctildei;
    std::vector<double> dtildei;
  };



  AdditiveRungeKuttaIntegrator::AdditiveRungeKuttaIntegrator(
    const AdditiveRungeKuttaScheme scheme)
  {
    TimeSteppingOceano::runge_kutta_method_oceano erk;
    TimeSteppingOceano::runge_kutta_method_oceano irk;
    switch (scheme)
      {
      
        // First comes the three-stage scheme of order two by Giraldo et al., (2012).
        // The implicit part is the trapezoidal BDF2 scheme. The explicit part has
        // enhanced stability and monotonicity region ...
        case stage_3_order_2: 
          {
            erk = TimeSteppingOceano::THREE_STAGE_SECOND_ORDER;
            irk = TimeSteppingOceano::TRAPEZOIDAL_BDF2;
            break;
          }

        default:
          AssertThrow(false, ExcNotImplemented());
      }
    TimeSteppingOceano::ExplicitRungeKutta erk_integrator(erk);
    erk_integrator.get_coefficients(ai, bi, ci);
    TimeSteppingOceano::ImplicitRungeKutta irk_integrator(irk);
    irk_integrator.get_coefficients(atildei, btildei, ctildei, dtildei);
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
  void AdditiveRungeKuttaIntegrator::perform_time_step(
    Operator                &pde_operator,
    const double             current_time,
    const double             time_step,
    VectorType              &solution_height,
    VectorType              &solution_discharge,
    VectorType              &solution_tracer,
    VectorType              &vec_ri_height,
    VectorType              &vec_ri_discharge,
    VectorType              &vec_ri_tracer,
    std::vector<VectorType> &vec_ki_height,
    std::vector<VectorType> &vec_ki_discharge,
    std::vector<VectorType> &vec_ki_tracer) const
  {
    AssertDimension(ci.size(), bi.size());

#ifndef OCEANO_WITH_TRACERS
    (void) solution_tracer;
    (void) vec_ri_tracer;
    (void) vec_ki_tracer;
#endif

    std::vector<double> b_i = bi;
    for (unsigned int i = 0; i < bi.size(); ++i) b_i[i] *= time_step;

    std::vector<std::vector<double>> a_i = ai;
    for (unsigned int stage = 0; stage < ai.size(); ++stage)
      for (unsigned int i = 0; i < ai[stage].size(); ++i) a_i[stage][i] *= time_step;

    std::vector<std::vector<double>> a_tilde_i = atildei;
    for (unsigned int stage = 0; stage < atildei.size(); ++stage)
      for (unsigned int i = 0; i < atildei[stage].size(); ++i) a_tilde_i[stage][i] *= time_step;

    pde_operator.factor_matrix = dtildei[0] * time_step;

    pde_operator.perform_stage_hydro(0,
                                     current_time,
                                     (0 == ci.size() - 1 ?
                                       &b_i[0] :
                                       &a_i[0][0]),
                                     (0 == ci.size() - 1 ?
                                       &b_i[0] :
                                       &a_tilde_i[0][0]),
                                     {solution_height, solution_discharge},
                                     vec_ki_height,
                                     vec_ki_discharge,
                                     solution_height,
                                     solution_discharge,
                                     vec_ri_height,
                                     vec_ri_discharge);
#ifdef OCEANO_WITH_TRACERS
    pde_operator.perform_stage_tracers(0,
                                       (0 == ci.size() - 1 ?
                                         &b_i[0] :
                                         &a_i[0][0]),
                                       {solution_height, solution_discharge, solution_tracer},
                                       vec_ki_tracer,
                                       solution_tracer,
                                       vec_ri_tracer);
#endif

    for (unsigned int stage = 1; stage < ci.size(); ++stage)
      {
        const double c_i = ci[stage];
        pde_operator.factor_matrix = dtildei[stage] * time_step;

        pde_operator.perform_stage_hydro(stage,
                                         current_time + c_i * time_step,
                                         (stage == ci.size() - 1 ?
                                           &b_i[0] :
                                           &a_i[stage][0]),
                                         (stage == ci.size() - 1 ?
                                           &b_i[0] :
                                           &a_tilde_i[stage][0]),
                                         {vec_ri_height, vec_ri_discharge},
                                         vec_ki_height,
                                         vec_ki_discharge,
                                         solution_height,
                                         solution_discharge,
                                         vec_ri_height,
                                         vec_ri_discharge);
#ifdef OCEANO_WITH_TRACERS
        pde_operator.perform_stage_tracers(stage,
                                           (stage == ci.size() - 1 ?
                                             &b_i[0] :
                                             &a_i[stage][0]),
                                          {vec_ri_height, vec_ri_discharge, vec_ri_tracer},
                                          vec_ki_tracer,
                                          solution_tracer,
                                          vec_ri_tracer);
#endif
      }
  }

} // namespace TimeIntegrator

#endif //ADDITIVERUNGEKUTTAINTEGRATOR_HPP
