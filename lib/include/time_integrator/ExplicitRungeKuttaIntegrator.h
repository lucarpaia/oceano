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
#ifndef EXPLICITRUNGEKUTTAINTEGRATOR_HPP
#define EXPLICITRUNGEKUTTAINTEGRATOR_HPP

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

  enum ExplicitRungeKuttaScheme
  {
    stage_1_order_1, /* Forward Euler */
    stage_2_order_2, /* RK2 coded as SSP22 Gottlieb and Shu */
    stage_3_order_3, /* SSP33 Gottlieb and Shu */
    stage_4_order_4, /* Classic Fourth order Runge-Kutta */
    stage_3_order_2, /* Three stage second order of Giraldo */
  };



  // @sect3{Strong Stability Preserving explicit Runge--Kutta time integrators}

  // The next few lines implement the Strong Stability Preserving Runge--Kutta
  // methods for the hydrodynamic and tracer equations. The hydrodynamics and
  // the tracers must be solved with the same time-integrator for consistency
  // reason (the so called "tracer consistency with the mass-equation"). We
  // are obliged to time timestep both hydrodynamics and tracers variables
  // with a unique call to `perform_time_step`. To distinguish the two cases,
  // thus avoiding compiler warnings or fake loops when tracers are absent we
  // use preprocessor. Although this may worsen the code readibility, we believe it is
  // better then creating a derived class that overloads `perform_time_step`.
  // The Strong Stability Preserving Runge-Kutta methods have specific tableaux with coefficients
  // $\beta_i$ and $\alpha_i$ as shown in the introduction. As usual in Runge--Kutta
  // method, we can deduce time steps, $c_i = \sum_{j=1}^{i-2} b_i + a_{i-1}$
  // from those coefficients. The main advantage of this kind of scheme is the
  // non-linear stability.
  //
  // We define a single class for the four integrators, distinguished by the
  // enum described above. To each scheme, we then fill the vectors for the
  // $b_i$ and $a_i$ to the given variables in the class.
  class ExplicitRungeKuttaIntegrator //: public RungeKuttaIntegrator
  {
  public:
    ExplicitRungeKuttaIntegrator(const ExplicitRungeKuttaScheme scheme);
    ~ExplicitRungeKuttaIntegrator(){};

    unsigned int n_stages() const
    {
      return bi.size();
    }

    template <typename VectorType>
    void reinit(const VectorType              &solution_height,
                const VectorType              &solution_discharge,
                const VectorType              &solution_tracer,
                VectorType                    &vec_ri_height,
                VectorType                    &vec_ri_discharge,
                VectorType                    &vec_ri_tracer,
                std::vector<VectorType>       &vec_ki_height,
                std::vector<VectorType>       &vec_ki_discharge,
                std::vector<VectorType>       &vec_ki_tracer) const;

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
                           VectorType              &postprocess_velocity,
                           std::vector<VectorType> &vec_ki_height,
                           std::vector<VectorType> &vec_ki_discharge,
                           std::vector<VectorType> &vec_ki_tracer) const;

  private:
    std::vector<std::vector<double>> ai;
    std::vector<double> bi;
    std::vector<double> ci;

  };



  ExplicitRungeKuttaIntegrator::ExplicitRungeKuttaIntegrator(
    const ExplicitRungeKuttaScheme scheme)
  {
    TimeSteppingOceano::runge_kutta_method_oceano erk;
    // First comes the three-stage scheme of order three by Kennedy et al.
    // (2000). While its stability region is significantly smaller than for
    // the other schemes, it only involves three stages, so it is very
    // competitive in terms of the work per stage.
    switch (scheme)
      {
        case stage_1_order_1:
          {
            erk = TimeSteppingOceano::FORWARD_EULER;
            break;
          }

          // The next scheme is the Strong-Stability-Preserving
          // of order two.
        case stage_2_order_2: 
          {
            erk = TimeSteppingOceano::SSP_SECOND_ORDER;
            break;
          }

          // The next scheme is a five-stage scheme of order four, again
          // defined in the paper by Kennedy et al. (2000).
        case stage_3_order_3: 
          {
            erk = TimeSteppingOceano::SSP_THIRD_ORDER;
            break;
          }

        case stage_4_order_4: 
          {
            erk = TimeSteppingOceano::RK_CLASSIC_FOURTH_ORDER;
            break;
          }

        case stage_3_order_2: 
          {
            erk = TimeSteppingOceano::THREE_STAGE_SECOND_ORDER;
            break;
          }

        default:
          AssertThrow(false, ExcNotImplemented());
      }
    TimeSteppingOceano::ExplicitRungeKutta rk_integrator(erk);
    rk_integrator.get_coefficients(ai, bi, ci);
  }

  // This is a reinit function for the auxiliary vectors that are necessary
  // for the time integrator:
  template <typename VectorType>
  void ExplicitRungeKuttaIntegrator::reinit(
    const VectorType              &solution_height,
    const VectorType              &solution_discharge,
    const VectorType              &solution_tracer,
    VectorType                    &vec_ri_height,
    VectorType                    &vec_ri_discharge,
    VectorType                    &vec_ri_tracer,
    std::vector<VectorType>       &vec_ki_height,
    std::vector<VectorType>       &vec_ki_discharge,
    std::vector<VectorType>       &vec_ki_tracer) const
  {
    vec_ri_height.reinit(solution_height);
    vec_ri_discharge.reinit(solution_discharge);
    vec_ri_tracer.reinit(solution_tracer);

    for (unsigned int stage = 0; stage < n_stages()+1; ++stage)
      {
        vec_ki_height[stage].reinit(solution_height);
        vec_ki_discharge[stage].reinit(solution_discharge);
        vec_ki_tracer[stage].reinit(solution_tracer);
      }
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
  void ExplicitRungeKuttaIntegrator::perform_time_step(
    Operator                &pde_operator,
    const double             current_time,
    const double             time_step,
    VectorType              &solution_height,
    VectorType              &solution_discharge,
    VectorType              &solution_tracer,
    VectorType              &postprocess_velocity,
    VectorType              &vec_ri_height,
    VectorType              &vec_ri_discharge,
    VectorType              &vec_ri_tracer,
    std::vector<VectorType> &vec_ki_height,
    std::vector<VectorType> &vec_ki_discharge,
    std::vector<VectorType> &vec_ki_tracer) const
  {
    AssertDimension(ci.size(), bi.size());

#ifndef OCEANO_WITH_TRACERS
    (void) vec_ki_tracer;
#endif

    std::vector<double> b_i = bi;
    for (unsigned int i = 0; i < bi.size(); ++i) b_i[i] *= time_step;
    std::vector<std::vector<double>> a_i = ai;
    for (unsigned int stage = 0; stage < ai.size(); ++stage)
      for (unsigned int i = 0; i < ai[stage].size(); ++i) a_i[stage][i] *= time_step;

    std::vector<VectorType> vec_ri = {solution_height,
                                      solution_discharge,
                                      solution_tracer,
                                      postprocess_velocity};

    pde_operator.perform_stage_hydro(0,
                                     current_time,
                                     (0 == ci.size() - 1 ?
                                       &b_i[0] :
                                       &a_i[0][0]),
                                     vec_ri,
                                     vec_ki_height,
                                     vec_ki_discharge,
                                     solution_height,
                                     solution_discharge,
                                     vec_ri_height,
                                     vec_ri_discharge);
#ifdef OCEANO_WITH_TRACERS
    const VectorType vec_rn_height = solution_height;
    pde_operator.perform_stage_tracers(0,
                                       (0 == ci.size() - 1 ?
                                         &b_i[0] :
                                         &a_i[0][0]),
                                       vec_ri,
                                       solution_height,
                                       vec_ri_height,
                                       vec_ki_tracer,
                                       solution_tracer,
                                       vec_ri_tracer);
#endif
#ifdef OCEANO_WITH_MASSCONSERVATIONCHECK
    pde_operator.check_mass(b_i[0],
                            vec_ri,
                            solution_height);
#ifdef OCEANO_WITH_TRACERS
    pde_operator.check_tracer_mass(b_i[0],
                                   vec_ri,
                                   solution_height,
                                   solution_tracer);
#endif
#endif

    for (unsigned int stage = 1; stage < ci.size(); ++stage)
      {
        const double c_i = ci[stage];

        vec_ri = {vec_ri_height,
                  vec_ri_discharge,
                  vec_ri_tracer,
                  postprocess_velocity};

        pde_operator.perform_stage_hydro(stage,
                                         current_time + c_i * time_step,
                                         (stage == ci.size() - 1 ?
                                           &b_i[0] :
                                           &a_i[stage][0]),
                                         vec_ri,
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
                                           vec_ri,
                                           vec_rn_height,
                                           (stage == ci.size() - 1 ?
                                             solution_height :
                                             vec_ri_height),
                                           vec_ki_tracer,
                                           solution_tracer,
                                           vec_ri_tracer);
#endif
#ifdef OCEANO_WITH_MASSCONSERVATIONCHECK
        pde_operator.check_mass(b_i[stage],
                                vec_ri,
                                solution_height);
#ifdef OCEANO_WITH_TRACERS
        pde_operator.check_tracer_mass(b_i[stage],
                                       vec_ri,
                                       solution_height,
                                       solution_tracer);
#endif
#endif
      }
  }

} // namespace TimeIntegrator

#endif //EXPLICITRUNGEKUTTAINTEGRATOR_HPP
