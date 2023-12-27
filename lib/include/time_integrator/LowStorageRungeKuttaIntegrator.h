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
#ifndef LOWSTORAGERUNGEKUTTAINTEGRATOR_HPP
#define LOWSTORAGERUNGEKUTTAINTEGRATOR_HPP

#include <deal.II/base/timer.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/fe/fe_system.h>

/**
 * Namespace containing the time stepping methods.
 */

namespace TimeIntegrator 
{

  using namespace dealii;

  using Number = double;

  enum LowStorageRungeKuttaScheme
  {
    stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_7_order_4, /* Tselios, Simos, 2007 */
    stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
  };



  // @sect3{Low-storage explicit Runge--Kutta time integrators}

  // The next few lines implement a few low-storage variants of Runge--Kutta
  // methods. These methods have specific Butcher tableaux with coefficients
  // $b_i$ and $a_i$ as shown in the introduction. As usual in Runge--Kutta
  // method, we can deduce time steps, $c_i = \sum_{j=1}^{i-2} b_i + a_{i-1}$
  // from those coefficients. The main advantage of this kind of scheme is the
  // fact that only two vectors are needed per stage, namely the accumulated
  // part of the solution $\mathbf{w}$ (that will hold the solution
  // $\mathbf{w}^{n+1}$ at the new time $t^{n+1}$ after the last stage), the
  // update vector $\mathbf{r}_i$ that gets evaluated during the stages, plus
  // one vector $\mathbf{k}_i$ to hold the evaluation of the operator. Such a
  // Runge--Kutta setup reduces the memory storage and memory access. As the
  // memory bandwidth is often the performance-limiting factor on modern
  // hardware when the evaluation of the differential operator is
  // well-optimized, performance can be improved over standard time
  // integrators. This is true also when taking into account that a
  // conventional Runge--Kutta scheme might allow for slightly larger time
  // steps as more free parameters allow for better stability properties.
  //
  // In this tutorial programs, we concentrate on a few variants of
  // low-storage schemes defined in the article by Kennedy, Carpenter, and
  // Lewis (2000), as well as one variant described by Tselios and Simos
  // (2007). There is a large series of other schemes available, which could
  // be addressed by additional sets of coefficients or slightly different
  // update formulas.
  //
  // We define a single class for the four integrators, distinguished by the
  // enum described above. To each scheme, we then fill the vectors for the
  // $b_i$ and $a_i$ to the given variables in the class.
  class LowStorageRungeKuttaIntegrator //: public RungeKuttaIntegrator
  {
  public:
    LowStorageRungeKuttaIntegrator(const LowStorageRungeKuttaScheme scheme);
    ~LowStorageRungeKuttaIntegrator(){};

    unsigned int n_stages() const
    {
      return bi.size();
    }

    template <typename Operator>
    void perform_stage_debuggggggggg(const Operator                                   &pde_operator,
                                     const Number                                      current_time,
                                     const Number                                      factor_solution,
                                     const Number                                      factor_ai,
                                     const LinearAlgebra::distributed::Vector<Number> &current_ri,
                                     LinearAlgebra::distributed::Vector<Number> &      vec_ki,
                                     LinearAlgebra::distributed::Vector<Number> &      solution,
                                     LinearAlgebra::distributed::Vector<Number> &      next_ri) const;

    template <typename VectorType, typename Operator>                                   
    void perform_time_step(const Operator &pde_operator,
                           const double    current_time,
                           const double    time_step,
                           VectorType &    solution,
                           VectorType &    vec_ri,
                           VectorType &    vec_ki) const;

  private:
    std::vector<double> bi;
    std::vector<double> ai;
    std::vector<double> ci;

  };
  
  
  
  LowStorageRungeKuttaIntegrator::LowStorageRungeKuttaIntegrator(const LowStorageRungeKuttaScheme scheme)
  {
    TimeStepping::runge_kutta_method lsrk;
    // First comes the three-stage scheme of order three by Kennedy et al.
    // (2000). While its stability region is significantly smaller than for
    // the other schemes, it only involves three stages, so it is very
    // competitive in terms of the work per stage.
    switch (scheme)
      {
        case stage_3_order_3:
          {
            lsrk = TimeStepping::LOW_STORAGE_RK_STAGE3_ORDER3;
            break;
          }

          // The next scheme is a five-stage scheme of order four, again
          // defined in the paper by Kennedy et al. (2000).
        case stage_5_order_4: 
          {
            lsrk = TimeStepping::LOW_STORAGE_RK_STAGE5_ORDER4;
            break;
          }

          // The following scheme of seven stages and order four has been
          // explicitly derived for acoustics problems. It is a balance of
          // accuracy for imaginary eigenvalues among fourth order schemes,
          // combined with a large stability region. Since DG schemes are
          // dissipative among the highest frequencies, this does not
          // necessarily translate to the highest possible time step per
          // stage. In the context of the present tutorial program, the
          // numerical flux plays a crucial role in the dissipation and thus
          // also the maximal stable time step size. For the modified
          // Lax--Friedrichs flux, this scheme is similar to the
          // `stage_5_order_4` scheme in terms of step size per stage if only
          // stability is considered, but somewhat less efficient for the HLL
          // flux.
        case stage_7_order_4:
          {
            lsrk = TimeStepping::LOW_STORAGE_RK_STAGE7_ORDER4;
            break;
          }

          // The last scheme included here is the nine-stage scheme of order
          // five from Kennedy et al. (2000). It is the most accurate among
          // the schemes used here, but the higher order of accuracy
          // sacrifices some stability, so the step length normalized per
          // stage is less than for the fourth order schemes.
        case stage_9_order_5:
          {
            lsrk = TimeStepping::LOW_STORAGE_RK_STAGE9_ORDER5;
            break;
          }

        default:
          AssertThrow(false, ExcNotImplemented());
      }
    TimeStepping::LowStorageRungeKutta<LinearAlgebra::distributed::Vector<Number>> rk_integrator(lsrk);
    rk_integrator.get_coefficients(ai, bi, ci);
  }
  
  // Let us move to the function that does an entire stage of a Runge--Kutta
  // update. It calls EulerOperator::apply() followed by some updates
  // to the vectors, namely `next_ri = solution + factor_ai * k_i` and
  // `solution += factor_solution * k_i`. Rather than performing these
  // steps through the vector interfaces, we here present an alternative
  // strategy that is faster on cache-based architectures. As the memory
  // consumed by the vectors is often much larger than what fits into caches,
  // the data has to effectively come from the slow RAM memory. The situation
  // can be improved by loop fusion, i.e., performing both the updates to
  // `next_ki` and `solution` within a single sweep. In that case, we would
  // read the two vectors `rhs` and `solution` and write into `next_ki` and
  // `solution`, compared to at least 4 reads and two writes in the baseline
  // case. Here, we go one step further and perform the loop immediately when
  // the mass matrix inversion has finished on a part of the
  // vector. MatrixFree::cell_loop() provides a mechanism to attach an
  // `std::function` both before the loop over cells first touches a vector
  // entry (which we do not use here, but is e.g. used for zeroing the vector)
  // and a second `std::function` to be called after the loop last touches
  // an entry. The callback is in form of a range over the given vector (in
  // terms of the local index numbering in the MPI universe) that can be
  // addressed by `local_element()` functions.
  //
  // For this second callback, we create a lambda that works on a range and
  // write the respective update on this range. Ideally, we would add the
  // `DEAL_II_OPENMP_SIMD_PRAGMA` before the local loop to suggest to the
  // compiler to SIMD parallelize this loop (which means in practice that we
  // ensure that there is no overlap, also called aliasing, between the index
  // ranges of the pointers we use inside the loops). It turns out that at the
  // time of this writing, GCC 7.2 fails to compile an OpenMP pragma inside a
  // lambda function, so we comment this pragma out below. If your compiler is
  // newer, you should be able to uncomment these lines again.
  //
  // Note that we select a different code path for the last
  // Runge--Kutta stage when we do not need to update the `next_ri`
  // vector. This strategy gives a considerable speedup. Whereas the inverse
  // mass matrix and vector updates take more than 60% of the computational
  // time with default vector updates on a 40-core machine, the percentage is
  // around 35% with the more optimized variant. In other words, this is a
  // speedup of around a third.
  template <typename Operator>
  void LowStorageRungeKuttaIntegrator::perform_stage_debuggggggggg(
                                   const Operator                                   &pde_operator,
                                   const Number                                      current_time,
                                   const Number                                      factor_solution,
                                   const Number                                      factor_ai,
                                   const LinearAlgebra::distributed::Vector<Number> &current_ri,
                                   LinearAlgebra::distributed::Vector<Number> &      vec_ki,
                                   LinearAlgebra::distributed::Vector<Number> &      solution,
                                   LinearAlgebra::distributed::Vector<Number> &      next_ri) const
  {
    {
      TimerOutput::Scope t(pde_operator.timer, "rk_stage - integrals L_h");
      
      for (auto &i : pde_operator.inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : pde_operator.subsonic_outflow_boundaries)
        i.second->set_time(current_time);
      
/*        pde_operator.data.loop(&EulerOperator::local_apply_cell,
                               &EulerOperator::local_apply_face,
                               &EulerOperator::local_apply_boundary_face,
                               this,
                               vec_ki,
                               current_ri,
                               true,
                               MatrixFree<dim, Number>::DataAccessOnFaces::values,
                               MatrixFree<dim, Number>::DataAccessOnFaces::values);*/

    }


    {   
      TimerOutput::Scope t(pde_operator.timer, "rk_stage - inv mass + vec upd");
      pde_operator.data.cell_loop(
        pde_operator.local_apply_inverse_mass_matrix,
        pde_operator,
        next_ri,
        vec_ki,   
        std::function<void(const unsigned int, const unsigned int)>(),
        [&](const unsigned int start_range, const unsigned int end_range) {
          const Number ai = factor_ai;
          const Number bi = factor_solution;
          if (ai == Number())
            {     
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  const Number k_i          = next_ri.local_element(i);
                  const Number sol_i        = solution.local_element(i);
                  solution.local_element(i) = sol_i + bi * k_i;
                }
            }
          else
            {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  const Number k_i          = next_ri.local_element(i);
                  const Number sol_i        = solution.local_element(i);
                  solution.local_element(i) = sol_i + bi * k_i;
                  next_ri.local_element(i)  = sol_i + ai * k_i;
                }
            }
        });
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
  void LowStorageRungeKuttaIntegrator::perform_time_step(const Operator &pde_operator,
                                                         const double    current_time,
                         const double    time_step,
                         VectorType &    solution,
                         VectorType &    vec_ri,
                         VectorType &    vec_ki) const
  {
    AssertDimension(ai.size() + 1, bi.size());

    pde_operator.perform_stage(current_time,
                               bi[0] * time_step,
                               ai[0] * time_step,
                               solution,
                               vec_ri,
                               solution,
                               vec_ri);
/*       perform_stage_debuggggggggg(pde_operator,
                                 current_time,
                                 bi[0] * time_step,
                                 ai[0] * time_step,
                                 solution,
                                 vec_ri,
                                 solution,
                                 vec_ri);*/

    for (unsigned int stage = 1; stage < bi.size(); ++stage)
      {
        const double c_i = ci[stage];
        pde_operator.perform_stage(current_time + c_i * time_step,
                                   bi[stage] * time_step,
                                   (stage == bi.size() - 1 ?
                                      0 :
                                      ai[stage] * time_step),
                                   vec_ri,
                                   vec_ki,
                                   solution,
                                   vec_ri);
      }
  }  
  
} // namespace TimeIntegrator

#endif //LOWSTORAGERUNGEKUTTAINTEGRATOR_HPP
