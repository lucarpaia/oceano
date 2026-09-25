/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 - 2026 by CNR-ISMAR
 *
 * This code, as the deal.II library is free software; you can use it,
 * redistribute it, and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation;
 * either version 2.1 of the License, or (at your option) any later
 * version. The full text of the license can be found in the file
 * LICENSE.md at the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Luca Arpaia, 2026
 *         Giuseppe Orlando, 2026
 */
#ifndef OCEANDGSEMIIMPLICIT_H
#define OCEANDGSEMIIMPLICIT_H
 
// The following files include the oceano libraries
#include <space_discretization/OceanDG.h>

/**
 * Namespace containing the spatial Operator
 */

namespace SpaceDiscretization
{

  using namespace dealii;

  using Number = double;



  // @sect3{The OceanoOperator with semi-implicit integrator}

  // This class implements the assembly of the evaluators for the ocean
  // problem using a semi-implicit scheme. The partitioning of the right-hand
  // side of the shallow water equations into stiff and non-stiff components
  // follows the requirement of using a large time step that is not restricted
  // by gravity wave speed or bottom friction coefficient. This is obtained by
  // treating semi-implicitly the mass flux term in the continuity equation,
  // the pressure gradient term and the bottom friction term in the momentum
  // equation.
  template <int dim, int n_tra, int degree, int n_points_1d>
  class OceanoOperatorSemiImplicit : public OceanoOperator<dim, n_tra, degree, n_points_1d>
  {
  public:
    OceanoOperatorSemiImplicit(
      IO::ParameterHandler      &param,
      ICBC::BcBase<dim, 1+dim+n_tra> *bc,
      TimerOutput               &timer_output,
      const unsigned int         max_iteration_height);
    ~OceanoOperatorSemiImplicit() = default;

    void
    perform_stage_hydro(
      const unsigned int                                             cur_stage,
      const Number                                                   cur_time,
      const Number                                                  *factor_residual,
      const Number                                                  *factor_tilde_residual,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &current_ri,
      std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_height,
      std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_discharge,
      LinearAlgebra::distributed::Vector<Number>                    &solution_height,
      LinearAlgebra::distributed::Vector<Number>                    &solution_discharge,
      LinearAlgebra::distributed::Vector<Number>                    &next_ri_height,
      LinearAlgebra::distributed::Vector<Number>                    &next_ri_discharge) const;

    using Base = OceanoOperator<dim, n_tra, degree, n_points_1d>;
    using Base::bc;

  protected:
    using Base::data;
    using Base::timer;
  };



  // The constructor simply initialize the base classes of the Ocean Operator.
  template <int dim, int n_tra, int degree, int n_points_1d>
  OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::OceanoOperatorSemiImplicit(
    IO::ParameterHandler             &param,
    ICBC::BcBase<dim, 1+dim+n_tra>   *bc,
    TimerOutput                      &timer,
    const unsigned int                max_iteration_height)
    : OceanoOperator<dim, n_tra, degree, n_points_1d>(
      param, bc, timer, max_iteration_height)
  {}



  // This routine is also similar to its explicit counterpart. In fact we apply the same
  // concepts to the Additive Runge-Kutta (ARK) method where bottom friction is treated
  // implicitly. The cost of ARK scheme are quite higher with respect to the explicit
  // scheme. The main problem is memory access: the vectors to access at each stage are
  // `2 * (n_stages-1) +1`, the factor two is related to the presence of the stiff and
  // non-stiff part of the residual. Since ARK is needed only for the momentum equation we
  // mantain the explicit code of the previous section for the continuity equation and we
  // use a different code for the momemtum equation. For the latter we compute the stiff
  // and non-stiff residuals. Note that for the stiff residual we need to build the
  // residual associated to the friction term which can be done with a cell loop only.
  //
  // As the explicit counterpart (see the commented code), we have also an overhead w.r.t
  // the standard Runge-Kutta scheme. This is related to the condensed water depth but also
  // to the implicit friction that does not allow to update the solution with a single call
  // to `cell_loop()`. First we have to perform vector updates (assemble the right-hand-side
  // composed of the old solution times the mass-matrix plus the ImEx residuals. This is
  // cumulated into the auxiliary `vec_ki_discharge.front()`. Then we update the water height.
  // Only after, we invert the mass-matrix and we put the result into the new solution.
  // Differently from the explicit scheme, the mass-matrix is modified by the Implicit scheme
  // (it contains the Jacobian of the implicit part) and this is why we have a different call
  // to the mass matrix inversion. Moreover this should explain why, into the last
  // `cell_loop()`, the src vector contains the last updated solution: it is needed to compute
  // the water depth and the Jacobian of the bottom friction.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::perform_stage_hydro(
    const unsigned int                                             current_stage,
    const Number                                                   current_time,
    const Number                                                  *factor_residual,
    const Number                                                  *factor_tilde_residual,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &current_ri,
    std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_height,
    std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_discharge,
    LinearAlgebra::distributed::Vector<Number>                    &solution_height,
    LinearAlgebra::distributed::Vector<Number>                    &solution_discharge,
    LinearAlgebra::distributed::Vector<Number>                    &next_ri_height,
    LinearAlgebra::distributed::Vector<Number>                    &next_ri_discharge) const
  {

    const unsigned int n_stages = vec_ki_height.size()-1;
    const bool is_last_stage_implicit = Base::factor_matrix > 0.;

    {
      TimerOutput::Scope t(timer, "rk_stage hydro - integrals L_h");

      for (auto &i : bc->supercritical_inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->height_inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->discharge_inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->absorbing_outflow_boundaries)
        i.second->set_time(current_time);
      bc->problem_data->set_time(current_time);

      data.loop(&Base::local_apply_cell_height,
                &Base::local_apply_face_height,
                &Base::local_apply_boundary_face_height,
                static_cast<const Base *>(this),
                vec_ki_height[current_stage+1],
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);

      if (current_stage == n_stages-1 && !is_last_stage_implicit)
        {
          data.loop(
                &Base::local_apply_cell_discharge,
                &Base::local_apply_face_discharge,
                &Base::local_apply_boundary_face_discharge,
                static_cast<const Base *>(this),
                vec_ki_discharge[2*current_stage+1],
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
        }
      else
        {
          data.loop(
                &Base::local_apply_cell_nonstiff_discharge,
                &Base::local_apply_face_discharge,
                &Base::local_apply_boundary_face_discharge,
                static_cast<const Base *>(this),
                vec_ki_discharge[2*current_stage+1],
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
          data.cell_loop(
                &Base::local_apply_cell_stiff_discharge,
                static_cast<const Base *>(this),
                vec_ki_discharge[2*current_stage+2],
                current_ri,
                true);
        }
    }


    {
      TimerOutput::Scope t(timer, "rk_stage hydro - inv mass + vec upd");
      data.cell_loop(
        &Base::local_apply_cell_mass_height,
        static_cast<const Base *>(this),
        vec_ki_height.front(),
        solution_height,
        [&](const unsigned int start_range, const unsigned int end_range) {
          /* DEAL_II_OPENMP_SIMD_PRAGMA */
          for (unsigned int i = start_range; i < end_range; ++i)
            {
              Number k_i           = vec_ki_height[1].local_element(i);
              vec_ki_height.front().local_element(i)  = factor_residual[0]  * k_i;
              for (unsigned int j = 1; j < current_stage+1; ++j)
		{
		  k_i              = vec_ki_height[j+1].local_element(i);
		  vec_ki_height.front().local_element(i) += factor_residual[j]  * k_i;
		}
            }
	},
        std::function<void(const unsigned int, const unsigned int)>(),
        0);

      if (current_stage == n_stages-1)
        {
          solution_height.zero_out_ghost_values();
          data.cell_loop(
            &Base::local_apply_inverse_mass_matrix_height,
            static_cast<const Base *>(this),
            solution_height,
            vec_ki_height.front());
        }
      else
        {
          next_ri_height = solution_height;
          next_ri_height.zero_out_ghost_values();
          data.cell_loop(
            &Base::local_apply_inverse_mass_matrix_height,
            static_cast<const Base *>(this),
            next_ri_height,
            vec_ki_height.front());
        }


      if (current_stage == n_stages-1 && !is_last_stage_implicit)
        {
          data.cell_loop(
            &Base::local_apply_inverse_mass_matrix_discharge,
            static_cast<const Base *>(this),
            next_ri_discharge,
            vec_ki_discharge.front(),
            [&](const unsigned int start_range, const unsigned int end_range) {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  Number kex_i           = vec_ki_discharge[1].local_element(i);
                  vec_ki_discharge.front().local_element(i)  = factor_residual[0] * kex_i;
		  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
                      kex_i              = vec_ki_discharge[2*j+1].local_element(i);
                      const Number kim_i = vec_ki_discharge[2*j].local_element(i);
                      vec_ki_discharge.front().local_element(i) += factor_residual[j]   * kex_i
                                                                 + factor_residual[j-1] * kim_i;
                    }
                }
            },
            [&](const unsigned int start_range, const unsigned int end_range) {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  const Number sol_i     = next_ri_discharge.local_element(i);
                  solution_discharge.local_element(i)  += sol_i;
                }
            },
            1);
        }
      else
        {
          data.cell_loop(
            &Base::local_apply_cell_mass_discharge,
            static_cast<const Base *>(this),
            vec_ki_discharge.front(),
            solution_discharge,
            [&](const unsigned int start_range, const unsigned int end_range) {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  Number kex_i           = vec_ki_discharge[1].local_element(i);
                  Number kim_i           = vec_ki_discharge[2].local_element(i);
                  vec_ki_discharge.front().local_element(i)  = factor_residual[0]       * kex_i
                                                             + factor_tilde_residual[0] * kim_i;
                  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
		      kex_i              = vec_ki_discharge[2*j+1].local_element(i);
	              kim_i              = vec_ki_discharge[2*j+2].local_element(i);
		      vec_ki_discharge.front().local_element(i) += factor_residual[j]        * kex_i
		                                                 + factor_tilde_residual[j]  * kim_i;
		    }
                }
	    },
            std::function<void(const unsigned int, const unsigned int)>(),
            1);

          if (current_stage == n_stages-1)
            {
              solution_discharge.zero_out_ghost_values();
              data.cell_loop(
                &Base::local_apply_inverse_modified_mass_matrix_discharge,
                static_cast<const Base *>(this),
                solution_discharge,
                {vec_ki_discharge.front(), current_ri[0], current_ri[1]},
                true);
            }
         else
            {
              data.cell_loop(
                &Base::local_apply_inverse_modified_mass_matrix_discharge,
                static_cast<const Base *>(this),
                next_ri_discharge,
                {vec_ki_discharge.front(), current_ri[0], current_ri[1]},
                true);
            }
        }
    }
  }

} // namespace SpaceDiscretization

#endif //OCEANDGSEMIIMPLICIT_H
