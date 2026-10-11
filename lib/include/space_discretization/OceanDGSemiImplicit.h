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
    double factor_matrix;

    OceanoOperatorSemiImplicit(
      IO::ParameterHandler      &param,
      ICBC::BcBase<dim, 1+dim+n_tra> *bc,
      TimerOutput               &timer_output,
      const unsigned int         max_iteration_height);
    ~OceanoOperatorSemiImplicit() = default;

    void local_apply_cell_nonstiff_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;

    void local_apply_cell_stiff_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;

    void local_apply_face_nonstiff_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;

    void local_apply_boundary_face_nonstiff_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;

    void local_apply_inverse_modified_mass_matrix_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;

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
    using Base::model;

    using Base::data_quadrature_cell_0;
    using Base::data_quadrature_cell_2;
    using Base::data_quadrature_face;
    using Base::data_quadrature_boundary;

  protected:
    using Base::data;
    using Base::timer;
    using Base::num_flux;
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



  // For an implicit-explicit (IMEX) scheme, the right-hand side must be
  // partitioned into a stiff part and a non-stiff part. Here, the stiff part
  // appears only in the momentum equation, namely the bottom friction term;
  // the non-stiff part is what is left. Since the bottom friction term is a
  // local operator, only a local cell-wise apply is needed for the stiff
  // part.
  //
  // For the continuity equation, we reuse the apply function of the base
  // class, since nothing changes there.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_cell_nonstiff_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, -1, n_points_1d, 1, Number> phi_height(data,cell_range,0);
    FEEvaluation<dim, -1, n_points_1d, dim, Number> phi_discharge(data,cell_range,1);
    FEEvaluation<dim, -1, n_points_1d, dim, Number> phi_velocity(data,cell_range,1);

    const auto inv_degree = degree > 0 ? 1./(degree*degree) : 0.;

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_height.reinit(cell);
        phi_height.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_discharge.reinit(cell);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values);
        phi_velocity.reinit(cell);
        phi_velocity.gather_evaluate(src.back(), EvaluationFlags::gradients);

        VectorizedArray<Number> area_cell;
        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
          area_cell[v] = inv_degree * data.get_cell_iterator(cell,v)->measure();

        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
          {
            const auto z_q = phi_height.get_value(q);
            const auto dz_q = phi_height.get_gradient(q);
            const auto q_q = phi_discharge.get_value(q);
            const auto du_q = phi_velocity.get_gradient(q);
            const auto point_q = phi_discharge.quadrature_point(q);

            const auto zb_q = data_quadrature_cell_0.get_data(cell, q)[0];
            const auto data_onthefly_q =
              evaluate_function<dim, Number, dim+1>(*bc->problem_data, point_q, 2);

            phi_discharge.submit_gradient(
              model.advective_diffusive_flux(
                z_q, q_q, model.depth(z_q, zb_q)*du_q, zb_q, area_cell),
              q);

            phi_discharge.submit_value(
              model.source_nonstiff(z_q, q_q, dz_q, zb_q, data_onthefly_q),
              q);
          }

        phi_discharge.integrate_scatter(EvaluationFlags::values |
                                        EvaluationFlags::gradients,
                                       dst);
      }
  }

  // The face and boundary operators are non-stiff and identical to
  // those of the base class. However, we redefine them here so that the
  // `loop()` function can be used with this class and member function
  // pointers, avoiding lambdas.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_cell_stiff_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, -1, n_points_1d, 1, Number> phi_height(data,cell_range,0);
    FEEvaluation<dim, -1, n_points_1d, dim, Number> phi_discharge(data,cell_range,1);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_height.reinit(cell);
        phi_height.gather_evaluate(src[0], EvaluationFlags::values);
        phi_discharge.reinit(cell);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
          {
            const auto q_q = phi_discharge.get_value(q);
            const auto z_q = phi_height.get_value(q);

            const auto zb_q = data_quadrature_cell_0.get_data(cell, q)[0];
            const auto cf_q = data_quadrature_cell_0.get_data(cell, q)[1];

            phi_discharge.submit_value(
              model.source_stiff(z_q, q_q, zb_q, cf_q),
              q);
          }

        phi_discharge.integrate_scatter(EvaluationFlags::values,
                                       dst);
      }
  }

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_face_nonstiff_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &face_range) const
  {
    FEFaceEvaluation<dim, -1, degree + 2, 1, Number> phi_height_m(data, face_range,
                                                                      true, 0, 1);
    FEFaceEvaluation<dim, -1, degree + 2, 1, Number> phi_height_p(data, face_range,
                                                                      false, 0, 1);
    FEFaceEvaluation<dim, -1, degree + 2, dim, Number> phi_discharge_m(data, face_range,
                                                                      true, 1, 1);
    FEFaceEvaluation<dim, -1, degree + 2, dim, Number> phi_discharge_p(data, face_range,
                                                                      false, 1, 1);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_height_p.reinit(face);
        phi_height_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_discharge_p.reinit(face);
        phi_discharge_p.gather_evaluate(src[1], EvaluationFlags::values);

        phi_height_m.reinit(face);
        phi_height_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_discharge_m.reinit(face);
        phi_discharge_m.gather_evaluate(src[1], EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_discharge_m.n_q_points; ++q)
          {
            const auto z_m    = phi_height_m.get_value(q);
            const auto z_p    = phi_height_p.get_value(q);
            const auto normal = phi_discharge_m.normal_vector(q);
            const auto zb_m   = data_quadrature_face.get_data(face, 2*q);
            const auto zb_p   = data_quadrature_face.get_data(face, 2*q+1);

            auto numerical_flux_p =
              num_flux.numerical_advflux_weak(z_m,
                                              z_p,
                                              phi_discharge_m.get_value(q),
                                              phi_discharge_p.get_value(q),
                                              normal,
                                              zb_m,
                                              zb_p);
            auto numerical_flux_m = -numerical_flux_p;

            numerical_flux_m -=
              num_flux.numerical_presflux_strong(z_m,
                                                 z_p,
                                                 normal,
                                                 zb_m,
                                                 zb_p);
            numerical_flux_p +=
              num_flux.numerical_presflux_strong(z_p,
                                                 z_m,
                                                 normal,
                                                 zb_p,
                                                 zb_m);

            phi_discharge_m.submit_value(numerical_flux_m, q);
            phi_discharge_p.submit_value(numerical_flux_p, q);
          }

        phi_discharge_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_discharge_m.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_boundary_face_nonstiff_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &face_range) const
  {
    FEFaceEvaluation<dim, -1, n_points_1d, 1, Number> phi_height(data, face_range, true, 0);
    FEFaceEvaluation<dim, -1, n_points_1d, dim, Number> phi_discharge(data, face_range, true, 1);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_height.reinit(face);
        phi_height.gather_evaluate(src[0], EvaluationFlags::values);
        phi_discharge.reinit(face);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
          {
            const auto z_m    = phi_height.get_value(q);
            const auto q_m    = phi_discharge.get_value(q);
            const auto normal = phi_discharge.normal_vector(q);
            const auto point  = phi_height.quadrature_point(q);
            const auto zb_m   =
              data_quadrature_boundary.get_data(face-data.n_inner_face_batches(), q);

            VectorizedArray<Number> z_p;
            Tensor<1, dim, VectorizedArray<Number>> q_p;
            bool at_outflow;

            const auto boundary_id = data.get_boundary_id(face);

            Base::get_boundary_value(
              boundary_id, z_m, q_m, normal, point, zb_m, &z_p, q_p, &at_outflow);

            auto flux =
              num_flux.numerical_advflux_weak(z_m, z_p, q_m, q_p, normal, zb_m, zb_m);

            auto pressure_numerical_fluxes =
              num_flux.numerical_presflux_strong(z_m, z_p, normal, zb_m, zb_m);
            flux += pressure_numerical_fluxes;

            if (at_outflow)
              for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
                {
                  auto rho_u_dot_n = q_m * normal;
                  if (rho_u_dot_n[v] < -1e-12)
                    for (unsigned int d = 0; d < dim; ++d)
                      flux[d][v] = 0.;
                }

            phi_discharge.submit_value(-flux, q);
          }

        phi_discharge.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  // Implements the inverse mass matrix for the discharge equation. The
  // mass matrix includes a friction term, which is handled implicitly and this
  // why we named "modified". As in the explicit case, we use the fast inversion.
  // An important difference here is that, due to the presence of the non-polynomial
  // friction term, the integration cannot be exact. We assume that the number of
  // quadrature points used for the fast inversion, equal to the number of degrees
  // of freedom, is accurate enough.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorSemiImplicit<dim, n_tra, degree, n_points_1d>::
  local_apply_inverse_modified_mass_matrix_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, -1, degree + 1, dim, Number> phi_discharge(data, cell_range, 1, 2);
    FEEvaluation<dim, -1, degree + 1, 1, Number> phi_height_ri(data, cell_range, 0, 2);
    FEEvaluation<dim, -1, degree + 1, dim, Number> phi_discharge_ri(data, cell_range, 1, 2);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim, Number>
      inverse(phi_discharge);
    MatrixFreeOperatorsOceano::CellwiseInverseMassMatrixLumped<dim, 0, dim, Number>
      inverse_dry(phi_discharge);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_discharge.reinit(cell);
        phi_discharge.read_dof_values(src[0]);

        phi_height_ri.reinit(cell);
        phi_height_ri.gather_evaluate(src[1], EvaluationFlags::values);
        phi_discharge_ri.reinit(cell);
        phi_discharge_ri.gather_evaluate(src[2], EvaluationFlags::values);

        if (phi_discharge.get_active_fe_index())
          {
            AlignedVector<VectorizedArray<Number>> inverse_jxw(phi_discharge.n_q_points);
	    inverse.fill_inverse_JxW_values(inverse_jxw);

            for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
              {
                const auto z_q = phi_height_ri.get_value(q);
                const auto q_q = phi_discharge_ri.get_value(q);
                const auto zb_q = data_quadrature_cell_2.get_data(cell, q)[0];
                const auto cf_q = data_quadrature_cell_2.get_data(cell, q)[1];

                inverse_jxw[q] *= 1. / ( 1. + factor_matrix
                  * model.bottom_friction.jacobian(model.velocity(z_q, q_q, zb_q),
                                                   cf_q,
                                                   model.depth(z_q, zb_q))
                                       );
              }

            inverse.apply(inverse_jxw, dim, phi_discharge.begin_dof_values(),
              phi_discharge.begin_dof_values());
          }
        else
          {
            for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
              {
                const auto z_q = phi_height_ri.get_value(q);
                const auto q_q = phi_discharge_ri.get_value(q);
                const auto zb_q = data_quadrature_cell_2.get_data(cell, q)[0];
                const auto cf_q = data_quadrature_cell_2.get_data(cell, q)[1];

                phi_height_ri.submit_value(1. + factor_matrix
                    * model.bottom_friction.jacobian(model.velocity(z_q, q_q, zb_q),
                                                     cf_q,
                                                     model.depth(z_q, zb_q)),
                                              q);
              }
            phi_height_ri.integrate(EvaluationFlags::values);

            AlignedVector<VectorizedArray<Number>> cell_matrix(1);
            cell_matrix[0] = phi_height_ri.get_dof_value(0);
            inverse_dry.apply(&cell_matrix[0], phi_discharge.begin_dof_values(),
              phi_discharge.begin_dof_values());
          }

        phi_discharge.set_dof_values(dst);
      }
  }



  // The partitioning of the right-hand side of the shallow water equations into 
  // stiff and non-stiff components follows the requirement of using a large time
  // step that is not restricted by gravity wave speed or bottom friction coefficient.
  // This is obtained by treating semi-implicitly the mass flux term in the
  // continuity equation, the pressure gradient term and the bottom friction term
  // in the momentum equation. The hydrodynamic system is solved using the Schur
  // complement which is accomplished in three steps: 1/ an explicit update of the
  // discharge 2/ the solution of the discrete wave equation. 3/ the correction
  // of the discharge.
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
    const bool is_last_stage_implicit = factor_matrix > 0.;

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
                &OceanoOperatorSemiImplicit::local_apply_cell_nonstiff_discharge,
                &OceanoOperatorSemiImplicit::local_apply_face_nonstiff_discharge,
                &OceanoOperatorSemiImplicit::local_apply_boundary_face_nonstiff_discharge,
                this,
                vec_ki_discharge[2*current_stage+1],
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
          data.cell_loop(
                &OceanoOperatorSemiImplicit::local_apply_cell_stiff_discharge,
                this,
                vec_ki_discharge[2*current_stage+2],
                current_ri,
                true);
        }
    }


    {
      TimerOutput::Scope t(timer, "rk_stage hydro - inv mass + vec upd");
      if (current_stage == n_stages-1 && !is_last_stage_implicit)
        {

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

            solution_height.zero_out_ghost_values();
            data.cell_loop(
              &Base::local_apply_inverse_mass_matrix_height,
              static_cast<const Base *>(this),
              solution_height,
              vec_ki_height.front());


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

          // The first step consists in upadating the discharge considering
          // only explicit contributions.
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

          data.cell_loop(
            &OceanoOperatorSemiImplicit::local_apply_inverse_modified_mass_matrix_discharge,
            this,
            next_ri_discharge,
            {vec_ki_discharge.front(), current_ri[0], current_ri[1]},
            true);


          // The second step consists in updating the free-surface.
          data.loop(&Base::local_apply_cell_height,
                    &Base::local_apply_face_height,
                    &Base::local_apply_boundary_face_height,
                    static_cast<const Base *>(this),
                    vec_ki_height.front(),
                    {next_ri_height, next_ri_discharge},
                    true,
                    MatrixFree<dim, Number>::DataAccessOnFaces::values,
                    MatrixFree<dim, Number>::DataAccessOnFaces::values);
      
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
                  vec_ki_height.front().local_element(i)  += factor_residual[0]  * k_i;
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


          // The last step is the correction of the discharge
          /*data.loop(
                &OceanoOperatorSemiImplicit::local_apply_cell_pressure,
                &OceanoOperatorSemiImplicit::local_apply_face_pressure,
                &OceanoOperatorSemiImplicit::local_apply_boundary_face_pressure,
                this,
                vec_ki_discharge.front(),
                solution_height,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);*/

          data.cell_loop(
            &OceanoOperatorSemiImplicit::local_apply_inverse_modified_mass_matrix_discharge,
            this,
            vec_ki_discharge.front(),
            {vec_ki_discharge.front(), current_ri[0], current_ri[1]},
            std::function<void(const unsigned int, const unsigned int)>(),
            [&](const unsigned int start_range, const unsigned int end_range) {
              if (current_stage == n_stages-1)
                {
                  // DEAL_II_OPENMP_SIMD_PRAGMA
                  for (unsigned int i = start_range; i < end_range; ++i)
                    {
                      Number k_i           = vec_ki_discharge.front().local_element(i);
                      const Number sol_i   = next_ri_discharge.local_element(i);
                      solution_discharge.local_element(i)  = sol_i + factor_matrix * k_i;
                    }
                }
              else
                {
                  // DEAL_II_OPENMP_SIMD_PRAGMA
                  for (unsigned int i = start_range; i < end_range; ++i)
                    {
                      Number k_i            = vec_ki_discharge.front().local_element(i);
                      const Number sol_i    = next_ri_discharge.local_element(i);
                      next_ri_discharge.local_element(i) = sol_i + factor_matrix  * k_i;
                    }
                }
            },
            1);
        }
    }
  }

} // namespace SpaceDiscretization

#endif //OCEANDGSEMIIMPLICIT_H
