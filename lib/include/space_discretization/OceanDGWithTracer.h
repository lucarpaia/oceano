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
 *         Giuseppe Orlando,   2024
 */
#ifndef OCEANDGWITHTRACER_HPP
#define OCEANDGWITHTRACER_HPP

// The following files include the oceano libraries
#include <space_discretization/OceanDG.h>

/**
 * Namespace containing the spatial Operator
 */

namespace SpaceDiscretization
{

  using namespace dealii;

  using Number = double;

  // We have seen the `evaluate_function` during the construction of
  // the DG scheme for the ocean, these are helper functions to provide compact
  // evaluation calls as multiple points get batched together via a
  // VectorizedArray argument. Here they are used for inquiry the data
  // initial/boundary conditions for the tracer equations. Both functions, apparently
  // do the same operation. Tthe reason of the duplication, which
  // is resolved with an overloading depending on the third dummy argument,
  // is to handle the case of one single tracer and multiple tracers without
  // and `if` statement.
  template <int dim, typename Number, int n_vars>
  VectorizedArray<Number>
  evaluate_function_tracer(
    const Function<dim>                       &function,
    const Point<dim, VectorizedArray<Number>> &p_vectorized,
    const VectorizedArray<Number>              /*dummy*/)
  {
    VectorizedArray<Number> result;
    for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
      {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        result[v] = function.value(p, 1+dim);
      }
    return result;
  }

  template <int dim, typename Number, int n_tra>
  Tensor<1, n_tra, VectorizedArray<Number>>
  evaluate_function_tracer(
    const Function<dim>                             &function,
    const Point<dim, VectorizedArray<Number>>       &p_vectorized,
    const Tensor<1, n_tra, VectorizedArray<Number>> &/*dummy*/)
  {
    AssertDimension(function.n_components, 1+dim+n_tra);
    Tensor<1, n_tra, VectorizedArray<Number>> result;
    for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
      {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        for (unsigned int t = 0; t < n_tra; ++t)
          result[t][v] = function.value(p, 1+dim+t);
      }
    return result;
  }



  // @sect3{The OceanoWithTracerOperation class}

  // This class implements the evaluators for the tracer equations. It is
  // derived from the `OceanoOperator`. Thanks to a `using` construct we can
  // access also the base class members with the same variable name. This
  // helped a lot during the implementation.
  //
  // All methods of this class are new methods to compute operations specific to
  // tracers only: we have the discontinuous Galerkin cell and face evaluators,
  // methods for projections and error computation, the well known `perform_stage`
  // that timesteps the tracers. Only `reinit` overloads a base class with the same
  // name; the overloaded functions set up the MatrixFree variable with
  // the tracers.
  template <int dim, int n_tra, int degree, int n_points_1d>
  class OceanoOperatorWithTracer : public OceanoOperator<dim, n_tra, degree, n_points_1d>
  {
  public:
    static constexpr unsigned int n_quadrature_points_1d = n_points_1d;

    OceanoOperatorWithTracer(IO::ParameterHandler           &param,
                             ICBC::BcBase<dim, 1+dim+n_tra> *bc,
                             TimerOutput                    &timer_output);

    void reinit(const Mapping<dim> &   mapping,
                const DoFHandler<dim> &dof_handler_height,
                const DoFHandler<dim> &dof_handler_discharge,
                const DoFHandler<dim> &dof_handler_tracer);

#if defined TIMEINTEGRATOR_LOWSTORAGERUNGEKUTTA
    void
    perform_stage_tracers(
      const                                                          Number factor_solution,
      const                                                          Number factor_ai,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &current_ri,
      LinearAlgebra::distributed::Vector<Number>                    &vec_ki_tracer,
      LinearAlgebra::distributed::Vector<Number>                    &solution_tracer,
      LinearAlgebra::distributed::Vector<Number>                    &next_ri_tracer) const;

#elif defined TIMEINTEGRATOR_EXPLICITRUNGEKUTTA || defined TIMEINTEGRATOR_ADDITIVERUNGEKUTTA
    void
    perform_stage_tracers(
      const unsigned int                                             cur_stage,
      const Number                                                  *factor_residual,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &current_ri,
      std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_tracer,
      LinearAlgebra::distributed::Vector<Number>                    &solution_tracer,
      LinearAlgebra::distributed::Vector<Number>                    &next_ri_tracer) const;

#endif
    void project_tracers(const Function<dim> &                       function,
                         LinearAlgebra::distributed::Vector<Number> &solution_tracer) const;

    std::array<double, n_tra> compute_errors_tracers(
      const Function<dim> &                             function,
      const LinearAlgebra::distributed::Vector<Number> &solution_tracer) const;

    void evaluate_vector_field(
      const LinearAlgebra::distributed::Vector<Number>        &solution_height,
      const LinearAlgebra::distributed::Vector<Number>        &solution_discharge,
      const LinearAlgebra::distributed::Vector<Number>        &solution_tracer,
      const std::vector<Point<dim>>                           &evaluation_points,
      LinearAlgebra::distributed::Vector<Number>              &computed_vector_quantities,
      std::vector<LinearAlgebra::distributed::Vector<Number>> &computed_scalar_quantities) const;

    using OceanoOperator<dim, n_tra, degree, n_points_1d>::bc;
    
  private:

    using OceanoOperator<dim, n_tra, degree, n_points_1d>::data;

    using OceanoOperator<dim, n_tra, degree, n_points_1d>::num_flux;

    using OceanoOperator<dim, n_tra, degree, n_points_1d>::timer;

    using OceanoOperator<dim, n_tra, degree, n_points_1d>::model;

    void local_apply_inverse_mass_matrix_tracer(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    void local_apply_cell_tracer(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;

    void local_apply_face_tracer(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;

    void local_apply_boundary_face_tracer(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;
  };



  template <int dim, int n_tra, int degree, int n_points_1d>
  OceanoOperatorWithTracer<dim, n_tra, degree, n_points_1d>::OceanoOperatorWithTracer(
    IO::ParameterHandler             &param,
    ICBC::BcBase<dim, 1+dim+n_tra>   *bc,
    TimerOutput                      &timer)
    : OceanoOperator<dim, n_tra, degree, n_points_1d>(param, bc, timer)
  {}



  // For the initialization of the ocean operator, we set up the MatrixFree
  // variable contained in the class.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorWithTracer<dim, n_tra, degree, n_points_1d>::reinit(
    const Mapping<dim> &   mapping,
    const DoFHandler<dim> &dof_handler_height,
    const DoFHandler<dim> &dof_handler_discharge,
    const DoFHandler<dim> &dof_handler_tracer)
  {
    const AffineConstraints<double>            dummy;
    std::vector<const DoFHandler<dim> *> dof_handlers;
    std::vector<const AffineConstraints<double> *>
      constraints = {&dummy, &dummy};
    const std::vector<Quadrature<1>> quadratures = {QGauss<1>(n_points_1d),
                                                    QGauss<1>(degree + 1)};

    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points |
       update_values);
    additional_data.mapping_update_flags_inner_faces =
      (update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
    additional_data.mapping_update_flags_boundary_faces =
      (update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, Number>::AdditionalData::none;

    dof_handlers.push_back(&dof_handler_height);
    dof_handlers.push_back(&dof_handler_discharge);

    constraints.push_back(&dummy);
    dof_handlers.push_back(&dof_handler_tracer);

    data.reinit(
      mapping, dof_handlers, constraints, quadratures, additional_data);
  }



  // @sect4{Local evaluators}

  // Now we proceed to the local evaluators for the ocean problem. The
  // evaluators are relatively simple and follow what has been presented in
  // step-37, step-48, or step-59. The first notable difference is the fact
  // that we use a FEEvaluation with a non-standard number of quadrature
  // points. Whereas we previously always set the number of quadrature points
  // to equal the polynomial degree plus one (ensuring exact integration on
  // affine element shapes), we now set the number quadrature points as a
  // separate variable (e.g. the polynomial degree plus two or three halves of
  // the polynomial degree) to more accurately handle nonlinear terms. Since
  // the evaluator is fed with the appropriate loop lengths via the template
  // argument and keeps the number of quadrature points in the whole cell in
  // the variable FEEvaluation::n_q_points, we now automatically operate on
  // the more accurate formula without further changes.
  //
  // The second difference is due to the fact that we are now evaluating a
  // multi-component system, as opposed to the scalar systems considered
  // previously. The matrix-free framework provides several ways to handle the
  // multi-component case. One variant utilizes an FEEvaluation
  // object with multiple components embedded into it, specified by the fourth
  // template argument `n_vars` for the components in the shallow water system.
  // The alternative variant followed here uses several FEEvaluation objects,
  // a scalar one for the height, a vector-valued one with `dim` components for the
  // momentum, and another scalar evaluator for the tracers. To ensure that
  // those components point to the correct part of the solution, the
  // constructor of FEEvaluation takes three optional integer arguments after
  // the required MatrixFree field, namely the number of the DoFHandler for
  // multi-DoFHandler systems (taking the first by default), the number of the
  // quadrature point in case there are multiple Quadrature objects (see more
  // below), and as a third argument the component within a vector system. As
  // we have a single vector for all components, we would go with the third
  // argument, and set it to `0` for the density, `1` for the vector-valued
  // momentum, and `dim+1` for the energy slot. FEEvaluation then picks the
  // appropriate subrange of the solution vector during
  // FEEvaluationBase::read_dof_values() and
  // FEEvaluation::distributed_local_to_global() or the more compact
  // FEEvaluation::gather_evaluate() and FEEvaluation::integrate_scatter()
  // calls.
  //
  // When it comes to the evaluation of the body force vector, we need to call
  // the `evaluate_function()` method we provided above; Since the body force,
  // in the general case is not a constant, we must call it inside the loop over
  // quadrature point data, which of course is quite expensive.
  // Once the body force has been computed we compute the body force term associated
  // the right-hand side of the shallow water equation inside the `source()` function,
  // a member function of the model class.
  //
  //
  // The rest follows the other tutorial programs. Since we have implemented
  // all physics for the governing equations in the separate `model.flux()`
  // and `model.source()` functions, all we have to do here is to call this function
  // given the current solution evaluated at quadrature points, returned by
  // `phi.get_value(q)`, and tell the FEEvaluation object to queue the flux
  // for testing it by the gradients of the shape functions (which is a Tensor
  // of outer `n_vars` components, each holding a tensor of `dim` components
  // for the $x,y,z$ component of the shallow water flux). One final thing worth
  // mentioning is the order in which we queue the data for testing by the
  // value of the test function, `phi.submit_value()`, in case we are given an
  // external function: We must do this after calling `phi.get_value(q)`,
  // because `get_value()` (reading the solution) and `submit_value()`
  // (queuing the value for multiplication by the test function and summation
  // over quadrature points) access the same underlying data field. Here it
  // would be easy to achieve also without temporary variable `w_q` since
  // there is no mixing between values and gradients. For more complicated
  // setups, one has to first copy out e.g. both the value and gradient at a
  // quadrature point and then queue results again by
  // FEEvaluationBase::submit_value() and FEEvaluationBase::submit_gradient().
  // Note that the flux term contains mass-flux and non-linear advection flux
  // while the pressure flux appears in the source term, where it is written
  // in a non-conservative form $h\nabla\zeta$. This why the `model.source()`
  // function take as argument the gradient of the free-surface.
  //
  // As a final note, we mention that we do not use the first MatrixFree
  // argument of this function, which is a call-back from MatrixFree::loop().
  // The interfaces imposes the present list of arguments, but since we are in
  // a member function where the MatrixFree object is already available as the
  // `data` variable, we stick with that to avoid confusion.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorWithTracer<dim, n_tra, degree, n_points_1d>::local_apply_cell_tracer(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, degree, n_points_1d, 1, Number> phi_height(data,0);
    FEEvaluation<dim, degree, n_points_1d, dim, Number> phi_discharge(data,1);
    FEEvaluation<dim, degree, n_points_1d, n_tra, Number> phi_tracer(data,2);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_height.reinit(cell);
        phi_height.gather_evaluate(src[0], EvaluationFlags::values);
        phi_discharge.reinit(cell);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values);
        phi_tracer.reinit(cell);
        phi_tracer.gather_evaluate(src[2], EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_tracer.n_q_points; ++q)
          {
            const auto z_q = phi_height.get_value(q);
            const auto q_q = phi_discharge.get_value(q);
            const auto t_q = phi_tracer.get_value(q);
            const VectorizedArray<Number> data_q =
              evaluate_function<dim, Number>(
                *bc->problem_data, phi_tracer.quadrature_point(q), 0);

            phi_tracer.submit_gradient(
              model.tracerflux(z_q, q_q, t_q, data_q), q);
          }

        phi_tracer.integrate_scatter(EvaluationFlags::gradients,
                                       dst);
      }
  }

  // The next function concerns the computation of integrals on interior
  // faces, where we need evaluators from both cells adjacent to the face. We
  // associate the variable `phi_m` with the solution component $\mathbf{w}^-$
  // and the variable `phi_p` with the solution component $\mathbf{w}^+$. We
  // distinguish the two sides in the constructor of FEFaceEvaluation by the
  // second argument, with `true` for the interior side and `false` for the
  // exterior side, with interior and exterior denoting the orientation with
  // respect to the normal vector.
  //
  // Note that the calls FEFaceEvaluation::gather_evaluate() and
  // FEFaceEvaluation::integrate_scatter() combine the access to the vectors
  // and the sum factorization parts. This combined operation not only saves a
  // line of code, but also contains an important optimization: Given that we
  // use a nodal basis in terms of the Lagrange polynomials in the points of
  // the Gauss-Lobatto quadrature formula, only $(p+1)^{d-1}$ out of the
  // $(p+1)^d$ basis functions evaluate to non-zero on each face. Thus, the
  // evaluator only accesses the necessary data in the vector and skips the
  // parts which are multiplied by zero. If we had first read the vector, we
  // would have needed to load all data from the vector, as the call in
  // isolation would not know what data is required in subsequent
  // operations. If the subsequent FEFaceEvaluation::evaluate() call requests
  // values and derivatives, indeed all $(p+1)^d$ vector entries for each
  // component are needed, as the normal derivative is nonzero for all basis
  // functions.
  //
  // The arguments to the evaluators as well as the procedure is similar to
  // the cell evaluation. We again use the more accurate (over-)integration
  // scheme due to the nonlinear terms, specified as the third template
  // argument in the list. At the quadrature points, we then go to our
  // free-standing functions for the numerical flux. They receives the solution
  // evaluated at quadrature points from both sides (i.e., $\mathbf{w}^-$ and
  // $\mathbf{w}^+$), as well as the normal vector onto the minus side.
  // We separate numerical fluxes coming from terms written in weak form from
  // terms written in strong form. A simple trick is used to have
  // discontinuous bathymetry at the quadrature points along the edges. The
  // quadrature point is shifted perpendicularly to the edge direction by a
  // very small quantity. Givent the outward sign of the normal, the offset is
  // positive for the "there" side and negative for the "here" side.
  //
  // For the shallow water equations mass-flux
  // and non-linear advection terms are coded in weak form while pressure force
  // appears in strong form. For instance, the weak form for the pressure term
  // has proved to preserve easily the lake at rest solution. Depending on weak/strong
  // formulation, the flux term has different forms. We cumulate numerical flux
  // in two different arrays for both sided.
  // As explained above, the weak numerical flux is already multiplied by the normal
  // vector from the minus side. We need to switch the sign because the
  // boundary term comes with a minus sign in the weak form derived in the
  // introduction. The flux is then queued for testing both on the minus sign
  // and on the plus sign, with switched sign as the normal vector from the
  // plus side is exactly opposed to the one from the minus side.  The strong
  // form goes with the same sign on both sides. For this reason we have used
  // another function to compute it.
  //
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorWithTracer<dim, n_tra, degree, n_points_1d>::local_apply_face_tracer(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &face_range) const
  {
    FEFaceEvaluation<dim, degree, n_points_1d, 1, Number> phi_height_m(data,
                                                                      true, 0);
    FEFaceEvaluation<dim, degree, n_points_1d, 1, Number> phi_height_p(data,
                                                                      false, 0);
    FEFaceEvaluation<dim, degree, n_points_1d, dim, Number> phi_discharge_m(data,
                                                                      true, 1);
    FEFaceEvaluation<dim, degree, n_points_1d, dim, Number> phi_discharge_p(data,
                                                                      false, 1);
    FEFaceEvaluation<dim, degree, n_points_1d, n_tra, Number> phi_tracer_m(data,
                                                                      true, 2);
    FEFaceEvaluation<dim, degree, n_points_1d, n_tra, Number> phi_tracer_p(data,
                                                                      false, 2);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_height_p.reinit(face);
        phi_height_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_discharge_p.reinit(face);
        phi_discharge_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_tracer_p.reinit(face);
        phi_tracer_p.gather_evaluate(src[2], EvaluationFlags::values);

        phi_height_m.reinit(face);
        phi_height_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_discharge_m.reinit(face);
        phi_discharge_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_tracer_m.reinit(face);
        phi_tracer_m.gather_evaluate(src[2], EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_tracer_m.n_q_points; ++q)
          {
            const VectorizedArray<Number> data_m =
              evaluate_function<dim, Number>(*bc->problem_data,
                phi_tracer_m.quadrature_point(q)-1e-12*phi_tracer_m.normal_vector(q), 0);
            const VectorizedArray<Number> data_p =
              evaluate_function<dim, Number>(*bc->problem_data,
                phi_tracer_m.quadrature_point(q)+1e-12*phi_tracer_m.normal_vector(q), 0);

            auto numerical_flux_p =
              num_flux.numerical_tracflux_weak(phi_height_m.get_value(q),
                                               phi_height_p.get_value(q),
                                               phi_discharge_m.get_value(q),
                                               phi_discharge_p.get_value(q),
                                               phi_tracer_m.get_value(q),
                                               phi_tracer_p.get_value(q),
                                               phi_tracer_m.normal_vector(q),
                                               data_m,
                                               data_p);

            phi_tracer_m.submit_value(-numerical_flux_p, q);
            phi_tracer_p.submit_value(numerical_flux_p, q);
          }

        phi_tracer_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_tracer_m.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  // For faces located at the boundary, we need to impose the appropriate
  // boundary conditions. In this tutorial program, we implement four cases as
  // mentioned above. The discontinuous Galerkin
  // method imposes boundary conditions not as constraints, but only
  // weakly. Thus, the various conditions are imposed by finding an appropriate
  // <i>exterior</i> quantity $\mathbf{w}^+$ that is then handed to the
  // numerical flux function also used for the interior faces. In essence,
  // we "pretend" a state on the outside of the domain in such a way that
  // if that were reality, the solution of the PDE would satisfy the boundary
  // conditions we want.
  //
  // For wall boundaries, we need to impose a no-normal-flux condition on the
  // momentum variable, whereas we use a Neumann condition for the density and
  // energy with $\rho^+ = \rho^-$ and $E^+ = E^-$. To achieve the no-normal
  // flux condition, we set the exterior values to the interior values and
  // subtract two times the velocity in wall-normal direction, i.e., in the
  // direction of the normal vector.
  //
  // For inflow boundaries, we simply set the given Dirichlet data
  // $\mathbf{w}_\mathrm{D}$ as a boundary value. An alternative would have been
  // to use $\mathbf{w}^+ = -\mathbf{w}^- + 2 \mathbf{w}_\mathrm{D}$, the
  // so-called mirror principle.
  //
  // The imposition of outflow is essentially a Neumann condition, i.e.,
  // setting $\mathbf{w}^+ = \mathbf{w}^-$. For the case of supercritical outflow,
  // we still need to impose a value for the energy, which we derive from the
  // respective function. A special step is needed for the case of
  // <i>backflow</i>, i.e., the case where there is a momentum flux into the
  // domain on the Neumann portion. According to the literature (a fact that can
  // be derived by appropriate energy arguments), we must switch to another
  // variant of the flux on inflow parts, see Gravemeier, Comerford,
  // Yoshihara, Ismail, Wall, "A novel formulation for Neumann inflow
  // conditions in biomechanics", Int. J. Numer. Meth. Biomed. Eng., vol. 28
  // (2012). Here, the momentum term needs to be added once again, which
  // corresponds to removing the flux contribution on the momentum
  // variables. We do this in a post-processing step, and only for the case
  // when we both are at an outflow boundary and the dot product between the
  // normal vector and the momentum (or, equivalently, velocity) is
  // negative.
  //
  // The fourth boundary conditions is very important in coastal ocean and
  // hydraulic simulation since the flow conditions are essentially
  // subcritical. The above outflow condition assume that all the eigenvalues
  // are outgoing which cannot be true in the subcritical regime.
  // We have implemented another outflow imposition where we recover the
  // information coming from the ingoing eigenvalue. We compute a
  // boundary state from a far-field state (typically a flow at rest) from
  // the theory of characteristics, that is by equating at the boundary location,
  // the outgoing Riemann invariant with the ingoing one.
  // As we work on data of several quadrature points at once for
  // SIMD vectorizations, we here need to explicitly loop over the array
  // entries of the SIMD array.
  //
  // In the implementation below, we check for the various types
  // of boundaries at the level of quadrature points. Of course, we could also
  // have moved the decision out of the quadrature point loop and treat entire
  // faces as of the same kind, which avoids some map/set lookups in the inner
  // loop over quadrature points. However, the loss of efficiency is hardly
  // noticeable, so we opt for the simpler code here. Also note that the final
  // `else` clause will catch the case when some part of the boundary was not
  // assigned any boundary condition via `OceanoOperator::set_..._boundary(...)`.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorWithTracer<dim, n_tra, degree, n_points_1d>::local_apply_boundary_face_tracer(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &face_range) const
  {
    FEFaceEvaluation<dim, degree, n_points_1d, 1, Number> phi_height(data, true, 0);
    FEFaceEvaluation<dim, degree, n_points_1d, dim, Number> phi_discharge(data, true, 1);
    FEFaceEvaluation<dim, degree, n_points_1d, n_tra, Number> phi_tracer(data, true, 2);

    const unsigned int n_vars = dim+1;

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_height.reinit(face);
        phi_height.gather_evaluate(src[0], EvaluationFlags::values);
        phi_discharge.reinit(face);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values);
        phi_tracer.reinit(face);
        phi_tracer.gather_evaluate(src[2], EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_tracer.n_q_points; ++q)
          {
            const auto z_m    = phi_height.get_value(q);
            const auto q_m    = phi_discharge.get_value(q);
            const auto t_m    = phi_tracer.get_value(q);
            const auto normal = phi_tracer.normal_vector(q);
            const VectorizedArray<Number> data_m =
              evaluate_function<dim, Number>(
                *bc->problem_data, phi_tracer.quadrature_point(q), 0);

            auto rho_u_dot_n = q_m * normal;

            VectorizedArray<Number> z_p;
            Tensor<1, dim, VectorizedArray<Number>> q_p;
            Tensor<1, n_vars, VectorizedArray<Number>> w_p;
            auto t_p = t_m;
            const auto boundary_id = data.get_boundary_id(face);
            if (bc->wall_boundaries.find(boundary_id) != bc->wall_boundaries.end())
              {
                z_p = z_m;
                q_p = q_m - 2. * rho_u_dot_n * normal;
                t_p = t_m;
              }
            else if (bc->inflow_boundaries.find(boundary_id) !=
                     bc->inflow_boundaries.end())
              {
                w_p =
                  evaluate_function<dim, Number, n_vars>(
                		  *bc->inflow_boundaries.find(boundary_id)->second,
                                  phi_tracer.quadrature_point(q));
                z_p = w_p[0];
                for (unsigned int d = 0; d < dim; ++d) q_p[d] = w_p[d+1];
                t_p = evaluate_function_tracer<dim, Number, n_tra>(
                		  *bc->inflow_boundaries.find(boundary_id)->second,
                                  phi_tracer.quadrature_point(q),
                                  phi_tracer.get_value(q));
              }
            else if (bc->supercritical_outflow_boundaries.find(boundary_id) !=
                     bc->supercritical_outflow_boundaries.end())
              {
                z_p = z_m;
                q_p = q_m;
                t_p = t_m;
              }
#if defined MODEL_SHALLOWWATER
            else if (bc->subcritical_outflow_boundaries.find(boundary_id) !=
                     bc->subcritical_outflow_boundaries.end())
              {
                w_p =
                  evaluate_function<dim, Number, n_vars>(
                    *bc->subcritical_outflow_boundaries.find(boundary_id)->second,
                      phi_tracer.quadrature_point(q));
                z_p = w_p[0];
                for (unsigned int d = 0; d < dim; ++d) q_p[d] = w_p[d+1];
                const auto r_p
                  = model.riemann_invariant_p(z_m, q_m, normal, data_m);
                const auto r_m
                  = model.riemann_invariant_m(z_p, q_p, normal, data_m);
                const auto c_b = 0.25 * (r_p - r_m);
                const auto h_b = c_b * c_b / model.g;
                const auto u_b = 0.5 * (r_p + r_m);

                z_p = h_b - data_m;
                const auto norm = 1./normal.norm_square();
                q_p =  u_b * h_b * norm * normal;
                t_p = t_m;
              }
#endif
            else
              AssertThrow(false,
                          ExcMessage("Unknown boundary id, did "
                                     "you set a boundary condition for "
                                     "this part of the domain boundary?"));

            auto flux = num_flux.numerical_tracflux_weak(
                z_m, z_p, q_m, q_p, t_m, t_p, normal, data_m, data_m);

            phi_tracer.submit_value(-flux, q);
          }

        phi_tracer.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  // The next function implements the inverse mass matrix operation. The
  // algorithms and rationale have been discussed extensively in the
  // introduction, so we here limit ourselves to the technicalities of the
  // MatrixFreeOperators::CellwiseInverseMassMatrix class. It does similar
  // operations as the forward evaluation of the mass matrix, except with a
  // different interpolation matrix, representing the inverse $S^{-1}$
  // factors. These represent a change of basis from the specified basis (in
  // this case, the Lagrange basis in the points of the Gauss--Lobatto
  // quadrature formula) to the Lagrange basis in the points of the Gauss
  // quadrature formula. In the latter basis, we can apply the inverse of the
  // point-wise `JxW` factor, i.e., the quadrature weight times the
  // determinant of the Jacobian of the mapping from reference to real
  // coordinates. Once this is done, the basis is changed back to the nodal
  // Gauss-Lobatto basis again. All of these operations are done by the
  // `apply()` function below. What we need to provide is the local fields to
  // operate on (which we extract from the global vector by an FEEvaluation
  // object) and write the results back to the destination vector of the mass
  // matrix operation.
  //
  // One thing to note is that we added two integer arguments (that are
  // optional) to the constructor of FEEvaluation, the first being 0
  // (selecting among the DoFHandler in multi-DoFHandler systems; here, we
  // only have one) and the second being 1 to make the quadrature formula
  // selection. As we use the quadrature formula 0 for the over-integration of
  // nonlinear terms, we use the formula 1 with the default $p+1$ (or
  // `degree+1` in terms of the variable name) points for the mass
  // matrix. This leads to square contributions to the mass matrix and ensures
  // exact integration, as explained in the introduction.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorWithTracer<dim, n_tra, degree, n_points_1d>::local_apply_inverse_mass_matrix_tracer(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, degree, degree + 1, n_tra, Number> phi_tracer(data, 2, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, n_tra, Number>
      inverse(phi_tracer);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_tracer.reinit(cell);
        phi_tracer.read_dof_values(src);

        inverse.apply(phi_tracer.begin_dof_values(), phi_tracer.begin_dof_values());

        phi_tracer.set_dof_values(dst);
      }
  }

  // @sect4{The apply() and related functions}

  // We now come to the function which implements the evaluation of the ocean
  // operator as a whole, i.e., $\mathcal M^{-1} \mathcal L(t, \mathbf{w})$,
  // calling into the local evaluators presented above. The steps should be
  // clear from the previous code. One thing to note is that we need to adjust
  // the time in the functions we have associated with the various parts of
  // the boundary, in order to be consistent with the equation in case the
  // boundary data is time-dependent. Then, we call MatrixFree::loop() to
  // perform the cell and face integrals, including the necessary ghost data
  // exchange in the `src` vector. The seventh argument to the function,
  // `true`, specifies that we want to zero the `dst` vector as part of the
  // loop, before we start accumulating integrals into it. This variant is
  // preferred over explicitly calling `dst = 0.;` before the loop as the
  // zeroing operation is done on a subrange of the vector in parts that are
  // written by the integrals nearby. This enhances data locality and allows
  // for caching, saving one roundtrip of vector data to main memory and
  // enhancing performance. The last two arguments to the loop determine which
  // data is exchanged: Since we only access the values of the shape functions
  // one faces, typical of first-order hyperbolic problems, and since we have
  // a nodal basis with nodes at the reference element surface, we only need
  // to exchange those parts. This again saves precious memory bandwidth.
  //
  // Once the spatial operator $\mathcal L$ is applied, we need to make a
  // second round and apply the inverse mass matrix. Here, we call
  // MatrixFree::cell_loop() since only cell integrals appear. The cell loop
  // is cheaper than the full loop as access only goes to the degrees of
  // freedom associated with the locally owned cells, which is simply the
  // locally owned degrees of freedom for DG discretizations. Thus, no ghost
  // exchange is needed here.
  //
  // Around all these functions, we put timer scopes to record the
  // computational time for statistics about the contributions of the various
  // parts.


  // Let us move to the function that does an entire stage of a Runge--Kutta
  // update. It calls OceanoOperator::apply() followed by some updates
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
#if defined TIMEINTEGRATOR_LOWSTORAGERUNGEKUTTA
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorWithTracer<dim, n_tra, degree, n_points_1d>::perform_stage_tracers(
    const Number                                                   factor_solution,
    const Number                                                   factor_ai,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &current_ri,
    LinearAlgebra::distributed::Vector<Number>                    &vec_ki_tracer,
    LinearAlgebra::distributed::Vector<Number>                    &solution_tracer,
    LinearAlgebra::distributed::Vector<Number>                    &next_ri_tracer) const  
  {
    {
      TimerOutput::Scope t(timer, "rk_stage tracer - integrals L_h");

      data.loop(&OceanoOperatorWithTracer::local_apply_cell_tracer,
                &OceanoOperatorWithTracer::local_apply_face_tracer,
                &OceanoOperatorWithTracer::local_apply_boundary_face_tracer,
                this,
                vec_ki_tracer,
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }


    {
      TimerOutput::Scope t(timer, "rk_stage tracer - inv mass + vec upd");
      data.cell_loop(
        &OceanoOperatorWithTracer::local_apply_inverse_mass_matrix_tracer,
        this,
        next_ri_tracer,
        vec_ki_tracer,
        std::function<void(const unsigned int, const unsigned int)>(),
        [&](const unsigned int start_range, const unsigned int end_range) {
          const Number ai = factor_ai;
          const Number bi = factor_solution;
          if (ai == Number())
            {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  const Number k_i          = next_ri_tracer.local_element(i);
                  const Number sol_i        = solution_tracer.local_element(i);
                  solution_tracer.local_element(i) = sol_i + bi * k_i;
                }
            }
          else
            {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  const Number k_i          = next_ri_tracer.local_element(i);
                  const Number sol_i        = solution_tracer.local_element(i);
                  solution_tracer.local_element(i) = sol_i + bi * k_i;
                  next_ri_tracer.local_element(i)  = sol_i + ai * k_i;
                }
            }
        },
        2);
    }    
  }

#elif defined TIMEINTEGRATOR_EXPLICITRUNGEKUTTA || defined TIMEINTEGRATOR_ADDITIVERUNGEKUTTA
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorWithTracer<dim, n_tra, degree, n_points_1d>::perform_stage_tracers(
    const unsigned int                                             current_stage,
    const Number                                                  *factor_residual,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &current_ri,
    std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_tracer,
    LinearAlgebra::distributed::Vector<Number>                    &solution_tracer,
    LinearAlgebra::distributed::Vector<Number>                    &next_ri_tracer) const
  {
    {
      TimerOutput::Scope t(timer, "rk_stage tracer - integrals L_h");

      data.loop(&OceanoOperatorWithTracer::local_apply_cell_tracer,
                &OceanoOperatorWithTracer::local_apply_face_tracer,
                &OceanoOperatorWithTracer::local_apply_boundary_face_tracer,
                this,
                vec_ki_tracer.front(),
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }


    {
      unsigned int n_stages = vec_ki_tracer.size()-1; //lrp: may be optimized passing as argument
      TimerOutput::Scope t(timer, "rk_stage tracer - inv mass + vec upd");
      data.cell_loop(
        &OceanoOperatorWithTracer::local_apply_inverse_mass_matrix_tracer,
        this,
        vec_ki_tracer[current_stage+1],
        vec_ki_tracer.front(),
        std::function<void(const unsigned int, const unsigned int)>(),
        [&](const unsigned int start_range, const unsigned int end_range) {
          if (current_stage == n_stages-1)
            {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  Number k_i           = vec_ki_tracer[1].local_element(i);
                  const Number sol_i   = solution_tracer.local_element(i);
                  solution_tracer.local_element(i)  = sol_i + factor_residual[0]  * k_i;
		  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
                      k_i = vec_ki_tracer[j+1].local_element(i);
                      solution_tracer.local_element(i) += factor_residual[j]  * k_i;
                    }
                }
            }
          else
            {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  Number k_i            = vec_ki_tracer[1].local_element(i);
                  const Number sol_i    = solution_tracer.local_element(i);
                  next_ri_tracer.local_element(i) = sol_i + factor_residual[0]  * k_i;
		  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
                      k_i = vec_ki_tracer[j+1].local_element(i);
                      next_ri_tracer.local_element(i) += factor_residual[j]  * k_i;
                    }
                }
            }
        },
        2);
    }
  }

#endif

  // Having discussed the implementation of the functions that deal with
  // advancing the solution by one time step, let us now move to functions
  // that implement other, ancillary operations. Specifically, these are
  // functions that compute projections, evaluate errors, and compute the speed
  // of information transport on a cell.
  //
  // The first of these functions is essentially equivalent to
  // VectorTools::project(), just much faster because it is specialized for DG
  // elements where there is no need to set up and solve a linear system, as
  // each element has independent basis functions. The reason why we show the
  // code here, besides a small speedup of this non-critical operation, is that
  // it shows additional functionality provided by
  // MatrixFreeOperators::CellwiseInverseMassMatrix.
  //
  // The projection operation works as follows: If we denote the matrix of
  // shape functions evaluated at quadrature points by $S$, the projection on
  // cell $K$ is an operation of the form $\underbrace{S J^K S^\mathrm
  // T}_{\mathcal M^K} \mathbf{w}^K = S J^K
  // \tilde{\mathbf{w}}(\mathbf{x}_q)_{q=1:n_q}$, where $J^K$ is the diagonal
  // matrix containing the determinant of the Jacobian times the quadrature
  // weight (JxW), $\mathcal M^K$ is the cell-wise mass matrix, and
  // $\tilde{\mathbf{w}}(\mathbf{x}_q)_{q=1:n_q}$ is the evaluation of the
  // field to be projected onto quadrature points. (In reality the matrix $S$
  // has additional structure through the tensor product, as explained in the
  // introduction.) This system can now equivalently be written as
  // $\mathbf{w}^K = \left(S J^K S^\mathrm T\right)^{-1} S J^K
  // \tilde{\mathbf{w}}(\mathbf{x}_q)_{q=1:n_q} = S^{-\mathrm T}
  // \left(J^K\right)^{-1} S^{-1} S J^K
  // \tilde{\mathbf{w}}(\mathbf{x}_q)_{q=1:n_q}$. Now, the term $S^{-1} S$ and
  // then $\left(J^K\right)^{-1} J^K$ cancel, resulting in the final
  // expression $\mathbf{w}^K = S^{-\mathrm T}
  // \tilde{\mathbf{w}}(\mathbf{x}_q)_{q=1:n_q}$. This operation is
  // implemented by
  // MatrixFreeOperators::CellwiseInverseMassMatrix::transform_from_q_points_to_basis().
  // The name is derived from the fact that this projection is simply
  // the multiplication by $S^{-\mathrm T}$, a basis change from the
  // nodal basis in the points of the Gaussian quadrature to the given finite
  // element basis. Note that we call FEEvaluation::set_dof_values() to write
  // the result into the vector, overwriting previous content, rather than
  // accumulating the results as typical in integration tasks -- we can do
  // this because every vector entry has contributions from only a single
  // cell for discontinuous Galerkin discretizations.
  //
  // The quadrature choosen to do the integral is the normal one stored in
  // the finite element evaluation class, thus a Gaussian quadrature
  // with the points lying at the interior
  // of the cell. This allows to mantain a discontinuous datum with
  // the jump passing through the edges of the cell.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorWithTracer<dim, n_tra, degree, n_points_1d>::project_tracers(
    const Function<dim> &                       function,
    LinearAlgebra::distributed::Vector<Number> &solution_tracer) const
  {
    FEEvaluation<dim, degree, degree + 1, n_tra, Number> phi_tracer(data, 2, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, n_tra, Number>
      inverse_tracer(phi_tracer);
    solution_tracer.zero_out_ghost_values();
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi_tracer.reinit(cell);
        phi_tracer.gather_evaluate(solution_tracer, EvaluationFlags::values);
        for (unsigned int q = 0; q < phi_tracer.n_q_points; ++q) 
          {
            phi_tracer.submit_dof_value(evaluate_function_tracer<dim, Number, n_tra>(
                                                    function,
                                                    phi_tracer.quadrature_point(q),
                                                    phi_tracer.get_value(q)),
                                 q);
          }
        inverse_tracer.transform_from_q_points_to_basis(n_tra,
                                                 phi_tracer.begin_dof_values(),
                                                 phi_tracer.begin_dof_values());
        phi_tracer.set_dof_values(solution_tracer);
      }
  }

  // The next function again repeats functionality also provided by the
  // deal.II library, namely VectorTools::integrate_difference(). We here show
  // the explicit code to highlight how the vectorization across several cells
  // works and how to accumulate results via that interface: Recall that each
  // <i>lane</i> of the vectorized array holds data from a different cell. By
  // the loop over all cell batches that are owned by the current MPI process,
  // we could then fill a VectorizedArray of results; to obtain a global sum,
  // we would need to further go on and sum across the entries in the SIMD
  // array. However, such a procedure is not stable as the SIMD array could in
  // fact not hold valid data for all its lanes. This happens when the number
  // of locally owned cells is not a multiple of the SIMD width. To avoid
  // invalid data, we must explicitly skip those invalid lanes when accessing
  // the data. While one could imagine that we could make it work by simply
  // setting the empty lanes to zero (and thus, not contribute to a sum), the
  // situation is more complicated than that: What if we were to compute a
  // velocity out of the momentum? Then, we would need to divide by the
  // density, which is zero -- the result would consequently be NaN and
  // contaminate the result. This trap is avoided by accumulating the results
  // from the valid SIMD range as we loop through the cell batches, using the
  // function MatrixFree::n_active_entries_per_cell_batch() to give us the
  // number of lanes with valid data. It equals VectorizedArray::size() on
  // most cells, but can be less on the last cell batch if the number of cells
  // has a remainder compared to the SIMD width.
  // 
  // Pay also attention to the implementation of the error formula. The error has
  // dimension `n_vars-(dim-1)` because we compute only one error for all the
  // momentum components. Also, the formula implies that the variables are stored
  // in the following order $(h,\, hu^i,\, T^j)$ with $i=1,...dim$ and $j=0,...N_T$
  // the number of tracers. First the error we compute the error on the depth,
  // the momentum error and finally we loop over the tracers. The number of tracers
  // is recovered from the number of variables and the problem dimension `n_vars-dim-1`.
  template <int dim, int n_tra, int degree, int n_points_1d>
  std::array<double, n_tra> OceanoOperatorWithTracer<dim, n_tra, degree, n_points_1d>::compute_errors_tracers(
    const Function<dim> &                             function,
    const LinearAlgebra::distributed::Vector<Number> &solution_tracer) const
  {
    TimerOutput::Scope t(timer, "compute errors");
    const unsigned int n_err = n_tra;
    double             errors_squared[n_err] = {};
    FEEvaluation<dim, degree, n_points_1d, n_tra, Number> phi_tracer(data, 2, 0);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        VectorizedArray<Number> local_errors_squared[n_err] = {};
        phi_tracer.reinit(cell);
        phi_tracer.gather_evaluate(solution_tracer, EvaluationFlags::values);
        for (unsigned int q = 0; q < phi_tracer.n_q_points; ++q)
          {
            const auto error = evaluate_function_tracer<dim, Number, n_tra>(
            		         function, 
                                 phi_tracer.quadrature_point(q),
                                 phi_tracer.get_value(q))
              		     - phi_tracer.get_value(q);
            const auto JxW = phi_tracer.JxW(q);
            for (unsigned int t = 0; t < n_tra; ++t)
              local_errors_squared[t] += (error[t] * error[t]) * JxW;
          }

        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
             ++v)
          for (unsigned int d = 0; d < n_err; ++d)
            errors_squared[d] += local_errors_squared[d][v];
      }

    Utilities::MPI::sum(errors_squared, MPI_COMM_WORLD, errors_squared);

    std::array<double, n_err> errors;
    for (unsigned int d = 0; d < n_err; ++d)
      errors[d] = std::sqrt(errors_squared[d]);

    return errors;
  }

  // For the main evaluation of the field variables, we first check that the
  // lengths of the arrays equal the expected values (namely the length of the
  // postprocessed variable vector equal the solution vector).
  // Then we loop over all evaluation points and
  // fill the respective information. First we fill the primal solution
  // variables, the so called prognostic variables. Then we compute derived variables,
  // the velocity $\mathbf u$ or the depth $h$. These variables are defined in
  // the model class and they can change from model to model.
  // For now these variables can depend only on the solution and on given data. For the
  // implementation of output variables depending on given data see the deal.ii documentation:
  // https://www.dealii.org/current/doxygen/deal.II/classDataPostprocessorVector.html
  // In general the output can also depend on the solution gradient (think for example to the
  // vorticity) but this part has been commented for now, see `do_schlieren_plot`.
  // For the postprocessed variables, a well defined order must be followed: first the
  // velocity vector, then all the scalars.
  //
  // The loop over the evaluation points seems not vectorized.
  // However the usual way to recover data is with
  // the `evaluate_function` that operates on vectorized data. With a few tricks the
  // the `evaluate_function` has been adapted to a non-vectorized loop.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorWithTracer<dim, n_tra, degree, n_points_1d>::evaluate_vector_field(
    const LinearAlgebra::distributed::Vector<Number>        &solution_height,
    const LinearAlgebra::distributed::Vector<Number>        &solution_discharge,
    const LinearAlgebra::distributed::Vector<Number>        &solution_tracer,
    const std::vector<Point<dim>>                           &evaluation_points,
    LinearAlgebra::distributed::Vector<Number>              &computed_vector_quantities,
    std::vector<LinearAlgebra::distributed::Vector<Number>> &computed_scalar_quantities) const
  {
    const unsigned int n_evaluation_points = solution_height.size();
    const std::vector<std::string> postproc_names = model.postproc_vars_name;

    //if (do_schlieren_plot == true)
    //  Assert(inputs.solution_gradients.size() == n_evaluation_points,
    //         ExcInternalError());
    Assert(computed_vector_quantities.size() == n_evaluation_points*dim,
           ExcInternalError());
    Assert(computed_scalar_quantities.size() == postproc_names.size()-dim,
           ExcInternalError());

    for (unsigned int p = 0; p < n_evaluation_points; ++p)
      {
        const auto height = solution_height[p];
        Tensor<1, dim> discharge;
        for (unsigned int d = 0; d < dim; ++d)
          discharge[d] = solution_discharge[dim*p+d];
        Tensor<1, n_tra> tracer;
        for (unsigned int t = 0; t < n_tra; ++t)
          tracer[t] = solution_tracer[n_tra*p+t];

        Point<dim, VectorizedArray<Number>> x_evaluation_points;
        for (unsigned int d = 0; d < dim; ++d)
          x_evaluation_points[d] = evaluation_points[p][d];

        const auto data = evaluate_function<dim, Number>(
            *bc->problem_data, x_evaluation_points, 0);

        const Tensor<1, dim> velocity = model.velocity<dim>(height, discharge, data[0]);

        for (unsigned int d = 0; d < dim; ++d)
          computed_vector_quantities[dim*p+d] = velocity[d];

        for (unsigned int v = 0; v < postproc_names.size()-dim; ++v)
          {
            if (postproc_names[v+dim] == "pressure")
              computed_scalar_quantities[v][p] =
                model.pressure<dim>(height, data[0]);
            else if (postproc_names[v+dim] == "depth")
              computed_scalar_quantities[v][p] = height + data[0];
            else if (postproc_names[v+dim] == "speed_of_sound")
              computed_scalar_quantities[v][p] =
                std::sqrt(model.square_wavespeed<dim>(height, data[0]));
            else if (postproc_names[v+dim] == "tracer")
              for (unsigned int t = 0; t < n_tra; ++t)
                computed_scalar_quantities[v][p] =
                  model.tracer<n_tra>(height, tracer, data[0])[t];
            else
              {
                std::cout << "Postprocessing variable " << postproc_names[dim+v]
                          << " does not exist. Consider to code it in your model"
                          << std::endl;
                 Assert(false, ExcNotImplemented());
              }
          }

        //if (do_schlieren_plot == true)
        //  computed_quantities[p](n_vars) =
        //    inputs.solution_gradients[p][0] * inputs.solution_gradients[p][0];
      }
  }

} // namespace SpaceDiscretization

#endif //OCEANDGWITHTRACER_HPP
