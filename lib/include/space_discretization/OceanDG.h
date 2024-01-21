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
#ifndef EULERDG_HPP
#define EULERDG_HPP 
 
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iomanip>
#include <iostream>

// The following file includes the CellwiseInverseMassMatrix data structure
// that we will use for the mass matrix inversion, the only new include
// file for this tutorial program:
#include <deal.II/matrix_free/operators.h>

// The following files include the oceano libraries
#if defined MODEL_EULER
#include <model/Euler.h>
#elif defined MODEL_SHALLOWWATER
#include <model/ShallowWater.h>
#endif
#include <numerical_flux/LaxFriedrichsModified.h>
#include <numerical_flux/HartenVanLeer.h>
#include <icbc/IcbcBase.h>

/**
 * Namespace containing the spatial Operator
 */

namespace SpaceDiscretization
{

  using namespace dealii;

  using Number = double;

  // This and the next function are helper functions to provide compact
  // evaluation calls as multiple points get batched together via a
  // VectorizedArray argument (see the step-37 tutorial for details). This
  // function is used for the subsonic outflow boundary conditions where we
  // need to set the energy component to a prescribed value. The next one
  // requests the solution on all components and is used for inflow boundaries
  // where all components of the solution are set.
  template <int dim, typename Number>
  VectorizedArray<Number>
  evaluate_function(const Function<dim> &                      function,
                    const Point<dim, VectorizedArray<Number>> &p_vectorized,
                    const unsigned int                         component)
  {
    VectorizedArray<Number> result;
    for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
      {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        result[v] = function.value(p, component);
      }
    return result;
  }


  template <int dim, typename Number, int n_vars>
  Tensor<1, n_vars, VectorizedArray<Number>>
  evaluate_function(const Function<dim> &                      function,
                    const Point<dim, VectorizedArray<Number>> &p_vectorized)
  {
    AssertDimension(function.n_components, n_vars);
    Tensor<1, n_vars, VectorizedArray<Number>> result;
    for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
      {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        for (unsigned int d = 0; d < n_vars; ++d)
          result[d][v] = function.value(p, d);
      }
    return result;
  }
  


  // @sect3{The OceanoOperation class}

  // This class implements the evaluators for the ocean problem, in analogy to
  // the `LaplaceOperator` class of step-37 or step-59. Since the present
  // operator is non-linear and does not require a matrix interface (to be
  // handed over to preconditioners), we skip the various `vmult` functions
  // otherwise present in matrix-free operators and only implement an `apply`
  // function as well as the combination of `apply` with the required vector
  // updates for the low-storage Runge--Kutta time integrator mentioned above
  // (called `perform_stage`). Furthermore, we have added three additional
  // functions involving matrix-free routines, namely one to compute an
  // estimate of the time step scaling (that is combined with the Courant
  // number for the actual time step size) based on the velocity and speed of
  // sound in the elements, one for the projection of solutions (specializing
  // VectorTools::project() for the DG case), and one to compute the errors
  // against a possible analytical solution or norms against some background
  // state.
  //
  // The rest of the class is similar to other matrix-free tutorials. As
  // discussed in the introduction, we provide a few functions to allow a user
  // to pass in various forms of boundary conditions on different parts of the
  // domain boundary marked by types::boundary_id variables, as well as
  // possible body forces.
  template <int dim, int n_vars, int degree, int n_points_1d>
  class OceanoOperator
  {
  public:
    static constexpr unsigned int n_quadrature_points_1d = n_points_1d;

    OceanoOperator(IO::ParameterHandler      &param,
                  ICBC::BcBase<dim, n_vars> *bc,
                  TimerOutput               &timer_output);

    void reinit(const Mapping<dim> &   mapping,
                const DoFHandler<dim> &dof_handler);

    void apply(const double                                      current_time,
               const LinearAlgebra::distributed::Vector<Number> &src,
               LinearAlgebra::distributed::Vector<Number> &      dst) const;

    void
    perform_stage(const Number cur_time,
                  const Number factor_solution,
                  const Number factor_ai,
                  const LinearAlgebra::distributed::Vector<Number> &current_ri,
                  LinearAlgebra::distributed::Vector<Number> &      vec_ki,
                  LinearAlgebra::distributed::Vector<Number> &      solution,
                  LinearAlgebra::distributed::Vector<Number> &next_ri) const;

    void project(const Function<dim> &                       function,
                 LinearAlgebra::distributed::Vector<Number> &solution) const;

    std::array<double, n_vars-dim+1> compute_errors(
      const Function<dim> &                             function,
      const LinearAlgebra::distributed::Vector<Number> &solution) const;

    double compute_cell_transport_speed(
      const LinearAlgebra::distributed::Vector<Number> &solution) const;

    void
    initialize_vector(LinearAlgebra::distributed::Vector<Number> &vector) const;

    ICBC::BcBase<dim, n_vars> *bc;
    
  private:
    MatrixFree<dim, Number> data;

    TimerOutput &timer;

    // The switch between the different model is realized with
    // Preprocessor keys. As already explained we have avoided pointers to 
    // interface classes. The Euler and the Shallow Water class must expose
    // the same interfaces with identical member functions.
#if defined MODEL_EULER
    Model::Euler model;
#elif defined MODEL_SHALLOWWATER
    Model::ShallowWater model;
#else
    Assert(false, ExcNotImplemented());
    return 0.;
#endif

    // Similarly the switch between the different numerical flux:
#if defined NUMERICALFLUX_LAXFRIEDRICHSMODIFIED
    NumericalFlux::LaxFriedrichsModified num_flux;
#elif defined NUMERICALFLUX_HARTENVANLEER
    NumericalFlux::HartenVanLeer         num_flux;
#else
    Assert(false, ExcNotImplemented());
    return 0.;
#endif

    void local_apply_inverse_mass_matrix(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    void local_apply_cell(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    void local_apply_face(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     face_range) const;

    void local_apply_boundary_face(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     face_range) const;
  };



  template <int dim, int n_vars, int degree, int n_points_1d>
  OceanoOperator<dim, n_vars, degree, n_points_1d>::OceanoOperator(
    IO::ParameterHandler             &param,
    ICBC::BcBase<dim, n_vars>        *bc,
    TimerOutput                      &timer)
    : bc(bc)
    , timer(timer)
    , model(param)
    , num_flux(param)
  {}



  // For the initialization of the ocean operator, we set up the MatrixFree
  // variable contained in the class. This can be done given a mapping to
  // describe possible curved boundaries as well as a DoFHandler object
  // describing the degrees of freedom. Since we use a discontinuous Galerkin
  // discretization in this tutorial program where no constraints are imposed
  // strongly on the solution field, we do not need to pass in an
  // AffineConstraints object and rather use a dummy for the
  // construction. With respect to quadrature, we want to select two different
  // ways of computing the underlying integrals: The first is a flexible one,
  // based on a template parameter `n_points_1d` (that will be assigned the
  // `n_points_1d` value specified at the top of this file). More accurate
  // integration is necessary to avoid the aliasing problem due to the
  // variable coefficients in the ocean operator. The second less accurate
  // quadrature formula is a tight one based on `degree+1` and needed for
  // the inverse mass matrix. While that formula provides an exact inverse
  // only on affine element shapes and not on deformed elements, it enables
  // the fast inversion of the mass matrix by tensor product techniques,
  // necessary to ensure optimal computational efficiency overall.
  template <int dim, int n_vars, int degree, int n_points_1d>
  void OceanoOperator<dim, n_vars, degree, n_points_1d>::reinit(
    const Mapping<dim> &   mapping,
    const DoFHandler<dim> &dof_handler)
  {
    const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler};
    const AffineConstraints<double>            dummy;
    const std::vector<const AffineConstraints<double> *> constraints = {&dummy};
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

    data.reinit(
      mapping, dof_handlers, constraints, quadratures, additional_data);
  }



  template <int dim, int n_vars, int degree, int n_points_1d>
  void OceanoOperator<dim, n_vars, degree, n_points_1d>::initialize_vector(
    LinearAlgebra::distributed::Vector<Number> &vector) const
  {
    data.initialize_dof_vector(vector);
  }



  // @sect4{Local evaluators}

  // Now we proceed to the local evaluators for the ocean problem. The
  // evaluators are relatively simple and follow what has been presented in
  // step-37, step-48, or step-59. The first notable difference is the fact
  // that we use an FEEvaluation with a non-standard number of quadrature
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
  // multi-component case. The variant shown here utilizes an FEEvaluation
  // object with multiple components embedded into it, specified by the fourth
  // template argument `n_vars` for the components in the shallow water system. As a
  // consequence, the return type of FEEvaluation::get_value() is not a scalar
  // any more (that would return a VectorizedArray type, collecting data from
  // several elements), but a Tensor of `n_vars` components. The functionality
  // is otherwise similar to the scalar case; it is handled by a template
  // specialization of a base class, called FEEvaluationAccess. An alternative
  // variant would have been to use several FEEvaluation objects, a scalar one
  // for the density, a vector-valued one with `dim` components for the
  // momentum, and another scalar evaluator for the energy. To ensure that
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
  // function, all we have to do here is to call this function
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
  //
  // As a final note, we mention that we do not use the first MatrixFree
  // argument of this function, which is a call-back from MatrixFree::loop().
  // The interfaces imposes the present list of arguments, but since we are in
  // a member function where the MatrixFree object is already available as the
  // `data` variable, we stick with that to avoid confusion.
  template <int dim, int n_vars, int degree, int n_points_1d>
  void OceanoOperator<dim, n_vars, degree, n_points_1d>::local_apply_cell(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, degree, n_points_1d, n_vars, Number> phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values
                              | EvaluationFlags::gradients);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto w_q = phi.get_value(q);
            const auto dzeta_q = phi.get_gradient(q);

            phi.submit_gradient(model.flux<dim, n_vars>(w_q), q);

//            std::cout << "Press grad " << q << ", " << dzeta_q[0] << std::endl;
#if defined MODEL_EULER
            const Tensor<1, dim, VectorizedArray<Number>> body_force_q =
              evaluate_function<dim, Number, dim>(
                *bc->body_force, phi.quadrature_point(q));
#endif
            phi.submit_value(model.source<dim, n_vars>(
              w_q,
#if defined MODEL_EULER
              body_force_q
#elif defined MODEL_SHALLOWWATER
              dzeta_q[0]
#endif
              ), q);
          }

        phi.integrate_scatter(((bc->body_force.get() != nullptr) ?
                                 EvaluationFlags::values :
                                 EvaluationFlags::nothing) |
                                EvaluationFlags::gradients,
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
  // free-standing function for the numerical flux. It receives the solution
  // evaluated at quadrature points from both sides (i.e., $\mathbf{w}^-$ and
  // $\mathbf{w}^+$), as well as the normal vector onto the minus side. As
  // explained above, the numerical flux is already multiplied by the normal
  // vector from the minus side. We need to switch the sign because the
  // boundary term comes with a minus sign in the weak form derived in the
  // introduction. The flux is then queued for testing both on the minus sign
  // and on the plus sign, with switched sign as the normal vector from the
  // plus side is exactly opposed to the one from the minus side.
  template <int dim, int n_vars, int degree, int n_points_1d>
  void OceanoOperator<dim, n_vars, degree, n_points_1d>::local_apply_face(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    FEFaceEvaluation<dim, degree, n_points_1d, n_vars, Number> phi_m(data,
                                                                      true);
    FEFaceEvaluation<dim, degree, n_points_1d, n_vars, Number> phi_p(data,
                                                                      false);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, EvaluationFlags::values);

        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
          {
            const auto numerical_flux =
              num_flux.euler_numerical_flux<dim, n_vars>(phi_m.get_value(q),
                                                         phi_p.get_value(q),
                                                         phi_m.normal_vector(q));
            phi_m.submit_value(-numerical_flux, q);
            phi_p.submit_value(numerical_flux, q);
          }

        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  // For faces located at the boundary, we need to impose the appropriate
  // boundary conditions. In this tutorial program, we implement four cases as
  // mentioned above. (A fifth case, for supersonic outflow conditions is
  // discussed in the "Results" section below.) The discontinuous Galerkin
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
  // setting $\mathbf{w}^+ = \mathbf{w}^-$. For the case of subsonic outflow,
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
  // negative. As we work on data of several quadrature points at once for
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
  template <int dim, int n_vars, int degree, int n_points_1d>
  void OceanoOperator<dim, n_vars, degree, n_points_1d>::local_apply_boundary_face(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    FEFaceEvaluation<dim, degree, n_points_1d, n_vars, Number> phi(data, true);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto w_m    = phi.get_value(q);
            const auto normal = phi.normal_vector(q);

            auto rho_u_dot_n = w_m[1] * normal[0];
            for (unsigned int d = 1; d < dim; ++d)
              rho_u_dot_n += w_m[1 + d] * normal[d];

            bool at_outflow = false;

            Tensor<1, n_vars, VectorizedArray<Number>> w_p;
            const auto boundary_id = data.get_boundary_id(face);
            if (bc->wall_boundaries.find(boundary_id) != bc->wall_boundaries.end())
              {
                w_p[0] = w_m[0];
                for (unsigned int d = 0; d < dim; ++d)
                  w_p[d + 1] = w_m[d + 1] - 2. * rho_u_dot_n * normal[d];
                for (unsigned int t = 0; t < n_vars-dim-1; ++t)
                  w_p[dim + 1 + t] = w_m[dim + 1 + t];
              }
            else if (bc->inflow_boundaries.find(boundary_id) !=
                     bc->inflow_boundaries.end())
              w_p =
                evaluate_function<dim, Number, n_vars>(
                		  *bc->inflow_boundaries.find(boundary_id)->second,
                                  phi.quadrature_point(q));
            else if (bc->subsonic_outflow_boundaries.find(boundary_id) !=
                     bc->subsonic_outflow_boundaries.end())
              {
                w_p          = w_m;
                w_p[dim + 1] = evaluate_function(
                  *bc->subsonic_outflow_boundaries.find(boundary_id)->second,
                  phi.quadrature_point(q),
                  dim + 1);
                at_outflow = true;
              }
            else
              AssertThrow(false,
                          ExcMessage("Unknown boundary id, did "
                                     "you set a boundary condition for "
                                     "this part of the domain boundary?"));

            auto flux = num_flux.euler_numerical_flux<dim, n_vars>(w_m, w_p, normal);

            if (at_outflow)
              for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
                {
                  if (rho_u_dot_n[v] < -1e-12)
                    for (unsigned int d = 0; d < dim; ++d)
                      flux[d + 1][v] = 0.;
                }

            phi.submit_value(-flux, q);
          }

        phi.integrate_scatter(EvaluationFlags::values, dst);
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
  template <int dim, int n_vars, int degree, int n_points_1d>
  void OceanoOperator<dim, n_vars, degree, n_points_1d>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, degree, degree + 1, n_vars, Number> phi(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, n_vars, Number>
      inverse(phi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);

        inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

        phi.set_dof_values(dst);
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
  template <int dim, int n_vars, int degree, int n_points_1d>
  void OceanoOperator<dim, n_vars, degree, n_points_1d>::apply(
    const double                                      current_time,
    const LinearAlgebra::distributed::Vector<Number> &src,
    LinearAlgebra::distributed::Vector<Number> &      dst) const
  {
    {
      TimerOutput::Scope t(timer, "apply - integrals");

      for (auto &i : bc->inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->subsonic_outflow_boundaries)
        i.second->set_time(current_time);

      data.loop(&OceanoOperator::local_apply_cell,
                &OceanoOperator::local_apply_face,
                &OceanoOperator::local_apply_boundary_face,
                this,
                dst,
                src,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

    {
      TimerOutput::Scope t(timer, "apply - inverse mass");

      data.cell_loop(&OceanoOperator::local_apply_inverse_mass_matrix,
                     this,
                     dst,
                     dst);
    }
  }



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
  template <int dim, int n_vars, int degree, int n_points_1d>
  void OceanoOperator<dim, n_vars, degree, n_points_1d>::perform_stage(
    const Number                                      current_time,
    const Number                                      factor_solution,
    const Number                                      factor_ai,
    const LinearAlgebra::distributed::Vector<Number> &current_ri,
    LinearAlgebra::distributed::Vector<Number> &      vec_ki,
    LinearAlgebra::distributed::Vector<Number> &      solution,
    LinearAlgebra::distributed::Vector<Number> &      next_ri) const
  {
    {
      TimerOutput::Scope t(timer, "rk_stage - integrals L_h");

      for (auto &i : bc->inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->subsonic_outflow_boundaries)
        i.second->set_time(current_time);

      data.loop(&OceanoOperator::local_apply_cell,
                &OceanoOperator::local_apply_face,
                &OceanoOperator::local_apply_boundary_face,
                this,
                vec_ki,
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }


    {
      TimerOutput::Scope t(timer, "rk_stage - inv mass + vec upd");
      data.cell_loop(
        &OceanoOperator::local_apply_inverse_mass_matrix,
        this,
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
  template <int dim, int n_vars, int degree, int n_points_1d>
  void OceanoOperator<dim, n_vars, degree, n_points_1d>::project(
    const Function<dim> &                       function,
    LinearAlgebra::distributed::Vector<Number> &solution) const
  {
    FEEvaluation<dim, degree, degree + 1, n_vars, Number> phi(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, n_vars, Number>
      inverse(phi);
    solution.zero_out_ghost_values();
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_dof_value(evaluate_function<dim, Number, n_vars>(
          					 function,
                                                 phi.quadrature_point(q)),
                               q);
        inverse.transform_from_q_points_to_basis(n_vars,
                                                 phi.begin_dof_values(),
                                                 phi.begin_dof_values());
        phi.set_dof_values(solution);
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
  template <int dim, int n_vars, int degree, int n_points_1d>
  std::array<double, n_vars-dim+1> OceanoOperator<dim, n_vars, degree, n_points_1d>::compute_errors(
    const Function<dim> &                             function,
    const LinearAlgebra::distributed::Vector<Number> &solution) const
  {
    TimerOutput::Scope t(timer, "compute errors");
    const unsigned int n_err = n_vars-dim+1;
    double             errors_squared[n_err] = {};
    FEEvaluation<dim, degree, n_points_1d, n_vars, Number> phi(data, 0, 0);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        VectorizedArray<Number> local_errors_squared[n_err] = {};
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto error =
              evaluate_function<dim, Number, n_vars>(
              			function, 
              			phi.quadrature_point(q)) -
              phi.get_value(q);
            const auto JxW = phi.JxW(q);

            local_errors_squared[0] += error[0] * error[0] * JxW;
            for (unsigned int d = 0; d < dim; ++d)
              local_errors_squared[1] += (error[d + 1] * error[d + 1]) * JxW;
            for (unsigned int t = 0; t < n_vars-dim-1; ++t)
              local_errors_squared[t+dim] += (error[t + dim + 1] * error[t + dim + 1]) * JxW;
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



  // This final function of the OceanoOperator class is used to estimate the
  // transport speed, scaled by the mesh size, that is relevant for setting
  // the time step size in the explicit time integrator. In the shallow water
  // equations, there are two speeds of transport, namely the convective
  // velocity $\mathbf{u}$ and the propagation of sound waves with sound
  // speed $c = \sqrt{\gamma p/\rho}$ relative to the medium moving at
  // velocity $\mathbf u$.
  //
  // In the formula for the time step size, we are interested not by
  // these absolute speeds, but by the amount of time it takes for
  // information to cross a single cell. For information transported along with
  // the medium, $\mathbf u$ is scaled by the mesh size,
  // so an estimate of the maximal velocity can be obtained by computing
  // $\|J^{-\mathrm T} \mathbf{u}\|_\infty$, where $J$ is the Jacobian of the
  // transformation from real to the reference domain. Note that
  // FEEvaluationBase::inverse_jacobian() returns the inverse and transpose
  // Jacobian, representing the metric term from real to reference
  // coordinates, so we do not need to transpose it again. We store this limit
  // in the variable `convective_limit` in the code below.
  //
  // The sound propagation is isotropic, so we need to take mesh sizes in any
  // direction into account. The appropriate mesh size scaling is then given
  // by the minimal singular value of $J$ or, equivalently, the maximal
  // singular value of $J^{-1}$. Note that one could approximate this quantity
  // by the minimal distance between vertices of a cell when ignoring curved
  // cells. To get the maximal singular value of the Jacobian, the general
  // strategy would be some LAPACK function. Since all we need here is an
  // estimate, we can avoid the hassle of decomposing a tensor of
  // VectorizedArray numbers into several matrices and go into an (expensive)
  // eigenvalue function without vectorization, and instead use a few
  // iterations (five in the code below) of the power method applied to
  // $J^{-1}J^{-\mathrm T}$. The speed of convergence of this method depends
  // on the ratio of the largest to the next largest eigenvalue and the
  // initial guess, which is the vector of all ones. This might suggest that
  // we get slow convergence on cells close to a cube shape where all
  // lengths are almost the same. However, this slow convergence means that
  // the result will sit between the two largest singular values, which both
  // are close to the maximal value anyway. In all other cases, convergence
  // will be quick. Thus, we can merely hardcode 5 iterations here and be
  // confident that the result is good.
  template <int dim, int n_vars, int degree, int n_points_1d>
  double OceanoOperator<dim, n_vars, degree, n_points_1d>::compute_cell_transport_speed(
    const LinearAlgebra::distributed::Vector<Number> &solution) const
  {
    TimerOutput::Scope t(timer, "compute transport speed");
    Number             max_transport = 0;
    FEEvaluation<dim, degree, degree + 1, n_vars, Number> phi(data, 0, 1);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        VectorizedArray<Number> local_max = 0.;
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto solution = phi.get_value(q);
            const auto velocity = model.velocity<dim, n_vars>(solution);

            const auto inverse_jacobian = phi.inverse_jacobian(q);
            const auto convective_speed = inverse_jacobian * velocity;
            VectorizedArray<Number> convective_limit = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              convective_limit =
                std::max(convective_limit, std::abs(convective_speed[d]));

            const auto speed_of_sound =
              std::sqrt(model.square_wavespeed<dim, n_vars>(solution));

            Tensor<1, dim, VectorizedArray<Number>> eigenvector;
            for (unsigned int d = 0; d < dim; ++d)
              eigenvector[d] = 1.;
            for (unsigned int i = 0; i < 5; ++i)
              {
                eigenvector = transpose(inverse_jacobian) *
                              (inverse_jacobian * eigenvector);
                VectorizedArray<Number> eigenvector_norm = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                  eigenvector_norm =
                    std::max(eigenvector_norm, std::abs(eigenvector[d]));
                eigenvector /= eigenvector_norm;
              }
            const auto jac_times_ev   = inverse_jacobian * eigenvector;
            const auto max_eigenvalue = std::sqrt(
              (jac_times_ev * jac_times_ev) / (eigenvector * eigenvector));
            local_max =
              std::max(local_max,
                       max_eigenvalue * speed_of_sound + convective_limit);
          }

        // Similarly to the previous function, we must make sure to accumulate
        // speed only on the valid cells of a cell batch.
        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
             ++v)
          for (unsigned int d = 0; d < 3; ++d)
            max_transport = std::max(max_transport, local_max[v]);
      }

    max_transport = Utilities::MPI::max(max_transport, MPI_COMM_WORLD);

    return max_transport;
  }

} // namespace SpaceDiscretization

#endif //EULERDG_HPP
