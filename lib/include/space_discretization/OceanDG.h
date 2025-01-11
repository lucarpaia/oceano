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
#ifndef OCEANDG_HPP
#define OCEANDG_HPP
 
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
#elif defined MODEL_SHALLOWWATERWITHTRACER
#include <model/ShallowWaterWithTracer.h>
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

  // This and the next functions are helper functions to provide compact
  // evaluation calls as multiple points get batched together via a
  // VectorizedArray argument (see the step-37 tutorial for details). This
  // functions are used for inquiry the data for the source terms and the
  // initial/boundary conditions. The first one request a single data
  // component. The second one requests the solution on all
  // components and it is used for supercritical inflow flow boundaries
  // where all components of the solution are set.
  template <int dim, typename Number>
  VectorizedArray<Number>
  evaluate_function(const Function<dim>                       &function,
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
  evaluate_function(const Function<dim>                       &function,
                    const Point<dim, VectorizedArray<Number>> &p_vectorized)
  {
    Assert(function.n_components >= n_vars, ExcInternalError());
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
  // updates for the Runge--Kutta time integrator mentioned above
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
  template <int dim, int n_tra, int degree, int n_points_1d>
  class OceanoOperator
  {
  public:
    static constexpr unsigned int n_quadrature_points_1d = n_points_1d;

    double factor_matrix;

    OceanoOperator(IO::ParameterHandler      &param,
                  ICBC::BcBase<dim, 1+dim+n_tra> *bc,
                  TimerOutput               &timer_output);

    void reinit(const Mapping<dim> &   mapping,
                const DoFHandler<dim> &dof_handler_height,
                const DoFHandler<dim> &dof_handler_discharge,
                const DoFHandler<dim> &dof_handler_tracer);

    void apply(const double                                      current_time,
               const LinearAlgebra::distributed::Vector<Number> &src,
               LinearAlgebra::distributed::Vector<Number> &      dst) const;

#if defined TIMEINTEGRATOR_EXPLICITRUNGEKUTTA
    void
    perform_stage_hydro(
      const unsigned int                                             cur_stage,
      const Number                                                   cur_time,
      const Number                                                  *factor_residual,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &current_ri,
      std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_height,
      std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_discharge,
      LinearAlgebra::distributed::Vector<Number>                    &solution_height,
      LinearAlgebra::distributed::Vector<Number>                    &solution_discharge,
      LinearAlgebra::distributed::Vector<Number>                    &next_ri_height,
      LinearAlgebra::distributed::Vector<Number>                    &next_ri_discharge) const;
#elif defined TIMEINTEGRATOR_ADDITIVERUNGEKUTTA
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
#endif

    void project_hydro(const Function<dim> &                       function,
                       LinearAlgebra::distributed::Vector<Number> &solution_height,
                       LinearAlgebra::distributed::Vector<Number> &solution_discharge) const;

    std::array<double, 2> compute_errors_hydro(
      const Function<dim> &                             function,
      const LinearAlgebra::distributed::Vector<Number> &solution_height,
      const LinearAlgebra::distributed::Vector<Number> &solution_discharge) const;

    double compute_cell_transport_speed(
      const LinearAlgebra::distributed::Vector<Number> &solution_height,
      const LinearAlgebra::distributed::Vector<Number> &solution_discharge) const;

    void evaluate_vector_field(
      const DoFHandler<dim>                                   &dof_handler_height,
      const LinearAlgebra::distributed::Vector<Number>        &solution_height,
      const LinearAlgebra::distributed::Vector<Number>        &solution_discharge,
      std::map<unsigned int, Point<dim>>                      &evaluation_points,
      LinearAlgebra::distributed::Vector<Number>              &computed_vector_quantities,
      std::vector<LinearAlgebra::distributed::Vector<Number>> &computed_scalar_quantities) const;

    void
    initialize_vector(LinearAlgebra::distributed::Vector<Number> &vector,
                      const unsigned int                          variable) const;

    ICBC::BcBase<dim, 1+dim+n_tra> *bc;

    // The switch between the different models is realized with
    // Preprocessor keys. As already explained we have avoided pointers to 
    // interface classes. The Euler and the Shallow Water class must expose
    // the same interfaces with identical member functions. Note that the
    // model class is public beacause it must be accessed, during postprocessing
    // outside the OceanoOperator class
#if defined MODEL_EULER
    Model::Euler model;
#elif defined MODEL_SHALLOWWATER
    Model::ShallowWater model;
#elif defined MODEL_SHALLOWWATERWITHTRACER
    Model::ShallowWaterWithTracer model;
#else
    Assert(false, ExcNotImplemented());
    return 0.;
#endif

   protected:
    // Similarly the switch between the different numerical flux:
#if defined NUMERICALFLUX_LAXFRIEDRICHSMODIFIED
    NumericalFlux::LaxFriedrichsModified num_flux;
#elif defined NUMERICALFLUX_HARTENVANLEER
    NumericalFlux::HartenVanLeer         num_flux;
#else
    Assert(false, ExcNotImplemented());
    return 0.;
#endif

    MatrixFree<dim, Number> data;

    TimerOutput &timer;

  private:
    void local_apply_inverse_mass_matrix_height(
      const MatrixFree<dim, Number>                    &data,
      LinearAlgebra::distributed::Vector<Number>       &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int>      &cell_range) const;

    void local_apply_inverse_mass_matrix_discharge(
      const MatrixFree<dim, Number>                    &data,
      LinearAlgebra::distributed::Vector<Number>       &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int>      &cell_range) const;

    void local_apply_inverse_modified_mass_matrix_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;

    void local_apply_cell_height(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;

    void local_apply_cell_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;

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

    void local_apply_cell_mass_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const LinearAlgebra::distributed::Vector<Number>              &src,
      const std::pair<unsigned int, unsigned int>                   &cell_range) const;

    void local_apply_face_height(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;

    void local_apply_face_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;

    void local_apply_boundary_face_height(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;

    void local_apply_boundary_face_discharge(
      const MatrixFree<dim, Number>                                 &data,
      LinearAlgebra::distributed::Vector<Number>                    &dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
      const std::pair<unsigned int, unsigned int>                   &face_range) const;
  };



  // The constructor initialize the fundamental classes of the Ocean Operator,
  // namely the boundary conditions, the model and the numerical fluxes.
  // Beside we set the name of the model variables. This is a trick to avoid
  // initialization of such names into the model class constructor which is not
  // templated (`dim` and `n_tra` are not available there).
  template <int dim, int n_tra, int degree, int n_points_1d>
  OceanoOperator<dim, n_tra, degree, n_points_1d>::OceanoOperator(
    IO::ParameterHandler             &param,
    ICBC::BcBase<dim, 1+dim+n_tra>   *bc,
    TimerOutput                      &timer)
    : bc(bc)
    , model(param)
    , num_flux(param)
    , timer(timer)
  {
     model.set_vars_name<dim, n_tra>();
  }



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
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::reinit(
    const Mapping<dim> &   mapping,
    const DoFHandler<dim> &dof_handler_height,
    const DoFHandler<dim> &dof_handler_discharge,
    const DoFHandler<dim> &/*dof_handler_tracer*/)
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

    data.reinit(
      mapping, dof_handlers, constraints, quadratures, additional_data);
  }



  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::initialize_vector(
    LinearAlgebra::distributed::Vector<Number> &vector,
    const unsigned int                          variable) const
  {
    data.initialize_dof_vector(vector, variable);
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
  // object with multiple components embedded into it, specified by an additional
  // template argument `n_vars` for the components in the shallow water system.
  // The alternative variant followed here uses several FEEvaluation objects, 
  // a scalar one for the height and a vector-valued one with `dim` components for the
  // momentum.
  // To ensure that those components point to the correct part of the solution, the
  // constructor of FEEvaluation takes three optional integer arguments after
  // the required MatrixFree field, namely the number of the DoFHandler for
  // multi-DoFHandler systems (taking the first by default), the number of the
  // quadrature point in case there are multiple Quadrature objects (see more
  // below), and as a third argument the component within a vector system. As
  // we have a single vector for all components, we would go with the third
  // argument, and set it to `0` for the density, `1` for the vector-valued
  // momentum. FEEvaluation then picks the
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
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::local_apply_cell_height(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, degree, n_points_1d, 1, Number> phi_height(data,0);
    FEEvaluation<dim, degree, n_points_1d, dim, Number> phi_discharge(data,1);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_height.reinit(cell);
        phi_discharge.reinit(cell);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_height.n_q_points; ++q)
          {
            const auto q_q = phi_discharge.get_value(q);
            phi_height.submit_gradient(model.massflux<dim>(q_q), q);
          }

        phi_height.integrate_scatter(EvaluationFlags::gradients,
                                       dst);
      }
  }

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::local_apply_cell_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, degree, n_points_1d, 1, Number> phi_height(data,0);
    FEEvaluation<dim, degree, n_points_1d, dim, Number> phi_discharge(data,1);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_height.reinit(cell);
        phi_height.gather_evaluate(src[0], EvaluationFlags::values
                              | EvaluationFlags::gradients);
        phi_discharge.reinit(cell);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
          {
            const auto q_q = phi_discharge.get_value(q);
            const auto z_q = phi_height.get_value(q);
            const auto dz_q = phi_height.get_gradient(q);

            Tensor<1, dim+3, VectorizedArray<Number>> data_q =
              evaluate_function<dim, Number, dim+3>(
                *bc->problem_data, phi_discharge.quadrature_point(q));

            phi_discharge.submit_gradient(
              model.advectiveflux<dim>(z_q, q_q, data_q[0]), q);

            phi_discharge.submit_value(
              model.source<dim>(z_q, q_q, dz_q, data_q),
              q);
          }

        phi_discharge.integrate_scatter(EvaluationFlags::values |
                                        EvaluationFlags::gradients,
                                       dst);
      }
  }

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::local_apply_cell_nonstiff_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, degree, n_points_1d, 1, Number> phi_height(data,0);
    FEEvaluation<dim, degree, n_points_1d, dim, Number> phi_discharge(data,1);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_height.reinit(cell);
        phi_height.gather_evaluate(src[0], EvaluationFlags::values
                              | EvaluationFlags::gradients);
        phi_discharge.reinit(cell);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
          {
            const auto q_q = phi_discharge.get_value(q);
            const auto z_q = phi_height.get_value(q);
            const auto dz_q = phi_height.get_gradient(q);

            Tensor<1, dim+3, VectorizedArray<Number>> data_q =
              evaluate_function<dim, Number, dim+3>(
                *bc->problem_data, phi_discharge.quadrature_point(q));

            phi_discharge.submit_gradient(
              model.advectiveflux<dim>(z_q, q_q, data_q[0]), q);

            phi_discharge.submit_value(
              model.source_nonstiff<dim>(z_q, q_q, dz_q, data_q),
              q);
          }

        phi_discharge.integrate_scatter(EvaluationFlags::values |
                                        EvaluationFlags::gradients,
                                       dst);
      }
  }

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::local_apply_cell_stiff_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, degree, n_points_1d, 1, Number> phi_height(data,0);
    FEEvaluation<dim, degree, n_points_1d, dim, Number> phi_discharge(data,1);

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

            Tensor<1, 2, VectorizedArray<Number>> data_q =
              evaluate_function<dim, Number, 2>(
                *bc->problem_data, phi_discharge.quadrature_point(q));

            phi_discharge.submit_value(
              model.source_stiff<dim>(z_q, q_q, data_q),
              q);
          }

        phi_discharge.integrate_scatter(EvaluationFlags::values,
                                       dst);
      }
  }

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::local_apply_cell_mass_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const LinearAlgebra::distributed::Vector<Number>              &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, degree, degree + 1, dim, Number> phi_discharge(data,1,1);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_discharge.reinit(cell);
        phi_discharge.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
          {
            const auto q_q = phi_discharge.get_value(q);
            phi_discharge.submit_value(q_q, q);
          }

        phi_discharge.integrate_scatter(EvaluationFlags::values,
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
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::local_apply_face_height(
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

        for (unsigned int q = 0; q < phi_height_m.n_q_points; ++q)
          {
            const VectorizedArray<Number> data_m =
              evaluate_function<dim, Number>(*bc->problem_data,
                phi_height_m.quadrature_point(q)-1e-12*phi_height_m.normal_vector(q), 0);
            const VectorizedArray<Number> data_p =
              evaluate_function<dim, Number>(*bc->problem_data,
                phi_height_m.quadrature_point(q)+1e-12*phi_height_m.normal_vector(q), 0);

            auto numerical_flux_p =
              num_flux.numerical_massflux_weak<dim>(phi_height_m.get_value(q),
                                                    phi_height_p.get_value(q),
                                                    phi_discharge_m.get_value(q),
                                                    phi_discharge_p.get_value(q),
                                                    phi_height_m.normal_vector(q),
                                                    data_m,
                                                    data_p);

            phi_height_m.submit_value(-numerical_flux_p, q);
            phi_height_p.submit_value(numerical_flux_p, q);
          }

        phi_height_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_height_m.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::local_apply_face_discharge(
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

            const VectorizedArray<Number> data_m =
              evaluate_function<dim, Number>(*bc->problem_data,
                phi_discharge_m.quadrature_point(q)-1e-12*phi_discharge_m.normal_vector(q), 0);
            const VectorizedArray<Number> data_p =
              evaluate_function<dim, Number>(*bc->problem_data,
                phi_discharge_m.quadrature_point(q)+1e-12*phi_discharge_m.normal_vector(q), 0);

            auto numerical_flux_p =
              num_flux.numerical_advflux_weak<dim>(z_m,
                                                   z_p,
                                                   phi_discharge_m.get_value(q),
                                                   phi_discharge_p.get_value(q),
                                                   normal,
                                                   data_m,
                                                   data_p);
            auto numerical_flux_m = -numerical_flux_p;

            numerical_flux_m -=
              num_flux.numerical_presflux_strong<dim>(z_m,
                                                      z_p,
                                                      normal,
                                                      data_m,
                                                      data_p);
            numerical_flux_p +=
              num_flux.numerical_presflux_strong<dim>(z_p,
                                                      z_m,
                                                      normal,
                                                      data_p,
                                                      data_m);

            phi_discharge_m.submit_value(numerical_flux_m, q);
            phi_discharge_p.submit_value(numerical_flux_p, q);
          }

        phi_discharge_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_discharge_m.integrate_scatter(EvaluationFlags::values, dst);
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
  // negative. As we work on data of several quadrature points at once for
  // SIMD vectorizations, we here need to explicitly loop over the array
  // entries of the SIMD array.
  //
  // The next boundary conditions are very important in coastal ocean and
  // hydraulic simulation for which the flow conditions are essentially
  // subcritical. The outflow condition seen sofar assumes that all the eigenvalues
  // are outgoing which cannot be true in the subcritical regime.
  // In subcritical regime we are sure only that one eigenvalue
  // is always ingoing, than it is quite safe to specify only
  // one boundary condition and the choice depends on the problem. It could be
  // the normal discharge (relevant for river inflows) or the water level height
  // (relevant for an open boundary with tide). Please note that in some
  // application it can be better to set also the tangential flow but we do not
  // treat this case here. Concerning the implementation, we use
  // the evaluate function by component to get the normal discharge.
  // Later it is projected along the boundary normal to get x and y
  // discharge components to be used in the Riemann solver.
  //
  // We have implemented an absorbing outflow boundary where we recover the
  // information coming from the ingoing eigenvalue. We compute a
  // boundary state from a far-field state (typically a flow at rest) from
  // the theory of characteristics, that is by equating at the boundary location,
  // the outgoing Riemann invariant with the ingoing one.
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
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::local_apply_boundary_face_height(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &face_range) const
  {
    FEFaceEvaluation<dim, degree, n_points_1d, 1, Number> phi_height(data, true, 0);
    FEFaceEvaluation<dim, degree, n_points_1d, dim, Number> phi_discharge(data, true, 1);

    const unsigned int n_vars = dim+1;

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_height.reinit(face);
        phi_height.gather_evaluate(src[0], EvaluationFlags::values);
        phi_discharge.reinit(face);
        phi_discharge.gather_evaluate(src[1], EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_height.n_q_points; ++q)
          {
            const auto z_m    = phi_height.get_value(q);
            const auto q_m    = phi_discharge.get_value(q);
            const auto normal = phi_height.normal_vector(q);
            const VectorizedArray<Number> data_m =
              evaluate_function<dim, Number>(
                *bc->problem_data, phi_height.quadrature_point(q), 0);

            auto rho_u_dot_n = q_m * normal;

            VectorizedArray<Number> z_p;
            Tensor<1, dim, VectorizedArray<Number>> q_p;
            Tensor<1, n_vars, VectorizedArray<Number>> w_p;
            const auto boundary_id = data.get_boundary_id(face);
            if (bc->wall_boundaries.find(boundary_id) != bc->wall_boundaries.end())
              {
                z_p = z_m;
                q_p = q_m - 2. * rho_u_dot_n * normal;
              }
            else if (bc->supercritical_inflow_boundaries.find(boundary_id) !=
                     bc->supercritical_inflow_boundaries.end())
              {
                w_p =
                  evaluate_function<dim, Number, n_vars>(
                    *bc->supercritical_inflow_boundaries.find(boundary_id)->second,
                    phi_height.quadrature_point(q));
                z_p = w_p[0];
                for (unsigned int d = 0; d < dim; ++d) q_p[d] = w_p[d+1];
              }
            else if (bc->supercritical_outflow_boundaries.find(boundary_id) !=
                     bc->supercritical_outflow_boundaries.end())
              {
                z_p = z_m;
                q_p = q_m;
              }
            else if (bc->height_inflow_boundaries.find(boundary_id) !=
                     bc->height_inflow_boundaries.end())
              {
                z_p =
                  evaluate_function<dim, Number>(
                    *bc->height_inflow_boundaries.find(boundary_id)->second,
                    phi_height.quadrature_point(q), 0);
                q_p = q_m;
              }
            else if (bc->discharge_inflow_boundaries.find(boundary_id) !=
                     bc->discharge_inflow_boundaries.end())
              {
                z_p = z_m;
                q_p =
                  evaluate_function<dim, Number>(
                    *bc->discharge_inflow_boundaries.find(boundary_id)->second,
                    phi_height.quadrature_point(q), 1) * -normal;
              }
            else if (bc->absorbing_outflow_boundaries.find(boundary_id) !=
                     bc->absorbing_outflow_boundaries.end())
              {
                w_p =
                  evaluate_function<dim, Number, n_vars>(
                    *bc->absorbing_outflow_boundaries.find(boundary_id)->second,
                      phi_height.quadrature_point(q));
                z_p = w_p[0];
                for (unsigned int d = 0; d < dim; ++d) q_p[d] = w_p[d+1];
                const auto r_p
                  = model.riemann_invariant_p<dim>(z_m, q_m, normal, data_m);
                const auto r_m
                  = model.riemann_invariant_m<dim>(z_p, q_p, normal, data_m);
                const auto c_b = 0.25 * (r_p - r_m);
                const auto h_b = c_b * c_b / model.g;
                const auto u_b = 0.5 * (r_p + r_m);

                z_p = h_b - data_m;
                q_p =  u_b * h_b * normal;
              }
            else
              AssertThrow(false,
                          ExcMessage("Unknown boundary id, did "
                                     "you set a boundary condition for "
                                     "this part of the domain boundary?"));

            auto flux =
              num_flux.numerical_massflux_weak<dim>(z_m, z_p, q_m, q_p,
                                                    normal, data_m, data_m);

            phi_height.submit_value(-flux, q);
          }

        phi_height.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::local_apply_boundary_face_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &face_range) const
  {
    FEFaceEvaluation<dim, degree, n_points_1d, 1, Number> phi_height(data, true, 0);
    FEFaceEvaluation<dim, degree, n_points_1d, dim, Number> phi_discharge(data, true, 1);

    const unsigned int n_vars = dim+1;

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
            const VectorizedArray<Number> data_m =
              evaluate_function<dim, Number>(
                *bc->problem_data, phi_discharge.quadrature_point(q), 0);

            auto rho_u_dot_n = q_m * normal;

            bool at_outflow = false;

            VectorizedArray<Number> z_p;
            Tensor<1, dim, VectorizedArray<Number>> q_p;
            Tensor<1, n_vars, VectorizedArray<Number>> w_p;
            const auto boundary_id = data.get_boundary_id(face);
            if (bc->wall_boundaries.find(boundary_id) != bc->wall_boundaries.end())
              {
                z_p = z_m;
                q_p = q_m - 2. * rho_u_dot_n * normal;
              }
            else if (bc->supercritical_inflow_boundaries.find(boundary_id) !=
                     bc->supercritical_inflow_boundaries.end())
              {
                w_p =
                  evaluate_function<dim, Number, n_vars>(
                    *bc->supercritical_inflow_boundaries.find(boundary_id)->second,
                    phi_discharge.quadrature_point(q));
                z_p = w_p[0];
                for (unsigned int d = 0; d < dim; ++d) q_p[d] = w_p[d+1];
              }
            else if (bc->supercritical_outflow_boundaries.find(boundary_id) !=
                     bc->supercritical_outflow_boundaries.end())
              {
                z_p = z_m;
                q_p = q_m;
                at_outflow = true;
              }
            else if (bc->height_inflow_boundaries.find(boundary_id) !=
                     bc->height_inflow_boundaries.end())
              {
                z_p =
                  evaluate_function<dim, Number>(
                    *bc->height_inflow_boundaries.find(boundary_id)->second,
                    phi_discharge.quadrature_point(q), 0);
                q_p = q_m;
              }
            else if (bc->discharge_inflow_boundaries.find(boundary_id) !=
                     bc->discharge_inflow_boundaries.end())
              {
                z_p = z_m;
                q_p =
                  evaluate_function<dim, Number>(
                    *bc->discharge_inflow_boundaries.find(boundary_id)->second,
                    phi_discharge.quadrature_point(q), 1) * -normal;
              }
            else if (bc->absorbing_outflow_boundaries.find(boundary_id) !=
                     bc->absorbing_outflow_boundaries.end())
              {
                w_p =
                  evaluate_function<dim, Number, n_vars>(
                    *bc->absorbing_outflow_boundaries.find(boundary_id)->second,
                      phi_discharge.quadrature_point(q));
                z_p = w_p[0];
                for (unsigned int d = 0; d < dim; ++d) q_p[d] = w_p[d+1];
                const auto r_p
                  = model.riemann_invariant_p<dim>(z_m, q_m, normal, data_m);
                const auto r_m
                  = model.riemann_invariant_m<dim>(z_p, q_p, normal, data_m);
                const auto c_b = 0.25 * (r_p - r_m);
                const auto h_b = c_b * c_b / model.g;
                const auto u_b = 0.5 * (r_p + r_m);

                z_p = h_b - data_m;
                q_p =  u_b * h_b * normal;
              }
            else
              AssertThrow(false,
                          ExcMessage("Unknown boundary id, did "
                                     "you set a boundary condition for "
                                     "this part of the domain boundary?"));

            auto flux =
              num_flux.numerical_advflux_weak<dim>(z_m, z_p, q_m, q_p, normal, data_m, data_m);

            auto pressure_numerical_fluxes =
              num_flux.numerical_presflux_strong<dim>(z_m, z_p, normal, data_m, data_m);
            flux += pressure_numerical_fluxes;

            if (at_outflow)
              for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
                {
                  if (rho_u_dot_n[v] < -1e-12)
                    for (unsigned int d = 0; d < dim; ++d)
                      flux[d][v] = 0.;
                }

            phi_discharge.submit_value(-flux, q);
          }

        phi_discharge.integrate_scatter(EvaluationFlags::values, dst);
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
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::local_apply_inverse_mass_matrix_height(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, degree, degree + 1, 1, Number> phi_height(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, 1, Number>
      inverse(phi_height);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_height.reinit(cell);
        phi_height.read_dof_values(src);

        inverse.apply(phi_height.begin_dof_values(), phi_height.begin_dof_values());

        phi_height.set_dof_values(dst);
      }
  }

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::local_apply_inverse_mass_matrix_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, degree, degree + 1, dim, Number> phi_discharge(data, 1, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim, Number>
      inverse(phi_discharge);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_discharge.reinit(cell);
        phi_discharge.read_dof_values(src);

        inverse.apply(phi_discharge.begin_dof_values(), phi_discharge.begin_dof_values());

        phi_discharge.set_dof_values(dst);
      }
  }

  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::local_apply_inverse_modified_mass_matrix_discharge(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number>                    &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
    const std::pair<unsigned int, unsigned int>                   &cell_range) const
  {
    FEEvaluation<dim, degree, degree + 1, dim, Number> phi_discharge(data, 1, 1);
    FEEvaluation<dim, degree, degree + 1, 1, Number> phi_height_ri(data, 0, 1);
    FEEvaluation<dim, degree, degree + 1, dim, Number> phi_discharge_ri(data, 1, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim, Number>
      inverse(phi_discharge);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi_discharge.reinit(cell);
        phi_discharge.read_dof_values(src[0]);

        phi_height_ri.reinit(cell);
        phi_height_ri.gather_evaluate(src[1], EvaluationFlags::values);
        phi_discharge_ri.reinit(cell);
        phi_discharge_ri.gather_evaluate(src[2], EvaluationFlags::values);

	AlignedVector<VectorizedArray<Number>> inverse_jxw(phi_discharge.n_q_points);
	inverse.fill_inverse_JxW_values(inverse_jxw);

        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
          {
            const VectorizedArray<Number> data_q =
              evaluate_function<dim, Number>(
                *bc->problem_data, phi_discharge.quadrature_point(q), 0);
            const VectorizedArray<Number> drag_q =
              evaluate_function<dim, Number>(
                *bc->problem_data, phi_discharge.quadrature_point(q), 1);
            const auto z_q = phi_height_ri.get_value(q);
            const auto q_q = phi_discharge_ri.get_value(q);

            inverse_jxw[q] *= 1. / ( 1. + factor_matrix
              * model.bottom_friction.jacobian<dim>(model.velocity<dim>(z_q, q_q, data_q),
                                                    drag_q,
                                                    z_q+data_q)
                                   );
          }

        inverse.apply(inverse_jxw, dim, phi_discharge.begin_dof_values(),
          phi_discharge.begin_dof_values());
//        inverse.apply(phi_discharge.begin_dof_values(), phi_discharge.begin_dof_values());

        phi_discharge.set_dof_values(dst);
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
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::apply(
    const double                                      current_time,
    const LinearAlgebra::distributed::Vector<Number> &src,
    LinearAlgebra::distributed::Vector<Number> &      dst) const
  {
    {
      TimerOutput::Scope t(timer, "apply - integrals");

      for (auto &i : bc->supercritical_inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->height_inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->discharge_inflow_boundaries)
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
  // to the vectors. Rather than performing these
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
  //
  // We code here the explicit Runge-Kutta method written
  // in the standard Butcher tableau form. This kind of method are generals then
  // the low storage ones, although less optimized. We update one single vector at
  // at every stage (`next_ri` for internal stages and `solution` for the last stage)
  // so we cannot benefit of loop fusion. Moreover we access to `n_stages` vector
  // (`n_stages-1` residual plus the solution) compared to only two vectors of the
  // low-storage scheme. For a low number of stage the difference is comparable.
  // We still use to perform the loop immediately when the mass matrix inversion has
  // finished on a part of the vector. The second `std::function` is in fact called
  // after the loop last touches an entry. A different code path is again used for
  // the last stage when we do not need to update the `next_ri` vector.
#if defined TIMEINTEGRATOR_EXPLICITRUNGEKUTTA
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::perform_stage_hydro(
    const unsigned int                                             current_stage,
    const Number                                                   current_time,
    const Number                                                  *factor_residual,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &current_ri,
    std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_height,
    std::vector<LinearAlgebra::distributed::Vector<Number>>       &vec_ki_discharge,
    LinearAlgebra::distributed::Vector<Number>                    &solution_height,
    LinearAlgebra::distributed::Vector<Number>                    &solution_discharge,
    LinearAlgebra::distributed::Vector<Number>                    &next_ri_height,
    LinearAlgebra::distributed::Vector<Number>                    &next_ri_discharge) const
  {
    {
      TimerOutput::Scope t(timer, "rk_stage hydro - integrals L_h");

      for (auto &i : bc->supercritical_inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->height_inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->discharge_inflow_boundaries)
        i.second->set_time(current_time);

      data.loop(&OceanoOperator::local_apply_cell_height,
                &OceanoOperator::local_apply_face_height,
                &OceanoOperator::local_apply_boundary_face_height,
                this,
                vec_ki_height.front(),
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);

      data.loop(&OceanoOperator::local_apply_cell_discharge,
                &OceanoOperator::local_apply_face_discharge,
                &OceanoOperator::local_apply_boundary_face_discharge,
                this,
                vec_ki_discharge.front(),
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }


    {
      unsigned int n_stages = vec_ki_height.size()-1;
      TimerOutput::Scope t(timer, "rk_stage hydro - inv mass + vec upd");
      data.cell_loop(
        &OceanoOperator::local_apply_inverse_mass_matrix_height,
        this,
        vec_ki_height[current_stage+1],
        vec_ki_height.front(),
        std::function<void(const unsigned int, const unsigned int)>(),
        [&](const unsigned int start_range, const unsigned int end_range) {
          if (current_stage == n_stages-1)
            {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  Number k_i           = vec_ki_height[1].local_element(i);
                  const Number sol_i   = solution_height.local_element(i);
                  solution_height.local_element(i)  = sol_i + factor_residual[0] * k_i;
		  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
                      k_i = vec_ki_height[j+1].local_element(i);
                      solution_height.local_element(i) += factor_residual[j]  * k_i;
                    }
                }
            }
          else
            {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  Number k_i            = vec_ki_height[1].local_element(i);
                  const Number sol_i    = solution_height.local_element(i);
                  next_ri_height.local_element(i) = sol_i + factor_residual[0]  * k_i;
		  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
                      k_i = vec_ki_height[j+1].local_element(i);
                      next_ri_height.local_element(i) += factor_residual[j]  * k_i;
                    }
                }
            }
        },
        0);


      data.cell_loop(
        &OceanoOperator::local_apply_inverse_mass_matrix_discharge,
        this,
        vec_ki_discharge[current_stage+1],
        vec_ki_discharge.front(),
        std::function<void(const unsigned int, const unsigned int)>(),
        [&](const unsigned int start_range, const unsigned int end_range) {
          if (current_stage == n_stages-1)
            {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  Number k_i           = vec_ki_discharge[1].local_element(i);
                  const Number sol_i   = solution_discharge.local_element(i);
                  solution_discharge.local_element(i)  = sol_i + factor_residual[0] * k_i;
		  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
                      k_i = vec_ki_discharge[j+1].local_element(i);
                      solution_discharge.local_element(i) += factor_residual[j]  * k_i;
                    }
                }
            }
          else
            {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  Number k_i            = vec_ki_discharge[1].local_element(i);
                  const Number sol_i    = solution_discharge.local_element(i);
                  next_ri_discharge.local_element(i) = sol_i + factor_residual[0]  * k_i;
		  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
                      k_i = vec_ki_discharge[j+1].local_element(i);
                      next_ri_discharge.local_element(i) += factor_residual[j]  * k_i;
                    }
                }
            }
        },
        1);
    }
  }

  // We apply the same concepts to the Additive Runge-Kutta method. The cost of
  // ARK scheme are quite higher. The main problem is memory access: the vectors to access
  // at each stage are `2 * (n_stages-1) +1`, the factor two is related to the presence of
  // the stiff and non-stiff part of the residual. Since ARK is needed only for the momentum
  // equation we mantain the explicit code of the previous section for the continuity equation
  // and we use a different code for the momemtum equation. For the latter we compute the
  // stiff and non-stiff residuals. Note that for the stiff residual we need to build the
  // residual associated to the friction term which can be done with only a cell loop.
  //
  // Another overhead is related to the fact that we cannot update the vector with a single
  // call to `cell_loop()`. First we have to perform vector updates (assemble the right-hand-side
  // composed of the old solution times the mass-matrix plus the ImEx residuals.
  // This is cumulated into the auxiliary `vec_ki_discharge.front()`.
  // Only after we invert the mass-matrix and we put the result into the new solution.
  // The mass-matrix is modified by the Implicit scheme (it contains the Jacobian of
  // the implicit part) and this is why we have a different call to the mass matrix inversion.
  // Moreover this should explain why, into the last `cell_loop()`, the src vector contains
  // the last updated solution: it is needed to compute the Jacobian of the bottom friction.
  //
  // As usual we use a different code path for the last stage. For the ARK scheme we benefit
  // from two facts: the last stage `b_i` coefficients must be equal between the explicit and the
  // implicit schemes, the mass-matrix is the standard one. Thus we assemble the residual once
  // per stage (instead of twice) and we re-assemble vector assemble and mass-inversion in a
  // single call.
#elif defined TIMEINTEGRATOR_ADDITIVERUNGEKUTTA
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::perform_stage_hydro(
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

    unsigned int n_stages = vec_ki_height.size()-1;

    {
      TimerOutput::Scope t(timer, "rk_stage hydro - integrals L_h");

      for (auto &i : bc->supercritical_inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->height_inflow_boundaries)
        i.second->set_time(current_time);
      for (auto &i : bc->discharge_inflow_boundaries)
        i.second->set_time(current_time);

      data.loop(&OceanoOperator::local_apply_cell_height,
                &OceanoOperator::local_apply_face_height,
                &OceanoOperator::local_apply_boundary_face_height,
                this,
                vec_ki_height.front(),
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);

      if (current_stage == n_stages-1)
        {
          data.loop(
                &OceanoOperator::local_apply_cell_discharge,
                &OceanoOperator::local_apply_face_discharge,
                &OceanoOperator::local_apply_boundary_face_discharge,
                this,
                vec_ki_discharge[2*current_stage+1],
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
        }
      else
        {
          data.loop(
                &OceanoOperator::local_apply_cell_nonstiff_discharge,
                &OceanoOperator::local_apply_face_discharge,
                &OceanoOperator::local_apply_boundary_face_discharge,
                this,
                vec_ki_discharge[2*current_stage+1],
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
          data.cell_loop(
                &OceanoOperator::local_apply_cell_stiff_discharge,
                this,
                vec_ki_discharge[2*current_stage+2],
                current_ri,
                true);
        }
    }


    {
      TimerOutput::Scope t(timer, "rk_stage hydro - inv mass + vec upd");
      data.cell_loop(
        &OceanoOperator::local_apply_inverse_mass_matrix_height,
        this,
        vec_ki_height[current_stage+1],
        vec_ki_height.front(),
        std::function<void(const unsigned int, const unsigned int)>(),
        [&](const unsigned int start_range, const unsigned int end_range) {
          if (current_stage == n_stages-1)
            {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  Number k_i           = vec_ki_height[1].local_element(i);
                  const Number sol_i   = solution_height.local_element(i);
                  solution_height.local_element(i)  = sol_i + factor_residual[0] * k_i;
		  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
                      k_i = vec_ki_height[j+1].local_element(i);
                      solution_height.local_element(i) += factor_residual[j]  * k_i;
                    }
                }
            }
          else
            {
              /* DEAL_II_OPENMP_SIMD_PRAGMA */
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  Number k_i            = vec_ki_height[1].local_element(i);
                  const Number sol_i    = solution_height.local_element(i);
                  next_ri_height.local_element(i) = sol_i + factor_residual[0]  * k_i;
		  for (unsigned int j = 1; j < current_stage+1; ++j)
		    {
                      k_i = vec_ki_height[j+1].local_element(i);
                      next_ri_height.local_element(i) += factor_residual[j]  * k_i;
                    }
                }
            }
        },
        0);

      if (current_stage == n_stages-1)
        {
          data.cell_loop(
            &OceanoOperator::local_apply_inverse_mass_matrix_discharge,
            this,
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
            &OceanoOperator::local_apply_cell_mass_discharge,
            this,
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
            &OceanoOperator::local_apply_inverse_modified_mass_matrix_discharge,
            this,
            next_ri_discharge,
            {vec_ki_discharge.front(), current_ri[0], current_ri[1]},
            true);
        }
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
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::project_hydro(
    const Function<dim> &                       function,
    LinearAlgebra::distributed::Vector<Number> &solution_height,
    LinearAlgebra::distributed::Vector<Number> &solution_discharge) const
  {
    FEEvaluation<dim, degree, degree + 1, 1, Number> phi_height(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, 1, Number>
      inverse_height(phi_height);
    solution_height.zero_out_ghost_values();
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi_height.reinit(cell);
        for (unsigned int q = 0; q < phi_height.n_q_points; ++q)
          phi_height.submit_dof_value(evaluate_function<dim, Number>(
          					 function,
                                                 phi_height.quadrature_point(q),
                                                 0),
                               q);
        inverse_height.transform_from_q_points_to_basis(1,
                                                 phi_height.begin_dof_values(),
                                                 phi_height.begin_dof_values());
        phi_height.set_dof_values(solution_height);
      }

    FEEvaluation<dim, degree, degree + 1, dim, Number> phi_discharge(data, 1, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim, Number>
      inverse_discharge(phi_discharge);
    solution_discharge.zero_out_ghost_values();
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        Tensor<1, dim, VectorizedArray<Number>> discharge;
        phi_discharge.reinit(cell);
        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q) 
          {
            for (unsigned int d = 0; d < dim; ++d)
              discharge[d] = evaluate_function<dim, Number>(function,
                                                phi_discharge.quadrature_point(q),
                                                1+d);
            phi_discharge.submit_dof_value(discharge, q);
          }
        inverse_discharge.transform_from_q_points_to_basis(dim,
                                                 phi_discharge.begin_dof_values(),
                                                 phi_discharge.begin_dof_values());
        phi_discharge.set_dof_values(solution_discharge);
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
  // dimension `2` because we compute only one error for all the
  // momentum components.
  template <int dim, int n_tra, int degree, int n_points_1d>
  std::array<double, 2> OceanoOperator<dim, n_tra, degree, n_points_1d>::compute_errors_hydro(
    const Function<dim> &                             function,
    const LinearAlgebra::distributed::Vector<Number> &solution_height,
    const LinearAlgebra::distributed::Vector<Number> &solution_discharge) const
  {
    TimerOutput::Scope t(timer, "compute errors");
    const unsigned int n_err = 2;
    double             errors_squared[n_err] = {};
    FEEvaluation<dim, degree, n_points_1d, 1, Number> phi_height(data, 0, 0);
    FEEvaluation<dim, degree, n_points_1d, dim, Number> phi_discharge(data, 1, 0);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi_height.reinit(cell);
        phi_height.gather_evaluate(solution_height, EvaluationFlags::values);
        phi_discharge.reinit(cell);
        phi_discharge.gather_evaluate(solution_discharge, EvaluationFlags::values);

        VectorizedArray<Number> local_errors_squared[n_err] = {};
        Tensor<1, 1+dim, VectorizedArray<Number>> error;
        for (unsigned int q = 0; q < phi_height.n_q_points; ++q)
          {
            error[0] = evaluate_function<dim, Number>(
                              function, 
              		      phi_height.quadrature_point(q),
              		      0) - 
              	       phi_height.get_value(q);
            const auto JxW = phi_height.JxW(q);
            local_errors_squared[0] += error[0] * error[0] * JxW;
          }
        for (unsigned int q = 0; q < phi_discharge.n_q_points; ++q)
          {
            for (unsigned int d = 0; d < dim; ++d)
              error[d+1] = evaluate_function<dim, Number>(
              			function, 
              			phi_discharge.quadrature_point(q),
              			d+1) -
              		  phi_discharge.get_value(q)[d];
            const auto JxW = phi_discharge.JxW(q);
            for (unsigned int d = 0; d < dim; ++d)
              local_errors_squared[1] += (error[d + 1] * error[d + 1]) * JxW;
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
  // speed $c = \sqrt{g h}$ relative to the medium moving at
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
  template <int dim, int n_tra, int degree, int n_points_1d>
  double OceanoOperator<dim, n_tra, degree, n_points_1d>::compute_cell_transport_speed(
    const LinearAlgebra::distributed::Vector<Number> &solution_height,
    const LinearAlgebra::distributed::Vector<Number> &solution_discharge) const
  {
    TimerOutput::Scope t(timer, "compute transport speed");
    Number             max_transport = 0;
    FEEvaluation<dim, degree, degree + 1, 1, Number> phi_height(data, 0, 1);
    FEEvaluation<dim, degree, degree + 1, dim, Number> phi_discharge(data, 1, 1);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi_height.reinit(cell);
        phi_height.gather_evaluate(solution_height, EvaluationFlags::values);
        phi_discharge.reinit(cell);
        phi_discharge.gather_evaluate(solution_discharge, EvaluationFlags::values);
        VectorizedArray<Number> local_max = 0.;
        for (unsigned int q = 0; q < phi_height.n_q_points; ++q)
          {
            const VectorizedArray<Number> data_q =
              evaluate_function<dim, Number>(
                *bc->problem_data, phi_height.quadrature_point(q), 0);
            const auto zq = phi_height.get_value(q);
            const auto qq = phi_discharge.get_value(q);
            const auto velocity = model.velocity<dim>(zq, qq, data_q);

            const auto inverse_jacobian = phi_height.inverse_jacobian(q);
            const auto convective_speed = inverse_jacobian * velocity;
            VectorizedArray<Number> convective_limit = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              convective_limit =
                std::max(convective_limit, std::abs(convective_speed[d]));

            const auto speed_of_sound =
              std::sqrt(model.square_wavespeed(zq, data_q));

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
  // In general the output can also depend on the solution gradient (think for example
  // to the vorticity) but this part has been commented for now, see `do_schlieren_plot`.
  // For the postprocessed variables, a well defined order must be followed: first the
  // velocity vector, then all the scalars.
  //
  // A few more words about the loop over the evaluation points. We use the class
  // `IndexSet` that represents a subset of dofs. It can be used to denote the set
  // of locally owned degrees of freedom or those among all degrees of freedom that
  // are stored on a particular processor in a distributed parallel computation.
  // We iterate over the `IndexSets` with the usual `begin()` and `end()`. Then the
  // access to the vector is done with the global index. Since we are operating
  // on locally owened dofs that represent a contiguous range, the access should be
  // as fast as a local access.
  // We are assuming that all the solution variables are distributed in the
  // same fashion over the processores. For safaty the last assert checks
  // the size of the solution vectors on each processor and throw an exception if
  // they are not equals.
  //
  // The iterator is not vectorized. However the usual way to recover data with
  // the `evaluate_function` operates on vectorized data. With a few tricks the
  // the `evaluate_function` has been adapted to a non-vectorized loop.
  //
  // In some case we would like to compute only the velocity field.
  // Instead of using a default parameter, which does not fit well with arguments
  // passed by reference, we just check if the entry scalar field is empty or not.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperator<dim, n_tra, degree, n_points_1d>::evaluate_vector_field(
    const DoFHandler<dim>                                   &dof_handler,
    const LinearAlgebra::distributed::Vector<Number>        &solution_height,
    const LinearAlgebra::distributed::Vector<Number>        &solution_discharge,
    std::map<unsigned int, Point<dim>>                      &evaluation_points,
    LinearAlgebra::distributed::Vector<Number>              &computed_vector_quantities,
    std::vector<LinearAlgebra::distributed::Vector<Number>> &computed_scalar_quantities) const
  {
    const std::vector<std::string> postproc_names = model.postproc_vars_name;

    Assert(computed_vector_quantities.locally_owned_size() == solution_height.locally_owned_size()*dim,
           ExcInternalError());
    Assert(computed_scalar_quantities.size() == postproc_names.size()-dim ||
           computed_scalar_quantities.empty(),
           ExcInternalError());
    Assert(solution_discharge.locally_owned_size() == solution_height.locally_owned_size()*dim,
           ExcInternalError());

    IndexSet myset = dof_handler.locally_owned_dofs();
    IndexSet::ElementIterator index = myset.begin();
    for (index=myset.begin(); index!=myset.end(); ++index)
      {
        const auto height = solution_height(*index);
        Tensor<1, dim> discharge;
        for (unsigned int d = 0; d < dim; ++d)
          discharge[d] = solution_discharge(*index*dim+d);

        Point<dim,VectorizedArray<Number>> x_evaluation_points;
        for (unsigned int d = 0; d < dim; ++d)
          x_evaluation_points[d] = evaluation_points[*index][d];

        const auto data = evaluate_function<dim, Number>(
            *bc->problem_data, x_evaluation_points, 0);

        const Tensor<1, dim> velocity = model.velocity<dim>(height, discharge, data[0]);

        for (unsigned int d = 0; d < dim; ++d)
          computed_vector_quantities(*index*dim+d) = velocity[d];

        if (!computed_scalar_quantities.empty())
          for (unsigned int v = 0; v < postproc_names.size()-dim; ++v)
            {
              if (postproc_names[v+dim] == "pressure")
                computed_scalar_quantities[v](*index) =
                  model.pressure(height, data[0]);
              else if (postproc_names[v+dim] == "depth")
                computed_scalar_quantities[v](*index) = height + data[0];
              else if (postproc_names[v+dim] == "speed_of_sound")
                computed_scalar_quantities[v](*index) =
                  std::sqrt(model.square_wavespeed(height, data[0]));
              else
                {
                  std::cout << "Postprocessing variable " << postproc_names[dim+v]
                            << " does not exist. Consider to code it in your model"
                            << std::endl;
                  Assert(false, ExcNotImplemented());
                }
            }
      }
  }

} // namespace SpaceDiscretization

#endif //OCEANDG_HPP
