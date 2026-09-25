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
 * Author: Martin Kronbichler (copied from), 2020
           Luca Arpaia, 2023
 *         Giuseppe Orlando, 2024
 */
#ifndef OCEANDGEXPLICIT_H
#define OCEANDGEXPLICIT_H
 
// The following files include the oceano libraries
#include <space_discretization/OceanDG.h>

/**
 * Namespace containing the spatial Operator
 */

namespace SpaceDiscretization
{

  using namespace dealii;

  using Number = double;



  // @sect3{The OceanoOperator with explicit integrator}

  // This class implements the assembly of the evaluators for the ocean problem using
  // the explicit scheme. In other words, it implements the evaluation of the ocean
  // operator as a whole, i.e., $\mathcal M^{-1} \mathcal L(t, \mathbf{w})$,
  // calling into the local evaluators in the base class `OceanoOperator`.
  // In fact this class is derived from `OceanoOperator` and simply selects the appropriate
  // assembly method. Thanks to a `using` declaration, we can also access base-class
  // members with the same variable name. This was very helpful during the
  // implementation.
  template <int dim, int n_tra, int degree, int n_points_1d>
  class OceanoOperatorExplicit : public OceanoOperator<dim, n_tra, degree, n_points_1d>
  {
  public:
    OceanoOperatorExplicit(
      IO::ParameterHandler      &param,
      ICBC::BcBase<dim, 1+dim+n_tra> *bc,
      TimerOutput               &timer_output,
      const unsigned int         max_iteration_height);
    ~OceanoOperatorExplicit() = default;

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

    using Base = OceanoOperator<dim, n_tra, degree, n_points_1d>;
    using Base::bc;

  protected:
    using Base::data;
    using Base::timer;
  };



  // The constructor simply initialize the base classes of the Ocean Operator.
  template <int dim, int n_tra, int degree, int n_points_1d>
  OceanoOperatorExplicit<dim, n_tra, degree, n_points_1d>::OceanoOperatorExplicit(
    IO::ParameterHandler             &param,
    ICBC::BcBase<dim, 1+dim+n_tra>   *bc,
    TimerOutput                      &timer,
    const unsigned int                max_iteration_height)
    : OceanoOperator<dim, n_tra, degree, n_points_1d>(
      param, bc, timer, max_iteration_height)
  {}



  // Let us move to the function that does an entire stage of a Runge--Kutta
  // update. It calls OceanoOperator::apply() followed by some updates
  // to the vectors. First we need to adjust
  // the time in the functions we have associated with the various parts of
  // the boundary, in order to be consistent with the equation in case the
  // boundary data is time-dependent. Then, the core of the routine is
  // MatrixFree::loop() that performs the cell and face integrals, including
  // the necessary ghost data exchange in the `src` vector. The seventh
  // argument to the function, `true`, specifies that we want to zero the
  // `dst` vector as part of the loop, before we start accumulating integrals
  // into it. This variant is preferred over explicitly calling `dst = 0.;`
  // before the loop as the zeroing operation is done on a subrange of the
  // vector in parts that are written by the integrals nearby. This enhances
  // data locality and allows for caching, saving one roundtrip of vector data
  // to main memory and enhancing performance. The last two arguments to the
  // loop determine which data is exchanged: Since we only access the values
  // of the shape functions one faces, typical of first-order hyperbolic
  // problems, and since we have a nodal basis with nodes at the reference
  // element surface, we only need to exchange those parts. This again saves
  // precious memory bandwidth.
  //
  // Once the spatial operator $\mathcal L$ is applied, we need to make a
  // second round and apply the inverse mass matrix. Here, we call
  // MatrixFree::cell_loop() since only cell integrals appear. The cell loop
  // is cheaper than the full loop as access only goes to the degrees of
  // freedom associated with the locally owned cells, which is simply the
  // locally owned degrees of freedom for DG discretizations. Thus, no ghost
  // exchange is needed here.
  //
  // Rather than performing these
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
  //
  // An important implementation detail is the static cast of the owning class
  // passed to the data loop. MatrixFree::loop() requires the pointer-to-member
  // functions and the owning object pointer to refer to the same class type.
  // Since the local_apply functions are members of the base class, we cast
  // this from the derived class pointer to a pointer to the base class (this
  // fix was found by chatGPT, otherwise impossible to compile).
  //
  // Finally we comment the zeroing out of the ghost cells. Passing to the
  // matrix-free loop a ghosted destination vector, is not allowed. This is done
  // for safety as it ensures that ghost slots start at zero before accumalating
  // values from multiple processors. This is why the solution vectors ghost cells
  // need to be cleaned out.
  template <int dim, int n_tra, int degree, int n_points_1d>
  void OceanoOperatorExplicit<dim, n_tra, degree, n_points_1d>::perform_stage_hydro(
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

      data.loop(&Base::local_apply_cell_discharge,
                &Base::local_apply_face_discharge,
                &Base::local_apply_boundary_face_discharge,
                static_cast<const Base *>(this),
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


      data.cell_loop(
        &Base::local_apply_inverse_mass_matrix_discharge,
        static_cast<const Base *>(this),
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

} // namespace SpaceDiscretization

#endif //OCEANDGEXPLICIT_H
