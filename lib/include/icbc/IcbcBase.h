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
#ifndef ICBC_ICBCBASE_HPP
#define ICBC_ICBCBASE_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>

/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  using namespace dealii;


  
  // With the following line we use the parallel output class of Deal.II.
  // The `ConditionalOStream` class has already been defined in the main and 
  // we could have passed it from the interface instead of redefining it here. 
  ConditionalOStream pcout(std::cout, 
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0); 
  
  // @sect3{Equation data}

  // We now define a base class for the boundary conditions. Given that
  // the Euler equations are a problem with $d+2$ equations in $d$ dimensions,
  // we need to tell the base class about the correct number of
  // components. This is realized templating with the dimension the icbc class.
  // The base class is overridden by derived classes which contain the  
  // boundary conditions specific to each test case. 
  template <int dim, int n_vars>  
  class BcBase
  {
  public:
        
    BcBase(){};
    virtual ~BcBase(){};
    
    // The next members are for the functions that appears into the boundary 
    // conditions or into the forcing terms. They are defined with the help of the
    // Deal.II `Function<dim>` which can design vector functions of the space and time.
    // The first four members associate, with the aid of a map, each boundary id with the
    // corresponding function for the boundary condition. The last member is used
    // for the problem spatially and time varying parameters, such as friction, bathymetry
    // or wind. 
    // Actually we use pointers to Function, but note that we do not use regular 
    // pointers but `unique_ptr`. These are particular pointers that 
    // destroy the object to which they point, after use. This avoids
    // memory leaks with dynamic allocation. No other pointer should point 
    // to its managed object. For this reason we will see that these particular
    // pointers cannot be copied but they can only be moved with `std::move()`
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
                                   supercritical_inflow_boundaries;
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
                                   supercritical_outflow_boundaries;
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
                                   height_inflow_boundaries;
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
                                   discharge_inflow_boundaries;
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
                                   absorbing_outflow_boundaries;
    std::set<types::boundary_id>   wall_boundaries;

    std::unique_ptr<Function<dim>> problem_data;

    // The subsequent four member functions are the ones that fill the boundary 
    // containers. They must be called from outside to specify the various types 
    // of boundaries. For an inflow boundary, we must specify all components in
    // terms of free-surface $\zeta$, momentum $h\mathbf{u}$ and, eventually tracers.
    // Given this information, we then store the
    // function alongside the respective boundary id in a map member variable of
    // this class. Likewise, we proceed for the subcritical/supercritical
    // outflow boundaries (where we request a function as well,
    // which we use to retrieve the far-field state or the energy). For the
    // wall (no-penetration) boundaries we impose zero normal velocity, no
    // function necessary, so we only request the boundary id. For the present
    // DG code where boundary conditions are solely applied as part of the weak
    // form (during time integration), the call to set the boundary conditions
    // can appear both before or after the `reinit()` call to this class. This
    // is different from continuous finite element codes where the boundary
    // conditions determine the content of the AffineConstraints object that is
    // sent into MatrixFree for initialization, thus requiring to be set before
    // the initialization of the matrix-free data structures.
    void set_supercritical_inflow_boundary(
      const types::boundary_id       boundary_id,
      std::unique_ptr<Function<dim>> inflow_function);

    void set_supercritical_outflow_boundary(
      const types::boundary_id       boundary_id,
      std::unique_ptr<Function<dim>> outflow_energy);

    void set_height_inflow_boundary(
      const types::boundary_id       boundary_id,
      std::unique_ptr<Function<dim>> inflow_energy);

    void set_discharge_inflow_boundary(
      const types::boundary_id       boundary_id,
      std::unique_ptr<Function<dim>> inflow_energy);

    void set_absorbing_outflow_boundary(
      const types::boundary_id       boundary_id,
      std::unique_ptr<Function<dim>> outflow_energy);

    void set_wall_boundary(const types::boundary_id boundary_id);

    void set_problem_data(std::unique_ptr<Function<dim>> problem_data);

    // The next member compose the different boundary conditions for each test case. 
    // It is overridden by the derived classes specific to each test case. 
    // In the boundary condition function the user defines the 
    // composition of the boundary, that is for each boundary id a boundary condition 
    // among the members defined above must be provided.
    virtual void set_boundary_conditions()
    {
      pcout << "ERROR in set_boundary_conditions()"
            << "The function is not written in the icbc derived class"
            << std::endl;
    }
        
  }; 

  
  
  // The checks added in each of the four function are used to
  // ensure that boundary conditions are mutually exclusive on the various
  // parts of the boundary, i.e., that a user does not accidentally designate a
  // boundary as both an inflow and say a supercritical outflow boundary.
  template <int dim, int n_vars>
  void BcBase<dim, n_vars>::set_supercritical_inflow_boundary(
    const types::boundary_id       boundary_id,
    std::unique_ptr<Function<dim>> inflow_function)
  {
    AssertThrow(supercritical_outflow_boundaries.find(boundary_id) ==
                    supercritical_outflow_boundaries.end() &&
                absorbing_outflow_boundaries.find(boundary_id) ==
                    absorbing_outflow_boundaries.end() &&
                height_inflow_boundaries.find(boundary_id) ==
                    height_inflow_boundaries.end() &&
                discharge_inflow_boundaries.find(boundary_id) ==
                    discharge_inflow_boundaries.end() &&
                wall_boundaries.find(boundary_id) == wall_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as supercritical inflow"));
    AssertThrow(inflow_function->n_components == n_vars,
                ExcMessage("Expected function with n_vars components"));

    supercritical_inflow_boundaries[boundary_id] = std::move(inflow_function);
  }


  template <int dim, int n_vars>
  void BcBase<dim, n_vars>::set_supercritical_outflow_boundary(
    const types::boundary_id       boundary_id,
    std::unique_ptr<Function<dim>> outflow_function)
  {
    AssertThrow(supercritical_inflow_boundaries.find(boundary_id) ==
                    supercritical_inflow_boundaries.end() &&
                absorbing_outflow_boundaries.find(boundary_id) ==
                    absorbing_outflow_boundaries.end() &&
                height_inflow_boundaries.find(boundary_id) ==
                    height_inflow_boundaries.end() &&
                discharge_inflow_boundaries.find(boundary_id) ==
                    discharge_inflow_boundaries.end() &&
                wall_boundaries.find(boundary_id) == wall_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as supercritical outflow"));
    AssertThrow(outflow_function->n_components == n_vars,
                ExcMessage("Expected function with n_vars components"));

    supercritical_outflow_boundaries[boundary_id] = std::move(outflow_function);
  }


  template <int dim, int n_vars>
  void BcBase<dim, n_vars>::set_height_inflow_boundary(
    const types::boundary_id       boundary_id,
    std::unique_ptr<Function<dim>> inflow_function)
  {
    AssertThrow(supercritical_inflow_boundaries.find(boundary_id) ==
                    supercritical_inflow_boundaries.end() &&
                supercritical_outflow_boundaries.find(boundary_id) ==
                    supercritical_outflow_boundaries.end() &&
                absorbing_outflow_boundaries.find(boundary_id) ==
                    absorbing_outflow_boundaries.end() &&
                discharge_inflow_boundaries.find(boundary_id) ==
                    discharge_inflow_boundaries.end() &&
                wall_boundaries.find(boundary_id) == wall_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as height inflow"));
    AssertThrow(inflow_function->n_components == n_vars,
                ExcMessage("Expected function with n_vars components"));

    height_inflow_boundaries[boundary_id] = std::move(inflow_function);
  }


  template <int dim, int n_vars>
  void BcBase<dim, n_vars>::set_discharge_inflow_boundary(
    const types::boundary_id       boundary_id,
    std::unique_ptr<Function<dim>> inflow_function)
  {
    AssertThrow(supercritical_inflow_boundaries.find(boundary_id) ==
                    supercritical_inflow_boundaries.end() &&
                supercritical_outflow_boundaries.find(boundary_id) ==
                    supercritical_outflow_boundaries.end() &&
                absorbing_outflow_boundaries.find(boundary_id) ==
                    absorbing_outflow_boundaries.end() &&
                height_inflow_boundaries.find(boundary_id) ==
                    height_inflow_boundaries.end() &&
                wall_boundaries.find(boundary_id) == wall_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as height inflow"));
    AssertThrow(inflow_function->n_components == n_vars,
                ExcMessage("Expected function with n_vars components"));

    height_inflow_boundaries[boundary_id] = std::move(inflow_function);
  }


  template <int dim, int n_vars>
  void BcBase<dim, n_vars>::set_absorbing_outflow_boundary(
    const types::boundary_id       boundary_id,
    std::unique_ptr<Function<dim>> outflow_function)
  {
    AssertThrow(supercritical_inflow_boundaries.find(boundary_id) ==
                    supercritical_inflow_boundaries.end() &&
                supercritical_outflow_boundaries.find(boundary_id) ==
                    supercritical_outflow_boundaries.end() &&
                height_inflow_boundaries.find(boundary_id) ==
                    height_inflow_boundaries.end() &&
                wall_boundaries.find(boundary_id) == wall_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as supercritical outflow"));
    AssertThrow(outflow_function->n_components == n_vars,
                ExcMessage("Expected function with n_vars components"));

    absorbing_outflow_boundaries[boundary_id] = std::move(outflow_function);
  }


  template <int dim, int n_vars>
  void BcBase<dim, n_vars>::set_wall_boundary(
    const types::boundary_id boundary_id)
  {
    AssertThrow(supercritical_inflow_boundaries.find(boundary_id) ==
                    supercritical_inflow_boundaries.end() &&
                supercritical_outflow_boundaries.find(boundary_id) ==
                    supercritical_outflow_boundaries.end() &&
                height_inflow_boundaries.find(boundary_id) ==
                    height_inflow_boundaries.end() &&
                absorbing_outflow_boundaries.find(boundary_id) ==
                    absorbing_outflow_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as wall boundary"));

    wall_boundaries.insert(boundary_id);
  }

  // The size of the data is fixed to five scalar quantities. The first component is the bathymetry.
  // The second is the bottom friction coefficient. The third and fourth components
  // are the cartesian components of the wind velocity (in order, eastward and northward).
  // The fifth one is the Coriolis parameter.
  template <int dim, int n_vars>
  void BcBase<dim, n_vars>::set_problem_data(
    std::unique_ptr<Function<dim>> problem_data)
  {
    AssertDimension(problem_data->n_components, dim+3);

    this->problem_data = std::move(problem_data);
  }

} // namespace ICBC

#endif //ICBC_ICBCBASE_HPP
