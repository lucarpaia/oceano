/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2007 - 2023 by the deal.II authors
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
 * Author: Moritz Allmaras, Texas A&M University, 2007
 	   Luca Arpaia,	CNR-ISMAR,                2023
 */
#include <deal.II/base/parameter_handler.h>

/**
 * Namespace containing the input/output file handler.
 */

namespace IO
{

  using namespace dealii;


 
  // The next class is responsible for preparing the `ParameterHandler` 
  // object and reading parameters from an input file. It includes a 
  // function `declare_parameters` that declares all the necessary 
  // parameters and a `read_parameters` function that is called from 
  // outside to initiate the parameter reading process.
  class ParameterReader : public Subscriptor
  {
  public:
    ParameterReader(ParameterHandler &);
    void read_parameters(const std::string &);
 
  private:
    void              declare_parameters();
    ParameterHandler &prm;
  };
 
  // The constructor stores a reference to the `ParameterHandler` object 
  // that is passed to it: 
  ParameterReader::ParameterReader(ParameterHandler &paramhandler)
    : prm(paramhandler)
  {}
 
  // The `declare_parameters` function declares all the parameters that 
  // our `ParameterHandler` object will be able to read from input files, 
  // along with their types, range conditions and the subsections they 
  // appear in. We will wrap all the entries that go into a section in a 
  // pair of braces to force the editor to indent them by one level, making 
  // it simpler to read which entries together form a section: 
  void ParameterReader::declare_parameters()
  {

    // Parameters for mesh include the mesh file and the
    // number of global refinement steps that are applied to the initial
    // coarse mesh.
    prm.enter_subsection("Mesh & geometry parameters");
    {
    
      prm.declare_entry("Mesh_filename",
                        "meshfile.msh",
                        Patterns::Anything(),
                        "Mesh filename in Gmsh format "
                        "version >= 4.1");
    
      // For the number of refinement steps, we allow integer values 
      // in the range $[0,\infty)$, where the omitted second argument to the 
      // `Patterns::Integer` object denotes the half-open interval.    
      prm.declare_entry("Number_of_refinements",
                        "6", Patterns::Integer(0),
                        "Number of global mesh refinement steps "
                        "applied to initial coarse grid");

      // The remeshing frequency depends on the test. Remeshing
      // should be done every timestep if the mesh adapts to fast phenomenon. Note that
      // the default value is set to a very high number or no mesh adaptation. If this
      // key is not specified the mesh adaptation is not active and the simulation will
      // continue with the initial mesh given above.
      prm.declare_entry("Remesh_tick",
                        "10000000000.", Patterns::Double(0),
                        "Time interval we remesh");

      // This are the two tresholds for the normalized cellwise error after/below which
      // the cell is marked for refinement/coarsening. The remesh thick is the frequency
      // for performing remeshing.
      prm.declare_entry("Threshold_for_refinement",
                        "0.50", Patterns::Double(0,1),
                        "Normalized error threshold after which a cell "
                        "is marked for refinement");

      prm.declare_entry("Threshold_for_coarsening",
                        "0.25", Patterns::Double(0,1),
                        "Normalized error threshold below which a cell "
                        "is marked for coarsening");

      // The max level of refinement is the
      // maximum grid level. The default value (zero) is important
      // because it fixes the correct behaviour of the code without mesh adaptation.
      // At this stage we have also fixed the maximum grid level to be less then
      // 100 which means that the most refined cell can be "only" $2^10$ times
      // smaller then the initial one, e.g. if you start with a resolution of 100 km
      // you can can have cells as smaller as 100 m.
      prm.declare_entry("Max_level_of_refinement",
                        "0", Patterns::Integer(0,100),
                        "Maximum level of grid refinement. "
                        "A value of one means that the grid, at maximum, "
                        "will be halved, etc ... "
                        "It controls the maximum resolution");

      // The min mesh size is a second parameter that keeps under control how small
      // are the elements after a mesh refinement stage. It is the minimum acceptable
      // mesh size or the maximum resolution, computed as the minimum distance between
      // the vertices of a cell. If a cell is smaller then such a threshold no
      // refinement is possible for that cell. This is useful for meshes with variable
      // mesh size, in order to let coarser regions to be refined until a desired
      // resolution while preventing other regions to be over-resolved.
      // By default it set to zero, so that the only parameter that controls the
      // minimum mesh size is the previous one.
      prm.declare_entry("Min_mesh_size",
                        "0.0", Patterns::Double(0),
                        "Minimum mesh size for all the cell. "
                        "It controls the maximum resolution");

      // A static refinement indicator can be read read from file, if the
      // preprocessing `AMR_FROMFILE` is defined. Otherwise this parameter
      // is ignored.
      prm.declare_entry("Static_refinement_indicator_filename",
                        "refinement_indicator.tzt.gz",
                        Patterns::Anything(),
                        "Name of the static refinement indicator file "
                        "in structured format (with extension txt.gz)");
     }
    prm.leave_subsection();

    // Paramteres for time includes the final time of the simulation and the Courant
    // number. For stability reason with DG the courant number should much less then one.
    // Here we fix the courant $\in [0,1]$ as the range accepted by the parameter reader.
    prm.enter_subsection("Time parameters");
    {
      prm.declare_entry("Final_time", "0.0", Patterns::Double(0),
                        "Final time of the simulation");

      prm.declare_entry("CFL", "1.0", Patterns::Double(0,1),
                        "Courant number");
     }
    prm.leave_subsection();

    // The next subsection is devoted to the physical parameters appearing 
    // in the equation. These are the gravitational acceleration $g$, the air
    // and water density, $\rho_{air}$ and $\rho_0$.
    // They need to lie in the half-open interval $[0,\infty)$
    // represented by calling the `Patterns::Double` class with only the 
    // left end-point as argument
    prm.enter_subsection("Physical constants");
    {
      prm.declare_entry("g", "9.81", Patterns::Double(0), "Gravity");

      prm.declare_entry("water_density",
                        "1025.",
                        Patterns::Double(0),
                        "Reference water density");

      prm.declare_entry("air_density",
                        "1.225",
                        Patterns::Double(0),
                        "Air density");

      prm.declare_entry("wind_drag",
                        "0.0025",
                        Patterns::Double(0),
                        "Wind drag coefficient");

      prm.declare_entry("coriolis_f0",
                        "1e-4",
                        Patterns::Double(0),
                        "Wind drag coefficient");

      prm.declare_entry("coriolis_beta",
                        "2e-11",
                        Patterns::Double(0),
                        "Wind drag coefficient");

      prm.declare_entry("bottom_friction",
                        "0.0",
                        Patterns::Double(0),
                        "Bottom friction coefficient or Manning number"
                        "depending on the bottom friction formulation");
    }
    prm.leave_subsection();

    // A physical model may need some external data. For the Shallow Water
    // equations these are typically the bathymetry or the wind field.
    // To better handle large dataseta, data are inputted via txt compressed
    // file (e.g. ".txt.gz"). The next subsection fills the different data file
    // names.
    // For the initial data, the default behaviour (no file given) switches to
    // a constant data initial condition.
    prm.enter_subsection("Input data files");
    {
      prm.declare_entry("Bathymetry_filename",
                        "bathymetry.txt.gz",
                        Patterns::Anything(),
                        "Name of the bathymetry file "
                        "in structured format (with extension txt.gz)");

      prm.declare_entry("Exact_solution_filename",
                        "exact_solution.txt.gz",
                        Patterns::Anything(),
                        "Name of the exact/reference solution file "
                        "in structured format (with extension txt.gz)");

      prm.declare_entry("Initial_freesurface_filename",
                        "0.",
                        Patterns::Anything(),
                        "Name of the file with the initial free-surface"
                        "in structured format (with extension txt.gz)."
                        "A constant double value is also accepted.");

      prm.declare_entry("Initial_discharge_x_filename",
                        "0.",
                        Patterns::Anything(),
                        "Name of the file with the initial discharge (x-component)"
                        "in structured format (with extension txt.gz)."
                        "A constant double value is also accepted.");

      prm.declare_entry("Initial_discharge_y_filename",
                        "0.",
                        Patterns::Anything(),
                        "Name of the file with the initial discharge (y-component)"
                        "in structured format (with extension txt.gz)."
                        "A constant double value is also accepted.");

      for (unsigned int count = 1; count < 20; ++count)
        prm.declare_entry("Initial_tracer_"+std::to_string(count)+"_filename",
                          "1."
                          +std::to_string(count)+"_filename in the parameter file",
                          Patterns::Anything(),
                          "Name of the file with the initial tracer"
                          "in structured format (with extension txt.gz)."
                          "A constant double value is also accepted.");
    }
    prm.leave_subsection();

    // Boundary condition may be supplied via the following list. For each
    // entry you write the boundary id of the gmsh file, the boundary condition
    // type and a filename in structured format containing the boundary data.
    // Alternatively to a boundary file, a double value for a constant boundary
    // data can be also given.
    prm.enter_subsection("Boundary conditions");
    {
      for (unsigned int count = 1; count < 20; ++count)
        prm.declare_entry("Boundary_"+std::to_string(count),
                          "There is no entry Boundary in the parameter file",
                          Patterns::Anything(),
                          "Boundary id, type and filename separated by a colon");
    }
    prm.leave_subsection();

    // Last but not least we would like to be able to change some properties
    // of the output, like the output filename or the output time frequency,
    // through entries in the configuration file, which is the purpose of the
    // last subsection:
    prm.enter_subsection("Output parameters");
    {
      prm.declare_entry("Output_filename",
                        "solution",
                        Patterns::Anything(),
                        "Name of the output file (without extension)");

      prm.declare_entry("Solution_tick",
                        "10000000000.",
                        Patterns::Double(0),
                        "Time interval we write the solution");

      prm.declare_entry("Point_history_tick",
                        "10000000000.",
                        Patterns::Double(0),
                        "Time interval we write the point history");

      for (unsigned int count = 1; count < 20; ++count)
        prm.declare_entry("Point_history_"+std::to_string(count),
                          "There is no entry Point_history in the parameter file",
                          Patterns::Anything(),
                          "Point history coordinates separated by a colon");

      prm.declare_entry("Output_error",
                        "false",
                        Patterns::Bool(),
                        "Flag to append also the error field to the output file");

      prm.declare_entry("Output_meshsize",
                        "false",
                        Patterns::Bool(),
                        "Flag to output the mesh size field");

      // Since different output formats may require different parameters for
      // generating output, it would be cumbersome if we had to declare all these
      // parameters by hand for every possible output format supported in the library.
      // Instead, each output format has a `FormatFlags::declare_parameters` function,
      // which declares all the parameters specific to that format in an own subsection.
      // The following call of `DataOutInterface<1>::declare_parameters` executes
      // declare_parameters for all available output formats, so that for each format
      // an own subsection will be created with parameters declared for that particular
      // output format.
      DataOutInterface<1>::declare_parameters(prm);
    }
    prm.leave_subsection();
  }
 
  // This is the main function in the ParameterReader class. It gets called
  // from outside, first declares all the parameters, and then reads them from
  // the input file whose filename is provided by the caller. After the call
  // to this function is complete, the prm object can be used to retrieve the
  // values of the parameters read in from the file :
  void ParameterReader::read_parameters(const std::string &parameter_file)
  {
    declare_parameters();
 
    prm.parse_input(parameter_file);
  }
  
} // namespace IO
