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
                        "6",
                        Patterns::Integer(0),
                        "Number of global mesh refinement steps "
                        "applied to initial coarse grid");
     }
    prm.leave_subsection();

    // Paramteres for time includes the final time of the simulation
    prm.enter_subsection("Time parameters");
    {
      prm.declare_entry("Final_time", "0.0", Patterns::Double(0),
                        "Final time of the simulation");
     }
    prm.leave_subsection();

    // The next subsection is devoted to the physical parameters appearing 
    // in the equation, which for now is only the real gas constant $\gamma$. 
    // Again, it need to lie in the half-open interval $[0,\infty)$ 
    // represented by calling the `Patterns::Double` class with only the 
    // left end-point as argument: 
    prm.enter_subsection("Physical constants");
    {
      prm.declare_entry("g", "1.4", Patterns::Double(0), "Real gas constant");
    }
    prm.leave_subsection();
 
    // Last but not least we would like to be able to change some properties 
    // of the output, like output filename and time interval, through entries in the 
    // configuration file, which is the purpose of the last subsection: 
    prm.enter_subsection("Output parameters");
    {
      prm.declare_entry("Output_filename",
                        "solution",
                        Patterns::Anything(),
                        "Name of the output file (without extension)");
                        
      prm.declare_entry("Output_tick",
                        "10000000000.",
                        Patterns::Double(0),
                        "Time interval we write the outputs");

      prm.declare_entry("Output_error",
                        "0",
                        Patterns::Integer(0),
                        "Flag to append also the error field to the output file");

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