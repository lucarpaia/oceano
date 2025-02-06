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
 *         Luca Arpaia,        2023
 */
#ifndef ICBC_REALISTIC_HPP
#define ICBC_REALISTIC_HPP

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // These are the the initial and boundary conditions classes for
  // realistic coastal experiments.
  // With respect to other ICBC derived classes, the classes contained
  // in this file are not specific to a test case but rather generic.
  // In fact all initial solution and boundary forcings are all
  // imported by external files that can be modified outside the
  // oceano model.

  using namespace dealii;

  // We define global parameters that help in the definition of the
  // initial and boundary conditions. For this test we just need the
  // initial water level that is always zero, all other info are read from
  // file:
  constexpr double z0      = 0.0;

  // @sect3{Equation data}
  
  // Next we implement a few classes that are useful to represent functions
  // for the initial and boundary conditions. Deal.II has a class `Function`
  // which returns function of space and time, thus we simply create a
  // derived class.
  // We specialize a few different derived Function classes to handle data.
  // What changes essentially is the number of components, the type of
  // function, i.e. analytical or from observations saved into file, and
  // the independent variable, i.e is the function time or space dependent?
  // We review these Function classes.

  // First we need to handle the problem surface data. The number of
  // components for the data is fixed to `dim+3=5` scalar quantities.
  // We list them here:
  // begin{itemize}
  // \item the first component is the bathymetry.
  // \item the second is the bottom friction coefficient.
  // \item the third and fourth components are the cartesian components
  // of the wind velocity (in order, eastward and northward).
  // \item The fifth one is the Coriolis parameter.
  // end{itemize}
  // Surface data in general depends on both time and space, although for
  // now we have only implemented the space dependence. As you may see the
  // time variable is never retrieved.
  //
  // In some simplified context we may also want to assign constant value
  // to the data. The parameter handler class may seems redundant but it is
  // not! Constants that appears in you data may be easily recovered from
  // the configuration file.
  //
  // More often the data is space-varying and need to be interpolated from
  // a reference field given on a fine mesh. Deal.ii has a special `Function`
  // class that computes the values by (bi-, tri-)linear interpolation from
  // a set of point data that are arranged on a uniformly spaced tensor
  // product mesh. It is called `InterpolatedUniformGridData`. Considering
  // the two-dimensional case, let there be points $x_0,\,...,\,x_{K−1}$ that
  // result from a uniform subdivision of the interval $\[x_0,x_{K−1}\]$
  // into $K−1$ sub-intervals of size $\Delta x=\frac{x_{K−1}−x_0}/{K−1}$,
  // and similarly $y_0,\,...,\,y_{L−1}$. Also consider bathymetry data
  // $z_{kl}$ defined at point $\left(x_k,y_l\right)^T$, then evaluating the
  // function at a point $x=(x,y,z)$ will find the box so that
  // $x_k\le x \le x_{k+1},\, y_l \le y \le y_{l+1}$ and do a bilinear
  // interpolation of the data on this cell. Let us talk about the constructor
  // of this class. It takes as argument the interval_endpoints, the number
  // of subintervals in each coordinate direction and a `dim`-dimensional
  // table of data at each of the mesh points defined by the
  // coordinate arrays above. We have also used an auxiliary class that reads
  // the file. Thanks to this class we open, read and store the header lines,
  // then read the data values and close the file.
  //
  // Last, a note on the `value()` method that actually do the job,
  // specialize the base class method and return all the external data
  // necessary to complete the computation.
  template <int dim>  
  class ProblemData : public Function<dim>
  {
  public:
    ProblemData(IO::ParameterHandler &prm);
    ~ProblemData(){};

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    std::string bathymetry_filename(IO::ParameterHandler &prm) const;
    IO::TxtDataReader<dim> bathymetry_data_reader;
    const Functions::InterpolatedUniformGridData<dim> bathymetry_data;

    double cf;
  };

  template <int dim>
  ProblemData<dim>::ProblemData(IO::ParameterHandler &prm)
    : Function<dim>(dim+3)
    , bathymetry_data_reader(bathymetry_filename(prm))
    , bathymetry_data(
        bathymetry_data_reader.endpoints,
        bathymetry_data_reader.n_intervals,
        Table<dim, double>(bathymetry_data_reader.n_intervals.front()+1,
                           bathymetry_data_reader.n_intervals.back()+1,
                           bathymetry_data_reader.get_data(bathymetry_data_reader.filename).begin()))
  {
    prm.enter_subsection("Physical constants");
    cf = prm.get_double("bottom_friction");
    prm.leave_subsection();
  }



  template <int dim>
  std::string ProblemData<dim>::bathymetry_filename(IO::ParameterHandler &prm) const
  {
    prm.enter_subsection("Input data files");
    std::string filename = prm.get("Bathymetry_filename");
    prm.leave_subsection();

    return filename;
  }

  template <int dim>
  double ProblemData<dim>::value(const Point<dim>  &x,
                                 const unsigned int component) const
  {
    if (component == 0)
      return bathymetry_data.value(x);
    else if (component == 1)
      return cf;
    else
      return 0.0;
  }



  // Apart from surface data we have the boundary conditions data.
  // Boundary data has `n_var` components at maximum. In practice in
  // coastal simulation we often use subcritical boundaries that we
  // need less conditions, thus less external data, but
  // we keep the function general so we have hard-coded `n_var`
  // components.
  // In coastal simulations of small semi-enclosed basin, the boundary
  // data can be considered constant over the boundary and only
  // time-varying. For this reason, for now, it is represented as a
  // one dimensional function of time only. The function is constructed
  // taking as input the initial time which is not necessary zero and,
  // in this way, it can be set externally by the user.
  template <int dim, int n_vars>  
  class BoundaryData : public Function<dim>
  {
  public:
    BoundaryData(const double                   time,
                 const std::vector<std::string> boundary_filenames);
    ~BoundaryData(){};

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    std::array<std::pair<std::string,int>, n_vars> bc_type;
    std::vector<Functions::InterpolatedUniformGridData<1>> bc_data;
    std::vector<Functions::ConstantFunction<1>> bc_constant;
  };

  template <int dim, int n_vars>
  BoundaryData<dim, n_vars>::BoundaryData(const double                   time,
                                          const std::vector<std::string> boundary_filenames)
    : Function<dim>(n_vars, time)
  {
    unsigned int count_constant = 0;
    unsigned int count_data = 0;
    for (unsigned int v = 0; v < n_vars; ++v)
      {
        if (!boundary_filenames[v].empty())
          {
            try
              {
                const double v0 = std::stod(boundary_filenames[v]);
                bc_constant.push_back(Functions::ConstantFunction<1>(v0));
                bc_type[v] = {"constant", count_constant};
                count_constant++;
              }
            catch (...)
              {
                IO::TxtDataReader<1> bc_data_reader(boundary_filenames[v]);
                bc_data.push_back(
                Functions::InterpolatedUniformGridData<1>(
                  bc_data_reader.endpoints,
                  bc_data_reader.n_intervals,
                  Table<1, double>(bc_data_reader.n_intervals.front()+1,
                                   bc_data_reader.get_data(bc_data_reader.filename).begin())) );
                bc_type[v] = {"data", count_data};
                count_data++;
              }
          }
      }
  }

  template <int dim, int n_vars>
  double BoundaryData<dim, n_vars>::value(const Point<dim>  &/*x*/,
                                          const unsigned int component) const
  {
    Point<1> t;
    t[0] = this->get_time();

    std::pair<std::string,int> type_and_count = bc_type[component];
    if (type_and_count.first == "constant")
      return bc_constant[type_and_count.second].value(t);
    else
      return bc_data[type_and_count.second].value(t);
  }



  // The class `ExactSolution` defines a reference functions that can be used
  // for diagnostic, for example to measure the errors with respect ot another
  // model. For the realistic test we do not have a reference solution, so we use
  // a simple water at rest state.
  template <int dim, int n_vars>  
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double          time,
                  IO::ParameterHandler &/*prm*/)
      : Function<dim>(n_vars, time)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };  

  template <int dim, int n_vars>
  double ExactSolution<dim, n_vars>::value(const Point<dim> & /*x*/,
                                           const unsigned int component) const
  {
    if (component == 0)
      return z0;
    else
      return 0.;
  }



  // The `Ic` and `Bc` classes define the initial/boundary condition for the
  // test-case. They are very similar in the templates and the constructor.
  // They both take as argument the parameter class and they stored it
  // internally. This means that we can read the Parameter file from
  // anywhere when we are implementing ic/bc and we can access constants or
  // filenames from which the initial/boundary data depends.
  //
  // The initial condition for the realistic test is taken from files. This
  // is done, as usual, with the class `Functions::InterpolatedUniformGridData`
  // , but this time organized in a vector where each component represents the
  // initial data for a prognostic variable.
  // To simplify the setting, the user can impose constant initial functions.
  // If no initial file is found, the code below automatically switch to the
  // class `Functions::ConstantFunction`, also organized in a vector.
  // If no constant data is given then a water-at-rest with unit tracers
  // will be the initial condition.
  template <int dim, int n_vars>
  class Ic : public Function<dim>
  {
  public:
    Ic(IO::ParameterHandler &prm);
    ~Ic(){};

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    std::array<std::pair<std::string,int>, n_vars> ic_type;
    std::vector<Functions::InterpolatedUniformGridData<dim>> ic_data;
    std::vector<Functions::ConstantFunction<dim>> ic_constant;
  };

  template <int dim, int n_vars>
  Ic<dim, n_vars>::Ic(IO::ParameterHandler &prm)
    : Function<dim>(n_vars, 0.)
  {
    std::array<std::string,n_vars> ic_vars_name;
    ic_vars_name[0] = "Initial_freesurface";
    ic_vars_name[1] = "Initial_discharge_x";
    ic_vars_name[2] = "Initial_discharge_y";
    for (unsigned int t = 0; t < n_vars-1-dim; ++t)
      ic_vars_name[dim+1+t] = "Initial_tracer_"+std::to_string(t+1);

    prm.enter_subsection("Input data files");
    unsigned int count_constant = 0;
    unsigned int count_data = 0;
    for (unsigned int v = 0; v < n_vars; ++v)
      {
        std::string filename = prm.get(ic_vars_name[v]+"_filename");
        if (filename == "There is no entry "+ic_vars_name[v]+"_filename in the parameter file")
          {
            const double v0 = prm.get_double(ic_vars_name[v]+"_value");
            ic_constant.push_back(Functions::ConstantFunction<dim>(v0));
            ic_type[v] = {"constant", count_constant};
            count_constant++;
          }
        else
          {
            IO::TxtDataReader<dim> ic_data_reader(filename);
            ic_data.push_back(
              Functions::InterpolatedUniformGridData<dim>(
                ic_data_reader.endpoints,
                ic_data_reader.n_intervals,
                Table<dim, double>(ic_data_reader.n_intervals.front()+1,
                                   ic_data_reader.n_intervals.back()+1,
                                   ic_data_reader.get_data(ic_data_reader.filename).begin())) );
            ic_type[v] = {"data", count_data};
            count_data++;
          }
      }
    prm.leave_subsection();
  }

  template <int dim, int n_vars>
  double Ic<dim, n_vars>::value(const Point<dim>  &x,
                                const unsigned int component) const
  {
    std::pair<std::string,int> type_and_count = ic_type[component];
    if (type_and_count.first == "constant")
      return ic_constant[type_and_count.second].value(x);
    else
      return ic_data[type_and_count.second].value(x);
  }



  // The boundary conditions are specified in the gmsh file, as flags
  // that identifies Physical Curves, as well as in the parameter file.
  // There, in an appropriate section, each flag must be associated to a
  // boundary condition (type and value of the boundary data). Within
  // the BcRealistic class, to set the boundary conditions, we have to
  // perfrom this association. This is done with a map whose key is the
  // boundary flag and the arguments are two strings: one for the boundary
  // condition types and one for the file that contains the boundary data.
  // After entering the parameter file a loop reads the boundary
  // information, if no boundary information is found an exception is
  // throw. That's not the end.
  // This class uses another class BoundaryData. Each time a boundary
  // condition that need some data (not the wall for instance but for
  // example the tidal value for an open boundary or the discharge value
  // for an upstream river boundary), we instantiate one new BoundaryData
  // class to handle this data.
  template <int dim, int n_vars>  
  class BcRealistic : public BcBase<dim, n_vars>
  {
  public:
  
    BcRealistic(IO::ParameterHandler &prm);
    ~BcRealistic(){};

    void set_boundary_conditions() override;

  private:
    ParameterHandler &prm;
  };
  
  template <int dim, int n_vars>
  BcRealistic<dim, n_vars>::BcRealistic(IO::ParameterHandler &prm)
    : prm(prm)
  {}

  template <int dim, int n_vars>
  void BcRealistic<dim, n_vars>::set_boundary_conditions()
  {
    std::map<types::boundary_id,
             std::pair<std::string, std::vector<std::string>>> boundaryId;
    std::vector<std::string> info;
    bool found_boundary = false;

    prm.enter_subsection("Boundary conditions");
    for (unsigned int num = 1; num < 20; ++num)
      {
         std::string boundary_info = prm.get("Boundary_"+std::to_string(num));
         if (boundary_info == "There is no entry Boundary in the parameter file")
           {
             continue;
           }
         else
           {
             found_boundary = true;
           }

         boost::split(info, boundary_info, boost::is_any_of(":"));
         if (info.size() == 2)
           {
             boundaryId[std::stoi(info[0])] = {info[1], {"no_file_needed"}};
             AssertThrow(info[1] == "wall",
               ExcMessage("In the parameter file you have not specified the boundary data.\n"
                          "Add:\n"
                          "set Boundary_x = "+info[0]+":"+info[1]+":filename.txt.gz\n"));
           }
         else if (info.size() == n_vars+2)
           {
             std::vector<std::string> boundary_filenames;
             for (unsigned int v = 0; v < n_vars; ++v)
               boundary_filenames.push_back(info[2+v]);
             boundaryId[std::stoi(info[0])] = {info[1], boundary_filenames};
           }
         else
           {
             AssertThrow(info.size() == n_vars+2,
               ExcMessage("In the parameter file you have not specified the boundary correctly.\n"
                          "Please check that:\n"
                          "set Boundary_x = "+info[0]+":"+info[1]+":filename_1.txt.gz:...:filename_nvar.txt.gz\n"));
           }
      }
    prm.leave_subsection();
    AssertThrow(found_boundary == true,
      ExcMessage("In the parameter file you have not specified the entry:\n"
                 "set Boundary_x = ...\n"
                 "for any boundary id.\n"
                 "Fill this entry for each boundary id"));

    for (auto &i : boundaryId)
      {
        std::pair<std::string, std::vector<std::string>> type_and_filenames;
        type_and_filenames = i.second;
        if ( type_and_filenames.first == "wall")
          this->set_wall_boundary(i.first);
        else if (type_and_filenames.first == "height_inflow")
          this->set_height_inflow_boundary(
            i.first, std::make_unique<BoundaryData<dim, n_vars>>(0, type_and_filenames.second));
        else if (type_and_filenames.first == "discharge_inflow")
          this->set_discharge_inflow_boundary(
            i.first, std::make_unique<BoundaryData<dim, n_vars>>(0, type_and_filenames.second));
        else if (type_and_filenames.first == "absorbing_outflow")
          this->set_absorbing_outflow_boundary(
            i.first, std::make_unique<BoundaryData<dim, n_vars>>(0, type_and_filenames.second));
      }
  }
} // namespace ICBC

#endif //ICBC_REALISTIC_HPP
