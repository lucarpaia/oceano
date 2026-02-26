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
#ifndef ICBC_THACKEROSCILLATIONS2D_HPP
#define ICBC_THACKEROSCILLATIONS2D_HPP

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // Thacker in 1981 found an exact solution of the shallow water equations
  // with wetting and drying. In fact, the shallow water equations admit analytic
  // periodic solutions (there is no damping) of a flood wave, eventually with Coriolis
  // effect that here is neglected. The test implemented is the is the Thacker solution
  // with a curved surface with the parameter proposed in the SWASH test suite (Delestre,2016).
  // With respect to the latter we only invert the sign of the bathymetry which is measured
  // positive downward. We have added an option to compare the solution of
  // our code, with variable bathyemtry at the grid level, with a standard Finite Volume
  // with piecewice constant bathymetry.
  // In that case we have to read a piecewice constant bathymetry per cell and the
  // code must be slightly changed. You should activate the following preprocessor:
#undef  ICBC_THACKEROSCILLATIONS2D_FINITEVOLUME

  using namespace dealii;
  
  // We define constant parameters that help in the definition of the initial
  // and boundary conditions. We use the same notation and parameter value of the
  // SWASH test case.
  constexpr double h0      = 0.1;
  constexpr double a       = 1.0;
  constexpr double radius0 = 0.8;
  constexpr double L       = 3.0;



  // @sect3{Equation data}
  //
  // We need a class to handle the problem data. Problem data are case dependent; for this 
  // reason it appears inside the `ICBC` namespace. The data in general depends on
  // both time and space. Deal.II has a class `Function` which returns function
  // of space and time, thus we simply create a derived class. The size of the data is 
  // fixed to `dim+3=5` scalar quantities. The first component is the bathymetry. 
  // The second is the bottom friction coefficient. The call to `value()` returns all the
  // external data necessary to complete the computation.
  //
  // Finally the parameter handler class allows to read constants from the prm file.
  // The parameter handler class may seems redundant but it is not! Constants that appears
  // in you data may be easily recovered from the configuration file. More important file 
  // names which contains the may be imported too. 
#ifndef ICBC_THACKEROSCILLATIONS2D_FINITEVOLUME
  template <int dim>  
  class ProblemData : public Function<dim>
  {
  public:
    ProblemData(IO::ParameterHandler &prm);
    ~ProblemData(){};

    inline double thackerOscillations2d_bathymetry(const Point<dim> & p) const;

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  ProblemData<dim>::ProblemData(IO::ParameterHandler &/*prm*/)
    : Function<dim>(dim+3)
  {}


  template <int dim>
  inline double ProblemData<dim>::thackerOscillations2d_bathymetry(
    const Point<dim> & x) const
  {
    Point<dim> x0;
    x0[0] = 0.5 *L;
    x0[1] = 0.5 *L;
    const double radius = (x - x0).norm();
    const double inv_a = 1./a;

    return h0 * (1. - inv_a*inv_a * radius*radius);
  }

#else
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
  {}


  template <int dim>
  std::string ProblemData<dim>::bathymetry_filename(IO::ParameterHandler &prm) const
  {
    prm.enter_subsection("Input data files");
    std::string filename = prm.get("Bathymetry_filename");
    prm.leave_subsection();

    return filename;
  }
#endif


  template <int dim>
  double ProblemData<dim>::value(const Point<dim> & x,
                                 const unsigned int component) const
  {
    if (component == 0)
#ifdef ICBC_THACKEROSCILLATIONS2D_FINITEVOLUME
      return bathymetry_data.value(x);
#else
      return thackerOscillations2d_bathymetry(x);
#endif
    else
      return 0.0;
  }



  // The class `ExactSolution` defines analytical functions that can be useful
  // to define initial and boundary conditions. Apart for the template for the
  // dimension which is in common with the base `Function` class, we have added
  // the number of variables.
  template <int dim, int n_vars>  
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double time,
                  IO::ParameterHandler &prm)
      : Function<dim>(n_vars, time)
    {
      prm.enter_subsection("Physical constants");
      g = prm.get_double("g");
      prm.leave_subsection();
    }

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private: 
    double g;
  };  

  // We return either the water depth or the momentum
  // depending on which component is requested. The initialization
  // is different if the Finite Volume method is chosen. In that
  // case, dry cells must be initialized at the bathymetry level,
  // in order to start to flood from that level.
  // For our non-linear approach with variable bathymetry we initialize
  // dry cells with a virtual free-surface under the bathymetry level
  // and it is up to the non-linear algorithm to find the "good" water
  // level.
  template <int dim, int n_vars>
  double ExactSolution<dim, n_vars>::value(const Point<dim> & x,
                                           const unsigned int component) const
  {
    const double t = this->get_time();

    Assert(dim == 2, ExcNotImplemented());        

    Point<dim> x0;
    x0[0] = 0.5 *L;
    x0[1] = 0.5 *L;
    const double radius = (x - x0).norm();

    const double inv_a = 1./a;
    const double omega = std::sqrt(8.*g*h0) * inv_a;
    const double a2 = a*a;
    const double radius02 = radius0*radius0;
    const double radius2 = radius*radius;
    const double A = (a2 - radius02)/(a2 + radius02);
    const double denom = 1./(1. - A * std::cos(omega*t));
    const double tmp = std::sqrt(1.0 - A*A) * denom;
    const double inv_temp = 1./tmp;

    double zb;
    double h;
    double u;
    double v;
    if (radius > a * std::sqrt(inv_temp))
      {
#ifdef ICBC_THACKEROSCILLATIONS2D_FINITEVOLUME
        const double r2 = radius2;
#else
        const double r2 = a2 * inv_temp;
#endif
        zb = h0 * (1. - inv_a*inv_a * r2);
        h = 0.;
        u = 0.;
        v = 0.;
      }
    else
      {
        zb = h0 * (1. - inv_a*inv_a * radius2);
        h = h0 * (tmp - 1. - radius2*inv_a*inv_a *(tmp*tmp - 1.)) + zb;
        u = denom * 0.5 * omega * (x[0]-x0[0]) * A * std::sin(omega*t);
        v = denom * 0.5 * omega * (x[1]-x0[1]) * A * std::sin(omega*t);
      }    

    if (component == 0)
      return h - zb;
    else if (component == 1)
      return u;
    else if (component == 2)
      return v;
    else
      return 1.;
  }



  // The `Ic` and `Bc` classes define the initial/boundary condition for the
  // test-case. They are very similar in the templates and the constructor.
  // They both take as argument the parameter class and they stored it
  // internally. This means that we can read the Parameter file from
  // anywhere when we are implementing ic/bc and we can access constants or
  // filenames from which the initial/boundary data depends.
  // In this case the intial condition is recovered from the exact solution 
  // at time zero. This is realized here thanks to a derived class of
  // `ExactSolution` that overload the the constructor of the base class 
  // providing automatically a zero time.
  // The wave does not interact with the boundaries so we can safely consider
  // wall boundaries.
  template <int dim, int n_vars>
  class Ic : public ExactSolution<dim, n_vars>
  {
  public:
    Ic(IO::ParameterHandler &prm)
      : ExactSolution<dim, n_vars>(0.,prm)
    {}
    ~Ic(){};
  };



  template <int dim, int n_vars>  
  class BcThackerOscillations2d : public BcBase<dim, n_vars>
  {
  public:
 
    BcThackerOscillations2d(IO::ParameterHandler &prm)
      : prm(prm)
    {}
    ~BcThackerOscillations2d(){};
         
    void set_boundary_conditions() override;

  private:
    ParameterHandler &prm;
  }; 

  template <int dim, int n_vars>
  void BcThackerOscillations2d<dim, n_vars>::set_boundary_conditions()
  {
    this->set_wall_boundary(0);
  }    
} // namespace ICBC

#endif //ICBC_THACKEROSCILLATIONS2D_HPP
