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
 * Author: Luca Arpaia,        2025
 */
#ifndef VISCOSITYCOEFFICIENT_HPP
#define VISCOSITYCOEFFICIENT_HPP

/**
 * Namespace containing the so-called phyisics of the governing equations.
 * They are the different parametrizations.
 */

namespace Physics
{

  using namespace dealii;

  // @sect3{Implementation of the horizontal viscosity coefficient}

  // In the following classes, we implement the horizontal viscosity.
  // The horizontal viscosity can be the molecular one but, at the scale
  // at which coastal models operates, it will be more often an eddy viscosity.
  // In coastal flows, strong shear currents maybe under-resolved. This fact,
  // toghether with the low diffusion offered by high order finite elements may lead
  // to spurious modes at the grid scale. These modes can be cured either with
  // standard TVD schemes or with an horizontal "eddy" viscosity. At state of 
  // the art, a rigorous theory to model eddy viscosity in coastal flow is missing.
  // Coastal models use either constant viscosity or simple turbulence model based
  // on the mixing length.
  // The next class and its derived classes, basically compute the eddy viscosity. 
  class ViscosityCoefficientBase
  {
  public:
    ViscosityCoefficientBase(IO::ParameterHandler &prm);
    ~ViscosityCoefficientBase(){};

    double c_m;

    // The next function is the one that actually computes the viscosity coefficient.
    // It is overloaded by the same function defined in the derived classes.
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value(const Tensor<dim, dim, Number> &gradient_velocity,
            const Number                    area) const;
  };
  
  // Not surprisingly the constructor of the base class takes as arguments 
  // only the parameters handler class in order to read the physical constants
  // from the prm file.
  ViscosityCoefficientBase::ViscosityCoefficientBase(
    IO::ParameterHandler &prm)
  {
    prm.enter_subsection("Physical constants");
    c_m = prm.get_double("horizontal_viscosity");
    prm.leave_subsection();
  }



#if defined PHYSICS_VISCOSITYCOEFFICIENTCONSTANT
  // The first formulation of the viscosity coefficient is a constant
  // one, the value being read from the configuration file.
  // The proper choice of the coefficient, named here $C_M$ can be
  // determined by a sensitivity study.
  class ViscosityCoefficientConstant : public ViscosityCoefficientBase
  {
  public:
    ViscosityCoefficientConstant(IO::ParameterHandler &prm);
    ~ViscosityCoefficientConstant(){};

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value(const Tensor<dim, dim, Number> &gradient_velocity,
            const Number                    area) const;
  };

  ViscosityCoefficientConstant::ViscosityCoefficientConstant(
    IO::ParameterHandler &param)
    : ViscosityCoefficientBase(param)
  {}
  
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ViscosityCoefficientConstant::value(
      const Tensor<dim, dim, Number> &/*gradient_velocity*/,
      const Number                    /*area*/) const
  {
    return c_m;
  }



#elif defined PHYSICS_VISCOSITYCOEFFICIENTSMAGORINSKY
  // Alternatively a turbulent closure based on the mixed length was
  // proposed by Smagorinsky. The horizontal eddy viscosity is
  // composed, based on dimensional argument, with a length scale
  // which is taken proportional to the grid scale, thanks to
  // a coefficient $C_S$ and with a characteristic strain rate of the
  // resolved flow $\mathcal{S}$. Rewriting the length scale in terms
  // of grid cell area, we get the following formula:
  // \[
  // \nu = C_M |K| \mathcal{S}
  // \]
  // with $C_M=C_S^2$. The parameter $C_M$ is read from the
  // configuration file. If you take $C_S=0.5$ and a Prandtl number
  // of $Pr=10$ then we can select a value of `c_m=0.025`.
  // The strain rate is defined in the book (Turbulent Flows, Pope):
  // \[
  // \mathcal{S} = \sqrt(2*S_{ij}*S_{ij}),
  // \quad S_{ij} = \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j}
  //              + \frac{\partial u_j}{\partial x_i}  \right)
  // \]
  // with these definitions, after some calculation, we get the same
  // formula coded below which is the same one implemented in the coastal
  // ocean model SHYFEM (Umgiesser,2004).
  class ViscosityCoefficientSmagorinsky : public ViscosityCoefficientBase
  {
  public:
    ViscosityCoefficientSmagorinsky(IO::ParameterHandler &prm);
    ~ViscosityCoefficientSmagorinsky(){};

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value(const Tensor<dim, dim, Number> &gradient_velocity,
            const Number                    area) const;
  };

  ViscosityCoefficientSmagorinsky::ViscosityCoefficientSmagorinsky(
    IO::ParameterHandler &param)
    : ViscosityCoefficientBase(param)
  {}

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ViscosityCoefficientSmagorinsky::value(
      const Tensor<dim, dim, Number> &gradient_velocity,
      const Number                    area) const
  {
    Number strain_rate = gradient_velocity[0][0]*gradient_velocity[0][0];
    for (unsigned int d = 1; d < dim; ++d)
      {
        Number shear = gradient_velocity[0][d] + gradient_velocity[d][0];
        strain_rate += gradient_velocity[d][d]*gradient_velocity[d][d] + 0.5*shear*shear;
      }

    return c_m * area * std::sqrt(2.*strain_rate);
  }
#endif

} // namespace Physics
#endif //VISCOSITYCOEFFICIENT_HPP
