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
#ifndef DIFFUSIONCOEFFICIENT_HPP
#define DIFFUSIONCOEFFICIENT_HPP

/**
 * Namespace containing the so-called phyisics of the governing equations.
 * They are the different parametrizations.
 */

namespace Physics
{

  using namespace dealii;

  // @sect3{Implementation of the horizontal diffusion coefficient}

  // In the following classes, we implement the horizontal diffusivity.
  // The horizontal diffusivity can be the molecular one but, at the scale
  // at which coastal models operates, it will be more often an eddy diffusivity.
  // In coastal flows, transport brings closer water masses with different
  // properties, continuously building up strong gradients. These can be
  // accurately simulated with TVD schemes or with an horizontal "eddy" diffusion.
  // The next class and its derived classes, basically compute the eddy diffusivity. 
  class DiffusionCoefficientBase
  {
  public:
    DiffusionCoefficientBase(IO::ParameterHandler &prm);
    ~DiffusionCoefficientBase(){};

    double c_m;

    // The next function is the one that actually computes the diffusion coefficient.
    // It is overloaded by the same function defined in the derived classes.
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value(const Tensor<dim, dim, Number> &gradient_discharge,
            const Number                    area) const;
  };
  
  // Not surprisingly the constructor of the base class takes as arguments 
  // only the parameters handler class in order to read the physical constants
  // from the prm file. Note that $C_M$ will have different meaning depending
  // on the diffusivity formulation.
  DiffusionCoefficientBase::DiffusionCoefficientBase(
    IO::ParameterHandler &prm)
  {
    prm.enter_subsection("Physical constants");
    c_m = prm.get_double("horizontal_diffusivity");
    prm.leave_subsection();
  }



#if defined PHYSICS_DIFFUSIONCOEFFICIENTCONSTANT
  // The first formulation of the diffusion coefficient is a constant
  // one named $C_M$, the value being read from the configuration file.
  // The proper choice of $C_M$ can be determined by a sensitivity study.
  class DiffusionCoefficientConstant : public DiffusionCoefficientBase
  {
  public:
    DiffusionCoefficientConstant(IO::ParameterHandler &prm);
    ~DiffusionCoefficientConstant(){};

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value(const Tensor<dim, dim, Number> &gradient_discharge,
            const Number                    area) const;
  };

  DiffusionCoefficientConstant::DiffusionCoefficientConstant(
    IO::ParameterHandler &param)
    : DiffusionCoefficientBase(param)
  {}
  
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    DiffusionCoefficientConstant::value(
      const Tensor<dim, dim, Number> &/*gradient_discharge*/,
      const Number                    /*area*/) const
  {
    return c_m;
  }



#elif defined PHYSICS_DIFFUSIONCOEFFICIENTSMAGORINSKY
  // Alternatively a turbulent closure based on the mixed length was
  // proposed by Smagorinsky. The horizontal eddy viscosity is
  // composed, based on dimensional argument, with a length scale
  // which is taken proportional to the grid scale thanks to
  // a coefficient $C_S$ and with a characteristic strain rate of the
  // resolved flow $\mathcal{S}$. Rewriting the length scale in terms
  // of grid area, we get the following formula:
  // \[
  // \nu = C_M |K| \mathcal{S}
  // \]
  // with $C_M=C_S^2/Pr$ scaled by the the Prandtl $Pr$ to have the
  // correct dimension. The parameter $C_M$ is read from the
  // configuration file. If you take $C_S=0.5$ and a Prandtl number
  // of $Pr=10$ then we can select the parameter as $C_M=0.025$.
  // The strain rate is defined in the book (Turbulent Flows, Pope):
  // \[
  // \mathcal{S} = \sqrt(2*S_{ij}*S_{ij}),
  // \quad S_{ij} = \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j}
  //              + \frac{\partial u_j}{\partial x_i}  \right)
  // \]
  // with these definitions, after some calculation, we get the same
  // formula coded in the coastal ocean model SHYFEM (Umgiesser,2004).
  class DiffusionCoefficientSmagorinsky : public DiffusionCoefficientBase
  {
  public:
    DiffusionCoefficientSmagorinsky(IO::ParameterHandler &prm);
    ~DiffusionCoefficientSmagorinsky(){};

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value(const Tensor<dim, dim, Number> &gradient_discharge,
            const Number                    area) const;
  };

  DiffusionCoefficientSmagorinsky::DiffusionCoefficientSmagorinsky(
    IO::ParameterHandler &param)
    : DiffusionCoefficientBase(param)
  {}

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    DiffusionCoefficientSmagorinsky::value(
      const Tensor<dim, dim, Number> &gradient_discharge,
      const Number                    area) const
  {
    Number strain_rate = gradient_discharge[0][0]*gradient_discharge[0][0];
    for (unsigned int d = 1; d < dim; ++d)
      {
        Number shear = gradient_discharge[0][d] + gradient_discharge[d][0];
        strain_rate += gradient_discharge[d][d]*gradient_discharge[d][d] + 0.5*shear*shear;
      }

    return c_m * area * std::sqrt(2.*strain_rate);
  }
#endif

} // namespace Physics
#endif //DIFFUSIONCOEFFICIENT_HPP
