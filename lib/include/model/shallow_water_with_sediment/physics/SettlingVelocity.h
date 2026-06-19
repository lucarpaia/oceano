/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 - 2026 by CNR-ISMAR
 *
 * This code, as the deal.II library is free software; you can use it,
 * redistribute it, and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation;
 * either version 2.1 of the License, or (at your option) any later
 * version.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Luca Arpaia, 2026
 */

#ifndef SETTLINGVELOCITY_H
#define SETTLINGVELOCITY_H

/**
 * Namespace containing suspended sediment physics parametrizations.
 */
namespace Physics
{

  using namespace dealii;

  // @sect3{Implementation of the sediment settling velocity}

  // In the following classes, we implement different formulas for the
  // particle settling velocity.
  class SettlingVelocityBase
  {
  public:
    SettlingVelocityBase() = default;
    ~SettlingVelocityBase() = default;

    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value(const Number g,
            const Number nu,
            const Number d50,
            const Number rho_water,
            const Number rho_sediment) const;
  };



  // The Stokes settling velocity calculates the constant, terminal
  // velocity of a small, spherical particle falling through a viscous
  // laminar fluid under the force of gravity:
  //
  // $$w_s = g d50^2 (s - 1) / (18 nu),  s = rho_s / rho_w.$$
  //
  // It is appropriate for small particles and low Reynolds number.
  class SettlingVelocityStokes : public SettlingVelocityBase
  {
  public:
    SettlingVelocityStokes();
    ~SettlingVelocityStokes() = default;

    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value(const Number g,
            const Number nu,
            const Number d50,
            const Number rho_water,
            const Number rho_sediment) const;
  };

  SettlingVelocityStokes::SettlingVelocityStokes()
    : SettlingVelocityBase()
  {}

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    SettlingVelocityStokes::value(
      const Number g,
      const Number nu,
      const Number d50,
      const Number rho_water,
      const Number rho_sediment) const
  {
    const Number s = Number(rho_sediment / rho_water);

    return g * d50 * d50 * (s - Number(1.0)) /
      (Number(18.0) * Number(nu));
  }
} // namespace Physics
#endif // SETTLINGVELOCITY_H
