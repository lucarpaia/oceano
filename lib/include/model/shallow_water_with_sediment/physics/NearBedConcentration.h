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

#ifndef NEARBEDCONCENTRATION_H
#define NEARBEDCONCENTRATION_H

/**
 * Namespace containing suspended sediment physics parametrizations.
 */
namespace Physics
{

  using namespace dealii;

  // @sect3{Implementation of the near-bed sediment concentration}

  // In the following classes, we implement different empirical formulas
  // for the near-bed concentration used in suspended sediment equation,
  // (see Sediment Transport Dynamics, Wu pag.275).
  class NearBedConcentrationBase
  {
  public:
    NearBedConcentrationBase() = default;
    ~NearBedConcentrationBase() = default;

    // The next function computes the near-bed concentration. It is
    // overloaded by the same function defined in the derived classes.
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value(const Number shields_parameter) const;
  };



  // The formula implemented here is the Zyserman and Fredsoe (1994) formula:
  //
  // $$c_b = A (theta - theta_{cr})^n / (1 + A/c_m (theta - theta_{cr})^n)$$
  //
  // for $theta > theta_{cr}$, and $c_b = 0$ otherwise.
  // Default empirical coefficients from Zyserman & Fredsoe (1994) are:
  //
  // $$A = 0.331,\quad n = 1.75,\quad c_m = 0.46\quad and\quad theta_cr = 0.0045.$$
  //
  // The Shields parameter is computed according to the bottom friction model,
  // for example for quadratic bottom friction:
  //
  // $$theta = C_D |u|^2/ ((s - 1) g d50),  s = rho_s / rho_w.$$
  class NearBedConcentrationZysermanFredsoe : public NearBedConcentrationBase
  {
  public:
    NearBedConcentrationZysermanFredsoe();
    ~NearBedConcentrationZysermanFredsoe() = default;

    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value(const Number shields_parameter) const;
  };

  NearBedConcentrationZysermanFredsoe::NearBedConcentrationZysermanFredsoe()
    : NearBedConcentrationBase()
  {}

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    NearBedConcentrationZysermanFredsoe::value(
      const Number shields_parameter) const
  {
    const Number excess_shields = std::max(shields_parameter - Number(0.0045), Number(0.0));
    const Number numerator = Number(0.331) * std::pow(excess_shields, Number(1.75));

    return numerator / (Number(1.0) + Number(0.72) * std::pow(excess_shields, Number(1.75)));
  }
} // namespace Physics
#endif // NEARBEDCONCENTRATION_H
