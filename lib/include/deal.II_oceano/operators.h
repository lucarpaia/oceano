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
#ifndef CELLWISEINVERSEMASSMATRIXOCEANO_HPP
#define CELLWISEINVERSEMASSMATRIXOCEANO_HPP

#include <deal.II/matrix_free/fe_evaluation.h>

/**
 * This class implements the operation of the action of the inverse of a
 * @ref GlossMassMatrix "mass matrix" on an element for a general case.
 * It is useful because the original class provided by the library is
 * valid only for the special case of an evaluation object with as many quadrature
 * points as there are cell degrees of freedom. Here you can use a generic 
 * quadrature formula. This is more flexible but leads to a noticeble overhead.
 * As the original class it uses algorithms from FEEvaluation and produces
 * the exact mass matrix for DGQ elements. This algorithm uses tensor products of
 * 1d mass matrices and than a LU factorization.
 *
 * The equation may contain variable coefficients, so the user is required
 * to provide an array for the inverse of the local coefficient (this class
 * provide a helper method 'fill_inverse_JxW_values' to get the inverse of a
 * constant-coefficient operator). By now, the local coefficient are scalar
 * equally applied in each component.
*/

namespace MatrixFreeOperatorsOceano
{

  using namespace dealii;

  template <int dim,
            int fe_degree,
            int n_components             = 1,
            typename Number              = double,
            typename VectorizedArrayType = VectorizedArray<Number>>
  class CellwiseInverseMassMatrix
  {
  public:
    /**
     * Default constructor.
     */
    CellwiseInverseMassMatrix(
      const FEEvaluationBase<dim,
                             n_components,
                             Number,
                             false,
                             VectorizedArrayType> &fe_eval);

    /**
     * Applies the inverse @ref GlossMassMatrix "mass matrix" operation on an input array. It is
     * assumed that the passed input and output arrays are of correct size,
     * namely FEEvaluation::dofs_per_cell long. The inverse of the
     * local coefficient (also containing the inverse JxW values) must be
     * passed as first argument. Passing more than one component in the
     * coefficient is not allowed, by now. This means that each component
     * has the same coefficient. When different coefficients for each component
     * will implemented, the coefficients will be interpreted as scalar in each
     * component
     */
    void apply(const AlignedVector<VectorizedArrayType> &mass_array,
               const VectorizedArrayType                *in_array,
               VectorizedArrayType                      *out_array,
               const unsigned int                        cell_in_lane) const;

    /**
    * Applies zero to an input array. Useful to handle the case of
    * singular matrix.
    */
    void nullify(VectorizedArrayType                     *out_array,
                 const unsigned int                       cell_in_lane) const;

  private:
    /**
     * A reference to the FEEvaluation object for getting the JxW_values.
     */
    const FEEvaluationBase<dim,
                           n_components,
                           Number,
                           false,
                           VectorizedArrayType> &fe_eval;
  };



  template <int dim,
            int fe_degree,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  CellwiseInverseMassMatrix<
    dim,
    fe_degree,
    n_components,
    Number,
    VectorizedArrayType>::CellwiseInverseMassMatrix(const FEEvaluationBase<dim,
                                                    n_components,
                                                    Number,
                                                    false,
                                                    VectorizedArrayType> &fe_eval)
    : fe_eval(fe_eval)
  {}



  template <int dim,
            int fe_degree,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  void
  CellwiseInverseMassMatrix<
    dim,
    fe_degree,
    n_components,
    Number,
    VectorizedArrayType>::
      apply(const AlignedVector<VectorizedArrayType> &mass_array,
            const VectorizedArrayType *in_array,
            VectorizedArrayType       *out_array,
            const unsigned int         cell_in_lane) const
  {
    const unsigned int given_degree =
        (fe_degree > -1) ? fe_degree :
                           fe_eval.get_shape_info().data.front().fe_degree;

    const unsigned int dofs_per_component =
        Utilities::pow(given_degree + 1, dim);

    FullMatrix<Number> cell_matrix(dofs_per_component, dofs_per_component);
    Vector<Number> cell_src(dofs_per_component),
                   cell_dst(dofs_per_component);

    for (unsigned int j=0; j<dofs_per_component; ++j)
      for (unsigned int i=0; i<dofs_per_component; ++i)
        cell_matrix(i,j) = mass_array[i + j * dofs_per_component][cell_in_lane];

    cell_matrix.gauss_jordan();
    for (unsigned int d = 0; d < n_components; ++d)
      {
        for (unsigned int i = 0; i < dofs_per_component; ++i)
          cell_src[i] = in_array[i + d * dofs_per_component][cell_in_lane];

        cell_matrix.vmult(cell_dst, cell_src);

        for (unsigned int i = 0; i < dofs_per_component; ++i)
          out_array[i + d * dofs_per_component][cell_in_lane] = cell_dst(i);
      }
  }



  template <int dim,
            int fe_degree,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  void
  CellwiseInverseMassMatrix<
    dim,
    fe_degree,
    n_components,
    Number,
    VectorizedArrayType>::
      nullify(VectorizedArrayType       *out_array,
              const unsigned int         cell_in_lane) const
  {
    const unsigned int given_degree =
        (fe_degree > -1) ? fe_degree :
                           fe_eval.get_shape_info().data.front().fe_degree;

    const unsigned int dofs_per_component =
        Utilities::pow(given_degree + 1, dim);

    for (unsigned int d = 0; d < n_components; ++d)
      for (unsigned int i = 0; i < dofs_per_component; ++i)
        out_array[i + d * dofs_per_component][cell_in_lane] = 0.;
  }
} // namespace MatrixFreeOperatorsOceano

#endif //CELLWISEINVERSEMASSMATRIXOCEANO_HPP
