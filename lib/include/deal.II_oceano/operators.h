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
 * The following classes implement the operation of the action of the inverse of a
 * @ref GlossMassMatrix "mass matrix" for a DGQ element.
 * It is useful because the original class `CellwiseInverseMassMatrix` provided by
 * the library is valid only for the special case of an evaluation object with as
 * many quadrature points as there are cell degrees of freedom. Here you can use a
 * generic quadrature formula. It is thus more flexible. Moreover we tailor the
 * singular case specificly to wet-dry applications. That is track fully dry cells
 * and we set the output array to zero (nothing moves in fully dry cells).
 *
*/

namespace MatrixFreeOperatorsOceano
{

  using namespace dealii;

 /**
  * The first class produces the exact mass matrix for DGQ elements. This algorithm
  * uses a LU factorization, implemented by the deal.ii Gauss-Jordan algorithm. For
  * the small matrices needed in Discontinuous Galerkin, the inversion cost is totally
  * acceptable. Indeed, the main problem is that the Gauss-Jordan routine does not
  * support vectorization and, at the end, it gives a noticeable overhead with respect
  * to the original deal.ii `CellwiseInverseMassMatrix`.
  *
  */
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
     * Applies the inverse @ref GlossMassMatrix "mass matrix" operation on an input array and
     * assign it to an output array. It is assumed that the passed input and output pointers are
     * of correct size, namely FEEvaluation::dofs_per_cell long for input/output and
     * FEEvaluation::dofs_per_cell power two for the mass matrix. We track the singular case
     * with a simple if statement that checks a mask passed as argument. The default behaviour
     * does not check the mask.
     * To call this function vectorized lanes must be unrolled, as we pass the cell index in the
     * batch lane in the last argument.
     */
    void apply(const VectorizedArrayType *mass_array,
               const VectorizedArrayType *in_array,
               VectorizedArrayType       *out_array,
               const unsigned int         cell_in_lane,
               const VectorizedArrayType  mask = 100) const;

  private:
    /**
     * A reference to the FEEvaluation object for getting info on the cell.
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
      apply(const VectorizedArrayType *mass_array,
            const VectorizedArrayType *in_array,
            VectorizedArrayType       *out_array,
            const unsigned int         cell_in_lane,
            const VectorizedArrayType  mask) const
  {
    const unsigned int given_degree =
        (fe_degree > -1) ? fe_degree :
                           fe_eval.get_shape_info().data.front().fe_degree;

    const unsigned int dofs_per_component =
        Utilities::pow(given_degree + 1, dim);

    const unsigned int n_q_points =
        fe_eval.get_shape_info().n_q_points;

    FullMatrix<Number> cell_matrix(dofs_per_component, dofs_per_component);
    Vector<Number> cell_src(dofs_per_component),
                   cell_dst(dofs_per_component);

    if (mask[cell_in_lane] < n_q_points)
      {
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
    else
      {
        for (unsigned int d = 0; d < n_components; ++d)
          for (unsigned int i = 0; i < dofs_per_component; ++i)
            out_array[i + d * dofs_per_component][cell_in_lane] = 0.;
      }
  }



 /**
  * The second class, instead of an exact matrix, it produces the lumped mass matrix for
  * DGQ elements. This algorithm simply inverts the diagonal. Note that it is only first
  * order accurate. However vectorization is supported and enables faster inversion with
  * respect to the first class `CellwiseInverseMassMatrix`.
  *
  */
  template <int dim,
            int fe_degree,
            int n_components             = 1,
            typename Number              = double,
            typename VectorizedArrayType = VectorizedArray<Number>>
  class CellwiseInverseMassMatrixLumped
  {
  public:
    /**
     * Default constructor.
     */
    CellwiseInverseMassMatrixLumped(
      const FEEvaluationBase<dim,
                             n_components,
                             Number,
                             false,
                             VectorizedArrayType> &fe_eval);

    /**
     * Applies the inverse @ref GlossMassMatrix "mass matrix" operation on an input array and
     * assign it to an output array. It is assumed that the passed input and output pointers are
     * of correct size, namely FEEvaluation::dofs_per_cell long. Note that for the mass matrix
     * we store only the diagonal. We track the singular case
     * with a `compare_and_apply_mask<SIMDComparison::less_than>` that check if the cell has
     * been masked. The default behaviour does not check the mask.
     */
    void apply(const VectorizedArrayType *mass_array,
               const VectorizedArrayType *in_array,
               VectorizedArrayType       *out_array,
               const VectorizedArrayType  mask = 100) const;

  private:
    /**
     * A reference to the FEEvaluation object for getting info on the cell.
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
  CellwiseInverseMassMatrixLumped<
    dim,
    fe_degree,
    n_components,
    Number,
    VectorizedArrayType>::CellwiseInverseMassMatrixLumped(
      const FEEvaluationBase<dim,
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
  CellwiseInverseMassMatrixLumped<
    dim,
    fe_degree,
    n_components,
    Number,
    VectorizedArrayType>::
      apply(const VectorizedArrayType *mass_array,
            const VectorizedArrayType *in_array,
            VectorizedArrayType       *out_array,
            const VectorizedArrayType  mask) const
  {
    const unsigned int given_degree =
        (fe_degree > -1) ? fe_degree :
                           fe_eval.get_shape_info().data.front().fe_degree;

    const unsigned int dofs_per_component =
        Utilities::pow(given_degree + 1, dim);

    const unsigned int n_q_points =
        fe_eval.get_shape_info().n_q_points;

    for (unsigned int d = 0; d < n_components; ++d)
      for (unsigned int i = 0; i < dofs_per_component; ++i)
        out_array[i + d * dofs_per_component] =
          compare_and_apply_mask<SIMDComparison::less_than>(
            mask, n_q_points,
            1./mass_array[i] * in_array[i + d * dofs_per_component],
            0.);
  }
} // namespace MatrixFreeOperatorsOceano

#endif //CELLWISEINVERSEMASSMATRIXOCEANO_HPP
