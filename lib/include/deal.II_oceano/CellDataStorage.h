/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2020 - 2023 by the deal.II authors
 * (GO: Are you sure that this is the right copyright at this point?. This is valid also for the other files that you wrote, not reported)
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
 * Author: Luca Arpaia, 2025
 */
#pragma once
// GO: This is a bit more modern (even though not standard C++ strictly speaking),
// but it's most a personal matter.
// Here you are saying: the file is included only once, even though it is referencied multiple times
// and it is the compiler that handles how to do it.
// With #ifndef you are saying: with the first inclusion the #define was not defined, now it is
// and in next calls, it is not included.

using namespace dealii; //GO: Is it needed?

/**
 * A class for storing at each cell a vector of object of length
 * `number_of_points_per_cell` of type `DataType`. This is an adaptation
 * of the the deal.ii class `CellDataStorage` which enables vectorization.
 * The main difference is the data access which does not occurs with an
 * iterator but with a simpler index. In case of vectorization, that is the
 * objects are of type `VectorizedArray<Number>`, this index batches
 * together multiple cells. For our SIMD implementation this
 * class should be more efficient than the original one.
 *
 * GO: @tparam DataType type of the stored data (it can be generic)
 */
template <typename DataType>
class CellDataStorage
{
public:
  /**
   * Default constructor.
   */
  CellDataStorage() = default; // GO: with default the type remains 'trivial', but again, personal choice

  /**
   * Default destructor.
   */
  ~CellDataStorage() = default;

  /**
   * Initialize class members, in particular the number of objects to be
   * stored for each cell. It clears also all the data stored in this object.
   * This function has to be called once at the beginning for static runs.
   * For dynamic runs it has to be called after every mesh refinement to
   * update the stored objects on the new grid.
   *
   * GO: @param number_of_data_points_per_cell number of points per cell
   */
  void initialize(const unsigned int number_of_data_points_per_cell)
  {
    number_of_points_per_cell = number_of_data_points_per_cell;
    data_cell.clear();
  }

  /**
   * Initialize a single point in the vector of objects.
   * This function has to be called on every point where data is to be
   * stored.
   *
   * GO: @param data_point datum to be stored
   */
  void submit_data(const DataType data_point)
  {
    data_cell.push_back(data_point);
  }

  /**
   * Get a point of the data located at the cell defined by the index `cell`
   * and at the point defined by the index `q`. It extract an object of type
   * `DataType`. To reduce overhead to a minimum, we check that we are not
   * exeeding the vector size with an `Assert` exception which is active in
   * debug mode only.
   *
   * GO: @param cell index of the cell
   * GO: @param q index of the point
   */
  DataType get_data(const unsigned int cell, const unsigned int q) const
  {
    Assert(data_cell.size() >= number_of_points_per_cell*cell + q,
      ExcMessage("The provided cell index or quadrature point does not belong"
                 "to the triangulation that corresponds to the CellDataStorage object."));
    return data_cell[number_of_points_per_cell*cell + q];
  }

private:
  std::vector<DataType> data_cell;        /*!< GO: Vector of stored objects */
  unsigned int number_of_points_per_cell; /*!< GO: Number of points per cell handled by each instance */
};
