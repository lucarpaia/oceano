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
#ifndef CELLDATASTORAGE_HPP
#define CELLDATASTORAGE_HPP

/**
 * A class for storing at each cell a vector of object of length
 * `number_of_points_per_cell` of type `DataType`. This is an adaptation
 * of the the deal.ii class `CellDataStorage` which enables vectorization.
 * The main difference is the data access which does not occurs with an
 * iterator but with a simpler index. In case of vecotrization, that is the
 * objects are of type `VectorizedArray<Number>`, this index batches
 * together multiple cells. For our our SIMD implementation this
 * class should be more efficient than the original one.
 */
template <typename DataType>
class CellDataStorage
{
public:
  /**
   * Default constructor.
   */
  CellDataStorage(){};

  /**
   * Default destructor.
   */
  ~CellDataStorage(){};

  /**
   * Initialize class members, in particular the number of objects to be
   * stored for each cell. It clears also all the data stored in this object.
   * This function has to be called once at the beginning for static runs.
   * For dynamic runs it has to be called after evrey mesh refinement to
   * update the stored objects on the new grid.
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
   */
  void submit_data(const DataType data_point)
  {
    data_cell.push_back(data_point);
  }

  /**
   * Get a point of the data located at the cell defined by the index `cell`
   * and at the point defined by the index `q`. It extract an object of type
   * `DataType`.
   */
  DataType get_data(const unsigned int cell, const unsigned int q) const
  {
    return data_cell[number_of_points_per_cell*cell + q];
  }

private:
  std::vector<DataType> data_cell;
  unsigned int number_of_points_per_cell;
};
#endif //CELLDATASTORAGE_HPP
