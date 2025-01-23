/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2007 - 2023 by the deal.II authors
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
 * Authors: Wolfgang Bangerth, Texas A&M University, 2014
 *          Luca Heltai, SISSA, 2014
 *          D. Sarah Stamps, MIT, 2014
 *	    Luca Arpaia,CNR-ISMAR, 2023
 */

// Let us start with the include files we need here.The remainder of the
// include files relate to reading the topography data from a file.
// Because the data is large, the file we read from is stored as gzip
// compressed data and we make use of some BOOST-provided functionality
// to read directly from gzipped data.
#include <deal.II/base/function_lib.h>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file.hpp>

#include <fstream>
#include <iostream>
#include <memory>

/**
 * Namespace containing the input/output file handler.
 */
namespace IO
{

  using namespace dealii;



  // @sect3{Reading data in txt format}
  //
  // The general layout of the class is discussed here.
  // Following is its declaration, including one three member functions
  // that we will need in initializing the <code>topography_data</code>
  // member variable. The class is templated with the dimension so that it
  // can be used for reading general one, two and three dimensional fields.
  // Member and methods are discussed one by one in the following.
  template <int dim>
  class TxtDataReader
  {
  public:
    TxtDataReader(std::string filename);
    ~TxtDataReader(){};

    std::string filename;

    const std::array<std::pair<double,double>, dim> endpoints;
    const std::array<unsigned int, dim> n_intervals;
    static std::vector<double> get_data(const std::string filename);

  private:
    std::array<std::pair<double,double>, dim> get_endpoints();
    std::array<unsigned int, dim> get_nintervals();
  };



  // Let us move to the implementation of the class. The interesting
  // parts of the class is the constructor. The former initializes
  // the end points of the data set we want to interpolate,
  // and the number of intervals into which the data is split.
  template <int dim>
  TxtDataReader<dim>::TxtDataReader(std::string filename)
    : filename(filename)
    , endpoints(get_endpoints())
    , n_intervals(get_nintervals())
  {}

  // The first member function fill a vector that contains the data.
  // Because the file is compressed by gzip,
  // we cannot just read it through an object of type std::ifstream, but
  // there are convenient methods in the BOOST library (see
  // http://www.boost.org) that allows us to read from compressed files
  // without first having to uncompress it on disk. The result is, basically,
  // just another input stream that, for all practical purposes, looks just like
  // the ones we always use.
  // Then an iterator to the first of the 83,600 elements
  // of a std::vector object returned by the <code>get_data()</code> function below.
  // Note that all this member functions is static because (i) they do not
  // access any member variables of the class, and (ii) because they are
  // called at a time when the object is not initialized fully anyway.
  //
  // When reading the data, we read the first $2\times dim$ lines that are the
  // header lines containing the endpoints (first $dim$ lines) and the number of
  // intervals $n_j$ in each coordinate direction (last $dim$ lines). The data
  // then has now a known size $\prod_{j=1}^{dim} n_j$.
  // The datum appears after the headers in the last column. It is appended to an
  // array that we return. Since the BOOST.iostreams
  // library does not provide a very useful exception when the input file
  // does not exist, is not readable, or does not contain the correct
  // number of data lines, we catch all exceptions it may produce and
  // create our own one. To this end, in the <code>catch</code>
  // clause, we let the program run into an <code>AssertThrow(false, ...)</code>
  // statement. Since the condition is always false, this always triggers an
  // exception. In other words, this is equivalent to writing
  // <code>throw ExcMessage("...")</code> but it also fills certain fields
  // in the exception object that will later be printed on the screen
  // identifying the function, file and line where the exception happened.
  template <int dim>
  std::vector<double>
    TxtDataReader<dim>::get_data(const std::string filename)
  {
    std::vector<double> read_data;
 
    boost::iostreams::filtering_istream in;
    in.push(boost::iostreams::basic_gzip_decompressor<>());
    in.push(boost::iostreams::file_source(filename));

    double skip_line;
    for (unsigned int line = 0; line < dim; ++line)
      {
        in >> skip_line >> skip_line;
      }
    unsigned int ni;
    unsigned int n;
    in >> n;
    n = n+1;
    for (unsigned int line = 0; line < dim-1; ++line)
      {
        in >> ni;
        n *= (ni+1);
      }


    for (unsigned int line = 0; line < n; ++line)
      {
        try
          {
            double xi, datai;
            in >> xi;
            for (unsigned int d = 0; d < dim-1; ++d)
              {
                in >> xi;
              }
            in >> datai;
            read_data.push_back(datai);
          }
        catch (...)
          {
            AssertThrow(false,
                        ExcMessage("Could not read all data points "
                                   "from the file <topography.txt.gz>!"));
          }
      }
 
    return read_data;
  }



  template <int dim>
  std::array<std::pair<double,double>, dim> TxtDataReader<dim>::get_endpoints()
  {
    std::array<std::pair<double,double>, dim> read_endpoints;
 
    boost::iostreams::filtering_istream in;
    in.push(boost::iostreams::basic_gzip_decompressor<>());
    in.push(boost::iostreams::file_source(filename));

    for (unsigned int line = 0; line < dim; ++line)
      {
        try
          {
            std::pair<double,double> endpoints;
            double endleft, endright;
            in >> endleft >> endright;
            endpoints = {endleft,endright};
 
            read_endpoints[line] = endpoints;
          }
        catch (...)
          {
            AssertThrow(false,
                        ExcMessage("Could not read the endpoints "
                                   "from the file <topography.txt.gz>!"));
          }
      }

    return read_endpoints;
  }

  template <int dim>
  std::array<unsigned int, dim> TxtDataReader<dim>::get_nintervals()
  {
    std::array<unsigned int, dim> read_nintervals;
 
    boost::iostreams::filtering_istream in;
    in.push(boost::iostreams::basic_gzip_decompressor<>());
    in.push(boost::iostreams::file_source(filename));

    double skip_line;
    for (unsigned int line = 0; line < dim; ++line)
      {
        in >> skip_line >> skip_line;
      }

    for (unsigned int line = 0; line < dim; ++line)
      {
        try
          {
            unsigned int nintervals;
            in >> nintervals;

            read_nintervals[line] = nintervals;
          }
        catch (...)
          {
            AssertThrow(false,
                        ExcMessage("Could not read the number of intervals "
                                   "from the file <topography.txt.gz>!"));
          }
      } 

    return read_nintervals;
  }
} // namespace IO
