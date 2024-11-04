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
 * Author: Luca Arpaia,	CNR-ISMAR,                2023
 */

/**
 * Namespace containing the input/output file handler.
 */

namespace IO
{

  using namespace dealii;


  
  // The next class is responsible for parsing correctly the command line.
  class CommandLineParser
  {
  public:
    CommandLineParser(){}
    void parse_command_line(int argc, char **argv);
 
  private:
    void usage();
  };
 


  void CommandLineParser::usage()
  {
    std::cout << "Oceano usage:" << std::endl;
    std::cout << " -i   --input <file>    :"
              << " use a configuration file in deal.II PRM format as input."
              << std::endl;
    std::cout << " -h   --help            :" << " print this message " 
              << std::endl;
  }
 
  // The function `parse_command_line` does not output anything. It just 
  // raises messages errors and exit the program in case of wrong usage
  // of the executable.
  void CommandLineParser::parse_command_line(int argc, char **argv)
  {
    if (argc == 1)
      {
        usage();
        AssertThrow(false,
                    ExcMessage("No configuration file was given.\n"
                               "Check again your command line."));
      }
    std::string option(argv[1]);
    if ( option == "-h" || option == "--help" )
      {
        usage();
        std::exit(1);
      }
    else if ( option == "-i" || option == "--input" )
      {
        if (argc == 2)
          {
            usage();
            AssertThrow(false,
                        ExcMessage("No configuration file was given after -i\n"
                                   "Check again your command line."));
         }
      }
    else
      {
        usage();
        AssertThrow(false,
                    ExcMessage("It was not possible to read the "
                               "configuration file.\n"
                               "Check again your command line."));
      }
  }
  
} // namespace IO
