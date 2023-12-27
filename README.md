Go into the oceano repository and create a build directory:
>> mkdir build
>> cd build
>> ccmake ../
Set the "deal.II_DIR" variable that specifies the Deal.II install directory. Then generate the Makefile
and compile with:
>> make
