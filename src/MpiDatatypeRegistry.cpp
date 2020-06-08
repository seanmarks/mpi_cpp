#include "MpiDatatypeRegistry.h"


MpiDatatypeRegistry::MpiDatatypeRegistry()
{
	registerDefaultTypes();
}


void MpiDatatypeRegistry::registerDefaultTypes()
{
	// First, register MPI primitives. These will be used to build derived datatypes.
	registerPrimitiveTypes();

	// Register a few commonly-used derived types
	registerType< std::array<int,1> >();
	registerType< std::array<int,2> >();
	registerType< std::array<int,3> >();
	registerType< std::array<float,1> >();
	registerType< std::array<float,2> >();
	registerType< std::array<float,3> >();
	registerType< std::array<double,1> >();
	registerType< std::array<double,2> >();
	registerType< std::array<double,3> >();

	registerType< std::complex<float> >();
	registerType< std::complex<double> >();

	// Matrix-like types
	registerType< std::array<std::array<double,1>,1> >();
	registerType< std::array<std::array<double,2>,2> >();
	registerType< std::array<std::array<double,3>,3> >();
}


void MpiDatatypeRegistry::registerPrimitiveTypes()
{
	// TODO mappings for MPI_Byte and MPI_Packed

	registerType<float>();
	registerType<double>();
	registerType<long double>();

	registerType<int>();
	registerType<long int>();
	registerType<short int>();
	registerType<unsigned int>();
	registerType<unsigned long int>();
	registerType<unsigned short int>();

	registerType<char>();
	registerType<unsigned char>();
}
