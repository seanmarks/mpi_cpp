// MpiDatatypeRegistry: map from C++ type to MpiDatatype
// - Also controls the registration of MPI_Ops associated
//   with the registered datatypes
//
// DEVELOPMENT (TODO)
// - Expand the interface for registering types and ops

#pragma once
#ifndef MPI_DATATYPE_REGISTRY_H
#define MPI_DATATYPE_REGISTRY_H

#include <iostream>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "MpiDatatype.h"
#include "MpiEnvironment.h"
#include "MpiOp.h"

// FIXME: copying registries (unique_ptr complications?)
class MpiDatatypeRegistry
{
 public:
	using MpiDatatypePtr = std::unique_ptr<MpiDatatype>;

	MpiDatatypeRegistry();


	//-------------------------//
	//------ Registration -----//
	//-------------------------//

	// Register type T as MpiDatatype T
	template<typename T>
	void registerType(const MpiDatatype& data_type);

	// Register a new type using its MpiDatatypeRegistrar
	// - The registrar may make use of existing MpiDatatypes (such as primitives like
	//   int and double) to create the new MpiDatatype
	// - This is useful for constructing derived types for template specializations
	//   - e.g. the same MpiDatatypeRegistrar can register different std::arrays 
	// - Returns a refernce to the new type
	template<typename T>
	const MpiDatatype& registerType();


	//------------------------------//
	//----- Using the Registry -----//
	//------------------------------//

	// Maps type T to the appropriate MpiDatatype
	// - If the mapping does not already exist, a new mapping is constructed if possible.
	// - If construction is not possible, an exception is thrown.
	template<typename T>
	const MpiDatatype& mapMpiDatatype();

	// Returns the MpiOp for the given type corresponding to the standard operation
	// represented by the StandardOp enum
	template<typename T>
	const MpiOp& mapStandardMpiOp(const MpiOp::StandardOp& op_enum);


	//-----------------------//
	//------ Exceptions -----//
	//-----------------------//

	// TODO: MpiDatatypeNotFoundException

	class MpiOpNotFoundException : public std::exception {
	 public:
		MpiOpNotFoundException() {}
		const char* what() const noexcept override {
			return "Given MpiOp not found";
		};
	};

 private:
	std::unordered_map<std::type_index, MpiDatatypePtr> mpi_datatype_map_;

	// Map from type T to its registered standard MpiOps
	std::unordered_map<std::type_index, MpiOp::StandardOpsMap> standard_mpi_op_map_;

	// TODO: Make these static?
	void registerDefaultTypes();
	void registerPrimitiveTypes();
};


#include "MpiDatatypeRegistrar.h"


//------------------------------------//
//----- Type and Op Registration -----//
//------------------------------------//


template<typename T>
void MpiDatatypeRegistry::registerType(const MpiDatatype& data_type)
{
	// First, ensure the type doesn't already exist
	std::type_index ti(typeid(T));
	const auto it = mpi_datatype_map_.find(ti);
	if ( it != mpi_datatype_map_.end() ) {
		std::stringstream err_ss;
		err_ss << "Error in " << FANCY_FUNCTION << "\n"
					 << "  type \"" << ti.name() << "\" is already registered \n"
					 << "  (NOTE: the type name printed above is implementation-dependent)\n";
		throw std::runtime_error( err_ss.str() );
	}

	// Register the mapping
	mpi_datatype_map_.insert( std::make_pair(ti, MpiDatatypePtr(new MpiDatatype(data_type))) );
}


template<typename T>
const MpiDatatype& MpiDatatypeRegistry::registerType()
{
	// First, ensure the type doesn't already exist
	std::type_index ti(typeid(T));  // convert to type index
	auto it = mpi_datatype_map_.find(ti);   //std::type_index(typeid(T)) );
	if ( it == mpi_datatype_map_.end() ) {
		// Register a new type
		using Registrar = MpiDatatypeRegistrar<T>;
		auto insert_result_pair = mpi_datatype_map_.insert(
				std::make_pair(
					ti, 
					MpiDatatypePtr(new MpiDatatype(Registrar::makeMpiDatatype(*this)))
				)
		);
		const auto& new_pair_it = insert_result_pair.first;
		const MpiDatatype& new_datatype = *(new_pair_it->second);

		// Add a new sub-map for this type's standard operations
		auto insert_it = standard_mpi_op_map_.insert( std::make_pair(ti, MpiOp::StandardOpsMap()) );
		bool success = insert_it.second;
		if ( success ) {
			auto& new_pair_it = insert_it.first;  // the new pair in the outer unordered_map
			auto& new_map = new_pair_it->second;  // the new submap itself

			// Use the registrar to register relevant operations
			Registrar::registerMpiOps( new_map );
		}
		else {
			std::stringstream err_ss;
			err_ss << "Error in " << FANCY_FUNCTION << "\n"
						 << "  standard op map for type \"" << ti.name() << "\" already exists.\n"
						 << "  (NOTE: the type name printed above is implementation-dependent)\n";
			throw std::runtime_error( err_ss.str() );
		}

		return new_datatype;
	}
	else {
		// TODO: return the existing type instead?
		std::stringstream err_ss;
		err_ss << "Error in " << FANCY_FUNCTION << "\n"
					 << "  type \"" << ti.name() << "\" is already registered \n"
					 << "  (NOTE: the type name printed above is implementation-dependent)\n";
		throw std::runtime_error( err_ss.str() );
	}
}


//-------------------------------//
//----- Type and Op Mapping -----//
//-------------------------------//


template<typename T>
inline
const MpiDatatype& MpiDatatypeRegistry::mapMpiDatatype()
{
	const auto it = mpi_datatype_map_.find( std::type_index(typeid(T)) );
	if ( it != mpi_datatype_map_.end() ) {
		return *(it->second);
	}
	else {
		// Attempt to register the type
		return registerType<T>();
	}
}


template<typename T>
inline
const MpiOp& MpiDatatypeRegistry::mapStandardMpiOp(const MpiOp::StandardOp& op_enum)
{
	// First, get the StandardOp->MpiOp map for the given type
	std::type_index t_index = std::type_index(typeid(T));
	const auto pair_it = standard_mpi_op_map_.find(t_index);
	if ( pair_it == standard_mpi_op_map_.end() ) {
		std::stringstream err_ss;
		err_ss << "Error in " << FANCY_FUNCTION << "\n"
					 << "  could not find any standard MpiOps for type \"" << t_index.name() << "\n"
					 << "  (NOTE: the type name printed above is implementation-dependent)\n";
		throw std::runtime_error( err_ss.str() );
	}
	const auto& map_for_type = pair_it->second;

	// Now find the particular operation
	const auto op_it = map_for_type.find(op_enum);
	if ( op_it == map_for_type.end() ) {
		std::stringstream err_ss;
		err_ss << "Error in " << FANCY_FUNCTION << "\n"
					 << "  standard op \"" << MpiOp::getName(op_enum)
		          << "\" is not defined for type \"" << t_index.name() << "\n"
					 << "  (NOTE: the type name printed above is implementation-dependent)\n";
		throw std::runtime_error( err_ss.str() );
	}
	return op_it->second;
}


#endif // #ifndef MPI_DATATYPE_REGISTRY_H
