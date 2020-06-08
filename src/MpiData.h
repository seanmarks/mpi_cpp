// MpiData - wraps an MPI buffer and its MPI_Datatype

#pragma once
#ifndef MPI_DATA_H
#define MPI_DATA_H

#include "MpiEnvironment.h"

#include <array>
#include <complex>
#include <vector>

#include "MpiDataTraits.h"
#include "MpiDatatype.h"
#include "utils.h"

// Helper class
class MpiDatatypeRegistry;

// MpiData - wrapper around an object representing an MPI buffer
// - Largely a pass-through class that provides a uniform interface for MpiCommunicator functions
//
// Wrapper around several variables that define an MPI buffer
// - A "data element" in the buffer is a single MPI_Datatype
// - The difference between "size" and "capacity" has the same semantics
//   as the difference between std::vector's size and capacity
//
// - Construct using committed MPI_Datatypes (not the wrapper class, MpiDatatype)
//   - Decouples construction of buffers from the registration of types
//   - MpiDatatype construction/destruction also causes commit/free operations
//
// - Note: need to pass an MpiDatatypeRegistry to constructors because it contains
//         the mappings from T --> MPI_Datatype/MpiDatatype
//
// - DEVELOPMENT (TODO)
//   - Specifying the MPI_Datatype in different ways
//   - Rename to MpiBuffer?
//
template<typename Container>
class MpiData
{
 public:
	// FIXME
	//using value_type = typename Container::value_type;
	//using size_type  = typename Container::size_type;

	// Default: empty
	// - TODO: delete?
	MpiData() {}

	MpiData(
		Container& data,
		MpiDatatypeRegistry& registry
	);

	void* data() {
		return static_cast<void*>( MpiDataTraits<Container>::data(*object_ptr_) );
	}
	const void* data() const {
		return static_cast<void*>( MpiDataTraits<Container>::data(*object_ptr_) );
	}

	int size() const {
		return MpiDataTraits<Container>::size(*object_ptr_);
	}

	// TODO: remove?
	// - For C++ containers like a std::vector, not all of the capacity is "useful"
	//   - Resizing past the current size (even if a reallocation doesn't occur) causes
	//     the elements from old_size to new_size to be default-constructed, overwriting
	//     any data that would have been received
	/*
	int capacity() const {
		return MpiDataTraits<Container>::capacity(*object_ptr_);
		//return capacity_;
	}
	*/

	void resize(const int new_size) {
		return MpiDataTraits<Container>::resize(*object_ptr_, new_size);
	}

	const MPI_Datatype& getDatatype() const {
		return data_type_;
	}

	bool isNull() const {
		return (object_ptr_ == nullptr);
	}
	bool isInitialized() const {
		return ( ! isNull() );
	}

 private:
	// Underlying buffer
	Container* object_ptr_ = nullptr;  // TODO: better name

	// Type of data stored
	MPI_Datatype data_type_ = MPI_DATATYPE_NULL;
};


#include "MpiDatatypeRegistry.h"


template<typename Container>
MpiData<Container>::MpiData(
	Container& data,
	MpiDatatypeRegistry& registry
):
	object_ptr_( &data ),
	data_type_( registry.mapMpiDatatype<typename MpiDataTraits<Container>::value_type>().get_MPI_Datatype() )
{}

#endif /* MPI_DATA_H */
