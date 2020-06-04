// MpiCommunicator
//
// ABOUT: Wrapper around MPI communicator and other library functions
//	- Based heavily on the Communicator implemented in PLUMED 2.4.0 (see http://plumed.org)
//  - Contains lots of template code for commonly-used STL containers that store
//    primitive types (e.g. int, double) in contiguous memory
//  - Shields the user from many of the tedious aspects of MPI library calls
// 
// FOR THE USER:
//  - Make sure std::vector buffers are prepared using resize(), not just reserve()
//
// NOTES:
//	- Most functions (e.g. get/set, send/recv) throw if MPI is not initialized
//  - MPI routines here almost always *assume* that the receive buffers have enough
//		memory reserved to handle the maximum amount of data that may be sent at a time.
//    They do not pre-reserve space for you, but will auto-resize based on the 
//    amount of data received.
//    - They are not designed to receive data over multiple calls
//    - Exceptions:
//       - allgatherv(): calls gather() to determine how much data each rank will send,
//                       and resizes all output arrays accordingly
//
//  - send/Isend functions:
//    - Infer the number of data elements to send based on MpiData::size
//      - For a std::vector, this is taken to be its size (not capacity)
//
//  - recv functions:
//    - Set MpiData::size based on how much data was received 
//      (except if MPI_STATUS_IGNORE is requested)
//       - If MPI_STATUS_IGNORE is requested, the functions assume that
//         all available space was filled
//          - e.g. vectors are filled to capacity, irrespective of starting size
//       - If you want a vector to receive an amount of data other
//         than its current size, you *MUST* pass it an MpiStatus object, even if
//         you do nothing with it after the call
//
//  - Irecv functions: (TODO)
//    - 
//
//  - The return codes of MPI calls are not checked because the default behavior of
//    MPI is to internally check for errors before returning, and abort on error
//    - There is no guarantee that MPI can continue past an error anyway
// 
// DEVELOPMENT: TODO
//  - Isend routines
//  - Include PLMD::Vector overloads?
//  - Pointer overloads
//  - Generalize template code for std::vectors of contiguous data arrays?
//    - Such a setup would cover std::vector<T>, PLMD::Vector, std::array<T,dim>, ...
//    - Could make use of MPI's own custom type system with MPI_Type_contiguous
//      - Reference:  https://tech.io/playgrounds/349/introduction-to-mpi/custom-types

#pragma once
#ifndef MPI_COMMUNICATOR_H
#define MPI_COMMUNICATOR_H

#include <array>
#include <complex>
#include <cstdlib>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "Assert.h"
#include "MpiEnvironment.h"
#include "MpiOp.h"
#include "MpiRequest.h"
#include "MpiStatus.h"


#ifndef MPI_ENABLED
// Dummy classes for when MPI is not available
class MPI_Comm {};

class MPI_Datatype;
#endif /* MPI_ENABLED */

// Forward declarations to prevent circular dependency
class MpiData;
class MpiDatatype;
template<typename T> class MpiDatatypeRegistrar;

class MpiCommunicator
{
 public:
	using StandardOp = MpiOp::StandardOp;

	template<typename T>
	friend class MpiDatatypeRegistrar;

	//-------------------------------------------//
	//----- Constructors and Initialization -----//
	//-------------------------------------------//

	// Default constructor
	// - Sets the communicator to MPI_COMM_SELF
	// - Will do MPI_Init if necessary
	MpiCommunicator();

	// Initialize from raw MPI communicator
	// - This is the "primary" constructor
	// - Duplicates the provided communicator, creating a new context
	// - Will do MPI_Init if necessary
	MpiCommunicator(const MPI_Comm& communicator);

	// Copy constructor 
	MpiCommunicator(const MpiCommunicator& communicator);

	// Destructor
	~MpiCommunicator();

	// TODO: Move static function calls to MpiEnvironment namespace?

	// Checks whether MPI is available (whether the code was compiled with MPI)
	static bool is_mpi_enabled();

	// Checks whether the MPI library has been initialized
	// - Returns "false" if MPI is not enabled
	// - Note: MPI mandates that MPI_Init may only be called *once* by each process
	static bool is_mpi_initialized();

	// Returns MPI_COMM_WORLD if MPI is enabled, else returns a dummy MPI_Comm
	// - Use this in order to avoid any direct references to MPI_COMM_WORLD in your code,
	//   so that in can compile if MPI is not enabled
	static MPI_Comm get_mpi_comm_world();

	// Returns MPI_COMM_SELF if MPI is enabled, else returns a dummy MPI_Comm
	// - Use this in order to avoid any direct references to MPI_COMM_WORLD in your code,
	//   so that in can compile if MPI is not enabled
	static MPI_Comm get_mpi_comm_self();

	// Assuming a dim-dimensional system in Cartesian space, this function
	// uses MPI_Dims_create() to determine the number of domain decomposition
	// cells that should be placed along each axis
	template<std::size_t dim>
	std::array<int,dim> calculateGridDimensions() const;


	//-----------------------------//
	//----- Get/Set Functions -----//
	//-----------------------------//

	// Duplicates the communicator provided
	void setCommunicator(const MPI_Comm& communicator);
	void setCommunicator(void* communicator_ptr);

	MPI_Comm& accessCommunicator() {
		return communicator_;
	}
	const MPI_Comm& getCommunicator() const {
		return communicator_;
	}

	// Returns the rank of this process in the commmunicator
	// - Returns 0 if MPI is not enabled
	int getRank() const;

	int getMasterRank() const { return master_rank_; }

	// Returns 'true' if this rank is the master rank
	bool isMasterRank() {
		return getRank() == getMasterRank();
	}

	// Returns the number of ranks in this communicator
	// - Returns 1 if MPI is not enabled, since a serial code is equivalent (in a sense)
	//   to an MPI run with one rank
	int getSize() const;
	int getNumRanks() const { return this->getSize(); }

	// Static mapping from primitive C++ types to MPI types
	/*
	template<typename T>
	static MPI_Datatype get_primitive_MPI_Datatype();
	*/

	bool isSerial() const {
		return ( this->getSize() == 1 );
	}

	//------------------------------//
	//----- Misc. MPI Routines -----//
	//------------------------------//

	// TODO abort

	//-------------------------------------//
	//----- MPI Datatype Registration -----//
	//-------------------------------------//

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


	//------------------------------------------------//
	//----- Point-to-Point Communication Methods -----//
	//------------------------------------------------//

	//----- Send -----//

	// Single value (by reference)
	template<typename T>
	void send(const T& data, const int destination, const int tag);

	// std::vector
	template<typename T, typename A>
	void send(const std::vector<T,A>& vec, const int destination, const int tag);

	// Wrapper around underlying MPI_Send
	void send(
		const MpiData& data,    // unified format for data buffer
		const int destination,  // rank of receiving process
		const int tag           // message identifier (should be unique)
	);


	//----- Recv -----//

	// Single element of a registered type
	template<typename T>
	void recv(T& data, const int source, const int tag, 
	          MpiStatus& status);

	// std::vector of a registered type
	// FIXME: known/unknown size
	template<typename T, typename A>
	void recv(std::vector<T,A>& vec, const int source, const int tag, 
	          MpiStatus& status);

	// Wrapper around underlying MPI_Recv
	// FIXME: known/unknown size
	void recv(
		MpiData& data,     // unified format for data buffer
		const int source,  // rank of source
		const int tag,     // message identifier (should be unique)
		MpiStatus& status
	);


	//----- Isend -----//

	// Single element of a registered type
	template<typename T>
	void Isend(const T& data, const int destination, const int tag, MpiRequest& request);

	// std::vector of a registered type
	// - Note: the *size* of the vector is used in the call to MPI_Isend, *not* its capacity
	template<typename T, typename A>
	void Isend(const std::vector<T,A>& data, const int destination, const int tag, MpiRequest& request);
	
	// Wrapper around underlying MPI_Isend
	void Isend(
		const MpiData& data, 
		const int destination,  // rank of receiving process
		const int tag,          // message identifier (should be unique)
		MpiRequest& request
	);


	//----- Irecv -----//

	// Single element of a registered type
	template<typename T>
	void Irecv(T& data, const int source, const int tag, MpiRequest& request);

	// std::vector of a registered type
	// - Note: the *size* of the vector is used in the call to MPI_Irecv, *not* its capacity
	//   - If you want to receive a variable amount of data, you should extent the 
	//     vector's size to its maximum capacity before calling this routine
	// FIXME: known/unknown size
	template<typename T,typename A>
	void Irecv(std::vector<T,A>& data, const int source, const int tag, MpiRequest& request);

	// Wrapper around underlying MPI_Irecv
	void Irecv(
		MpiData& data,
		const int source, // rank of source
		const int tag,
		MpiRequest& request
	);


	//--------------------------------------------//
	//----- Collective Communication Methods -----//
	//--------------------------------------------//

	// Pass as MpiData to perform MPI_IN_PLACE
	static const MpiData mpi_in_place;

	// Wrapper for MPI_Barrier()
	void barrier();


	//----- Bcast -----//

	// Wrapper for MPI_Bcast
	// - "data" is copied from rank "root" to all ranks (including root!)
	void bcast(
		MpiData&  data, 
		const int root
	);

	// reference to a block of contiguous memory of a primitive type 
	template<typename T>
	void bcast(T& data, const int root);


	//----- Allreduce -----//

	// Wrapper around underlying MPI_Allreduce
	// - This is called by the following allreduce() functions, which handle the
	//   conversion from raw data to MpiData
	// TODO remove?
	void allreduce(
		const MpiData& data_in,   // pass 'mpi_in_place' for MPI_IN_PLACE
		MpiData&       data_out,
		const MpiOp&   op
	);

	// Single element of a registered type
	template<typename T>
	void allreduce(const T& data_in, T& data_out, const MpiOp& op);

	// std::vector of a registered type
	template<typename T, typename A>
	void allreduce(const std::vector<T,A>& data_in, std::vector<T,A>& data_out, const MpiOp& op);

	// Helper templates that simplify the interface
	// - Same as above functions, but the operation is performed in place
	template<typename T>
	void allreduceInPlace(T& data, const MpiOp& op);
	template<typename T, typename A>
	void allreduceInPlace(std::vector<T,A>& data, const MpiOp& op);

	// Same as above, but for a standard operation
	template<typename T>
	void allreduce(const T& data_in, T& data_out, const MpiOp::StandardOp& op_enum) {
		allreduce(data_in, data_out, mapStandardMpiOp<T>(op_enum));
	}
	template<typename T, typename A>
	void allreduce(const std::vector<T,A>& data_in, std::vector<T,A>& data_out, 
	               const MpiOp::StandardOp& op_enum
	) {
		allreduce(data_in, data_out, mapStandardMpiOp<T>(op_enum));
	}
	template<typename T>
	void allreduceInPlace(T& data, const MpiOp::StandardOp& op_enum);
	template<typename T, typename A>
	void allreduceInPlace(std::vector<T,A>& data, const MpiOp::StandardOp& op_enum);

	// Extra templates, for the ultimate in laziness
	template<typename T>
	void allreduceSumInPlace(T& data);
	template<typename T, typename A>
	void allreduceSumInPlace(std::vector<T,A>& data);


	//----- Allgather -----//

	// Wrapper around underlying MPI_Allgather
	// - Amount of data to send/receive from each rank is inferred from the size
	//   of send_data/recv_data
	void allgather(
		const MpiData& send_data,
		MpiData&       recv_data
	);
		
	// Receive a single value of primitive type from each rank and put the results
	// into a std::vector
	template<typename T, typename A>
	void allgather(
		const T&          value,
		std::vector<T,A>& received_values
	);


	//----- Allgatherv -----//

	// Wrapper around underlying MPI_Allgatherv / MPI_Iallgatherv
	// - TODO: make protected/private?
	// - Assumes that appropriate blocks have already been prepared, that that 
	//   "recv_data" is the correct size to receive it all
	// - Variables
	//     recv_data: buffer for all data received
	//     recv_offsets: index in "recv_data.data_ptr" at which each block of data starts
	//     recv_counts: number of elements received
	void allgatherv(
		const MpiData&          send_data,
		const std::vector<int>& recv_counts,
		const std::vector<int>& recv_offsets,
		MpiData&                recv_data
	);
	void Iallgatherv(
		const MpiData&          send_data,
		const std::vector<int>& recv_counts,
		const std::vector<int>& recv_offsets,
		MpiData&                recv_data,
		MpiRequest&             request
	);

	// std::vectors of a registered type 
	// - Calls this->allgather() first to determine the number of data elements
	//   to receive from each rank
	template<typename T, typename A>
	void allgathervWithUnknownCounts(
		const std::vector<T,A>& send_data,
		std::vector<T,A>&       recv_data,
		std::vector<int>&       recv_counts,
		std::vector<int>&       recv_offsets
	);

	// std::vectors of a registered type 
	template<typename T, typename A>
	void allgatherv(
		const std::vector<T,A>& send_data,
		const std::vector<int>& recv_counts,
		std::vector<T,A>&       recv_data,
		std::vector<int>&       recv_offsets
	);

	template<typename T, typename A>
	void Iallgatherv(
		const std::vector<T,A>& send_data,
		const std::vector<int>& recv_counts,
		std::vector<T,A>&       recv_data,
		std::vector<int>&       recv_offsets,
		MpiRequest&             request
	);


	//----------------------//
	//----- Exceptions -----//
	//----------------------//

	class MpiOpNotFoundException : public std::exception {
	 public:
		MpiOpNotFoundException() {}
		const char* what() const noexcept override {
			return "Given MpiOp not found";
		};
	};

	//----- Map Wrappers -----//

	// Maps type T to the appropriate MpiDatatype using member variable 'mpi_datatype_map_'
	// - If the mapping does not already exist, a new mapping is constructed if possible
	// TODO Make private?
	template<typename T>
	const MpiDatatype& mapMpiDatatype();

 private:
	// Underlying MPI communicator
	bool is_communicator_initialized_ = false;
	MPI_Comm communicator_;

	// Rank (index) of the master process
	const int master_rank_ = 0;

	// Map from C++ std::type_index(typeid(T)) to MpiDatatypes
	std::unordered_map<std::type_index, std::unique_ptr<MpiDatatype>> mpi_datatype_map_;

	// Register a number of commonly-used types in 'mpi_datatype_map_'
	void registerDefaultMpiDatatypes();

	// Registers all predefined MPI_Datatypes in 'mpi_datatype_map_'
	// - These must be registered first, before any other types!
	void registerPrimitiveMpiDatatypes();

	// Map from type T to its registered standard MpiOps
	std::unordered_map<std::type_index, MpiOp::StandardOpsMap> standard_mpi_op_map_;

	// Returns the MpiOp for the given type corresponding to the standard operation 
	// represented by the StandardOp enum
	template<typename T>
	const MpiOp& mapStandardMpiOp(const MpiOp::StandardOp& op_enum);
}; // end class MpiCommunicator


//--------------------------------//
//--------------------------------//
//----- TEMPLATE DEFINITIONS -----//
//--------------------------------//
//--------------------------------//

// Templates... templates everywhere...

// Now include the definitions necessary to implement the following templates
#include "MpiData.h"
#include "MpiDatatype.h"
#include "MpiDatatypeRegistrar.h"


// Wrapper for MPI_Dims_create
template<std::size_t dim>
std::array<int,dim> MpiCommunicator::calculateGridDimensions() const
{
	if ( dim == 0 ) {
		throw std::runtime_error("MPI: Can't parition ranks along 0 dimensions.\n");
	}

	std::array<int,dim> grid_dimensions;

	// Use an int to avoid compiler warnings about comparing signed and
	// unsigned integers
	int dim_int = static_cast<int>(dim);

#ifdef MPI_ENABLED
	// Use MPI_Dimms_create to auto-select grid dimensions
	int num_ranks = this->getSize();
	for ( int d=0; d<dim_int; ++d ) {
		// MPI_Dimms create will only change grid_dimensions[d] if it's 0; any
		// positive values will be unchanged!
		grid_dimensions[d] = 0;
	}
	MPI_Dims_create( num_ranks, dim_int, grid_dimensions.data() );

#else
	for ( int d=0; d<dim_int; ++d ) {
		grid_dimensions[d] = 1;
	}
#endif /* MPI_ENABLED */

	return grid_dimensions;
}


//-------------------------------------//
//----- MPI_Datatype Registration -----//
//-------------------------------------//


// Register type T as MpiDatatype T
template<typename T>
void MpiCommunicator::registerType(const MpiDatatype& data_type)
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
	mpi_datatype_map_.insert( std::make_pair(ti, std::unique_ptr<MpiDatatype>(new MpiDatatype(data_type))) );
}


// Registers type T as an MpiDatatype by using its MpiDatatypeRegistrar
template<typename T>
const MpiDatatype& MpiCommunicator::registerType()
{
	const MpiDatatype* new_datatype_ptr;

	// First, ensure the type doesn't already exist
	std::type_index ti(typeid(T));  // convert to type index
	auto it = mpi_datatype_map_.find(ti);   //std::type_index(typeid(T)) );
	if ( it == mpi_datatype_map_.end() ) {
		// Register a new type
		using Registrar = MpiDatatypeRegistrar<T>;
		auto insert_result_pair = mpi_datatype_map_.insert( std::make_pair(ti, 
				std::unique_ptr<MpiDatatype>(new MpiDatatype(Registrar::makeMpiDatatype(*this)))) );
		const auto& new_pair_it = insert_result_pair.first;
		const MpiDatatype& new_datatype = *(new_pair_it->second);
		new_datatype_ptr = &new_datatype;

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
	}
	else {
		std::stringstream err_ss;
		err_ss << "Error in " << FANCY_FUNCTION << "\n"
					 << "  type \"" << ti.name() << "\" is already registered \n"
					 << "  (NOTE: the type name printed above is implementation-dependent)\n";
		throw std::runtime_error( err_ss.str() );
	}

	return *new_datatype_ptr;
}


// Maps type T to the appropriate MpiDatatype using member variable 'mpi_datatype_map_'
// - If the mapping does not already exist, a new mapping is constructed if possible
template<typename T>
const MpiDatatype& MpiCommunicator::mapMpiDatatype()
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


template<typename T> inline
const MpiOp& MpiCommunicator::mapStandardMpiOp(const MpiOp::StandardOp& op_enum)
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



//------------------------------------------------//
//----- Point-to-Point Communication Methods -----//
//------------------------------------------------//


//----- Send -----//

// Single element of registered type (by reference)
template<typename T>
void MpiCommunicator::send(const T& data, const int destination, const int tag)
{
	// MpiData constructor will not actually change the data
	const MpiData send_data( const_cast<T&>(data), *this );
	this->send(send_data, destination, tag);
}


// std::vector of a registered type
template<typename T, typename A>
void MpiCommunicator::send(
		const std::vector<T,A>& vec, const int destination, const int tag)
{
	// MpiData constructor will not actually change the data
	const MpiData send_data( const_cast<std::vector<T,A>&>(vec), *this );
	this->send(send_data, destination, tag);
}


//----- Recv -----//


// Single element of registered type (by reference)
template<typename T>
void MpiCommunicator::recv(
		T& data, const int source, const int tag, 
		MpiStatus& status)
{
	MpiData data_to_recv(data, *this);
	this->recv(data_to_recv, source, tag, status);
}


// std::vector of a registered type
template<typename T, typename A>
void MpiCommunicator::recv(
		std::vector<T,A>& vec, const int source, const int tag, 
		MpiStatus& status)
{
	// Check whether a variable amount of data is being received
	// - If the MPI_Status is not being ignored, set the vector's final size based on 
	//   how much data was actually received
	// - Else TODO
	// FIXME: MPI_Probe?
	bool dynamic_size = false;
	if ( status.ignore() ) {
		dynamic_size = true; 
		// For safety, extend the buffer to its maximum capacity
		vec.resize( vec.capacity() );
	}

	MpiData data_to_recv(vec, *this);
	this->recv(data_to_recv, source, tag, status);

	if ( dynamic_size ) {
		// Set final size
		data_to_recv.size = status.getCount<T>();
		vec.resize( data_to_recv.size );
	}
}


//----- Isend -----//

// Single element of a registered type
template<typename T>
void MpiCommunicator::Isend(
	const T& data, const int destination, const int tag, MpiRequest& request)
{
	MpiData mpi_data( const_cast<T&>(data), *this );
	this->Isend(mpi_data, destination, tag, request);
}


// std::vector of a registered type
template<typename T, typename A>
void MpiCommunicator::Isend(
	const std::vector<T,A>& data, const int destination, const int tag, MpiRequest& request)
{
	MpiData mpi_data( const_cast<std::vector<T,A>&>(data), *this );
	this->Isend(mpi_data, destination, tag, request);
}


//----- Irecv -----//

// Single element of a registered type
template<typename T>
void MpiCommunicator::Irecv(T& data, const int source, const int tag, MpiRequest& request)
{
	MpiData mpi_data( data, *this );
	this->Irecv(mpi_data, source, tag, request);
}


// std::vector of a registered type
// - Note: the *size* of the vector is used in the call to MPI_Irecv, *not* its capacity
//   - If you want to receive a variable amount of data, you should extent the 
//     vector's size to its maximum capacity before calling this routine
template<typename T, typename A>
void MpiCommunicator::Irecv(
	std::vector<T,A>& data, const int source, const int tag, MpiRequest& request)
{
	MpiData mpi_data( data, *this );
	this->Irecv(mpi_data, source, tag, request);
}



//------------------------------------//
//----- Collective Communication -----//
//------------------------------------//

//----- Bcast -----//

// reference to a block of contiguous memory of a primitive type 
template<typename T> inline
void MpiCommunicator::bcast(T& data, const int root)
{
	MpiData mpi_data( data, *this );
	this->bcast(mpi_data, root);
}


//----- Allreduce -----//


// Single object of a registered type
template<typename T> inline
void MpiCommunicator::allreduce(
		const T& data_in, T& data_out, const MpiOp& op)
{
	MpiData mpi_data_in( const_cast<T&>(data_in), *this );
	MpiData mpi_data_out( data_out, *this );
	this->allreduce(mpi_data_in, mpi_data_out, op);
}


// std::vector of a registered type
template<typename T, typename A> inline
void MpiCommunicator::allreduce(
		const std::vector<T,A>& data_in, std::vector<T,A>& data_out, const MpiOp& op)
{
	// Ensure that sizes match
	data_out.resize( data_in.size() );

	MpiData mpi_data_in( const_cast<T&>(data_in), *this );
	MpiData mpi_data_out( data_out, *this );
	this->allreduce(mpi_data_in, mpi_data_out, op);
}

// Same as above, but in place
template<typename T> inline
void MpiCommunicator::allreduceInPlace(T& data, const MpiOp& op) {
	MpiData mpi_data(data, *this);
	allreduce(mpi_in_place, mpi_data, op);
}
template<typename T, typename A> inline
void MpiCommunicator::allreduceInPlace(std::vector<T,A>& data, const MpiOp& op) {
	MpiData mpi_data(data, *this);
	allreduce(mpi_in_place, mpi_data, op);
}

// In place, and for a standard operation
template<typename T> inline
void MpiCommunicator::allreduceInPlace(T& data, const MpiOp::StandardOp& op_enum) {
	MpiData mpi_data(data, *this);
	allreduce(mpi_in_place, mpi_data, mapStandardMpiOp<T>(op_enum));
}
template<typename T, typename A> inline
void MpiCommunicator::allreduceInPlace(std::vector<T,A>& data, const MpiOp::StandardOp& op_enum) {
	MpiData mpi_data(data, *this);
	allreduce(mpi_in_place, mpi_data, mapStandardMpiOp<T>(op_enum));
}
template<typename T> inline
void MpiCommunicator::allreduceSumInPlace(T& data) {
	MpiData mpi_data(data, *this);
	allreduce(mpi_in_place, mpi_data, mapStandardMpiOp<T>(MpiOp::StandardOp::Sum));
}
template<typename T, typename A> inline
void MpiCommunicator::allreduceSumInPlace(std::vector<T,A>& data) {
	MpiData mpi_data(data, *this);
	allreduce(mpi_in_place, mpi_data, mapStandardMpiOp<T>(MpiOp::StandardOp::Sum));
}


//----- Allgather -----//


// Receive a single value of primitive type from each rank and put the results
// into a std::vector
template<typename T, typename A> inline
void MpiCommunicator::allgather(const T& value, std::vector<T,A>& received_values)
{
	// Value to send will not actually be modified
	MpiData send_data( const_cast<T&>(value), *this );

	// Allocate space in the receive buffer
	int num_ranks = this->getSize();
	received_values.resize(num_ranks);
	MpiData recv_data(received_values, *this);

	this->allgather(send_data, recv_data);
}


//----- Allgatherv -----//

template<typename T, typename A>
void MpiCommunicator::allgatherv(
	const std::vector<T,A>& send_data, const std::vector<int>& recv_counts,
	std::vector<T,A>& recv_data, std::vector<int>& recv_offsets)
{
	// Set up local receive buffer and offsets
	int num_ranks = this->getSize();
	recv_offsets.resize(num_ranks);
	int num_recv_total = 0;
	for ( int r=0; r<num_ranks; ++r ) {
		if ( r == 0 )
			recv_offsets[r] = 0;
		else {
			recv_offsets[r] = recv_offsets[r-1] + recv_counts[r-1];
		}
		num_recv_total += recv_counts[r];
	}
	recv_data.resize(num_recv_total);

	// Gather the data
	MpiData mpi_data_send( const_cast<std::vector<T,A>&>(send_data), *this );
	MpiData mpi_data_recv( recv_data, *this );
	this->allgatherv(mpi_data_send, recv_counts, recv_offsets,
	                 mpi_data_recv);
}



template<typename T, typename A>
void MpiCommunicator::allgathervWithUnknownCounts(
	const std::vector<T,A>& send_data,
	std::vector<T,A>& recv_data, std::vector<int>& recv_counts, std::vector<int>& recv_offsets)
{
	// First, gather the number of elements that each rank will send.
	int num_ranks = this->getSize();
	recv_counts.assign(num_ranks, 0);
	int local_block_size = send_data.size();
	this->allgather(local_block_size, recv_counts);

	// Use counts
	this->allgatherv(send_data, recv_counts, recv_data, recv_offsets);
}


template<typename T, typename A>
void MpiCommunicator::Iallgatherv(
	const std::vector<T,A>& send_data, const std::vector<int>& recv_counts,
	std::vector<T,A>& recv_data, std::vector<int>& recv_offsets, MpiRequest& request)
{
	// Set up local receive buffer and offsets
	int num_ranks = this->getSize();
	recv_offsets.resize(num_ranks);
	int num_recv_total = 0;
	for ( int r=0; r<num_ranks; ++r ) {
		if ( r == 0 )
			recv_offsets[r] = 0;
		else {
			recv_offsets[r] = recv_offsets[r-1] + recv_counts[r-1];
		}
		num_recv_total += recv_counts[r];
	}
	recv_data.resize(num_recv_total);

	// Gather the data
	MpiData mpi_data_send( const_cast<std::vector<T,A>&>(send_data), *this );
	MpiData mpi_data_recv( recv_data, *this );
	this->Iallgatherv(mpi_data_send, recv_counts, recv_offsets,
	                  mpi_data_recv, request);
}

#endif /* MPI_COMMUNICATOR_H */
