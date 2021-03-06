// MpiCommunicator
// - Written by Sean M. Marks (https://github.com/seanmarks)
//
// ABOUT: Wrapper around MPI communicator and other library functions
//	- Heavily influenced by the Communicator implemented in PLUMED 2.4.0 (see http://plumed.org)
//  - Contains lots of template code for commonly-used STL containers that store
//    primitive types (e.g. int, double) in contiguous memory.
//  - Shields the user from many of the tedious aspects of MPI library calls
//
// FOR THE USER:
//  - Make sure std::vector buffers are prepared using resize(), not just reserve()
//
// NOTES:
//	- Most functions throw if MPI is not initialized
//    - Notable exceptions: functions that return size/rank
//
//  - Unless stated otherwise, MPI wrappers that involve some rank receiving data
//    (e.g. recv, Irecv, bcast,scatter, allgather, ...) *assume* that the receive
//    buffers are appropriately sized before calling the function.
//    - ex. to receive 100 elements in a std::vector, the vector must be passed
//          a size (not capacity) of 100
//
//    - There are some exceptions
//       Ex. allgathervWithUnknownCounts()
//          - calls allgather() to determine how much data each rank will send,
//            and resizes all output arrays accordingly
//
//  - The return codes of MPI calls are not checked because the default behavior of
//    MPI is to internally check for errors before returning, and abort on error
//    - There is no guarantee that MPI can continue past an error anyway
//
// DEVELOPMENT (TODO)
//  - Pointer overloads
//
//  - Generalize template code for std::vectors of contiguous data arrays?
//    - Such a setup would cover std::vector<T>, PLMD::Vector, std::array<T,dim>, ...
//    - Could make use of MPI's own custom type system with MPI_Type_contiguous
//      - Reference:  https://tech.io/playgrounds/349/introduction-to-mpi/custom-types
//
//  - Non-blocking receives with final length unknown a priori
//
//  - Make MpiCommunicator data-sharing methods all const?
//    - They don't really change the communicator state (right?)
// - Add more notes about proper usage, esp. resizing buffers
//
// FIXME:
// - Copying an MpiCommunicator
//   - Copying the registry/registries

#pragma once
#ifndef MPI_COMMUNICATOR_H
#define MPI_COMMUNICATOR_H

#include <array>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "Assert.h"
#include "MpiData.h"
#include "MpiDatatype.h"
#include "MpiDatatypeRegistry.h"
#include "MpiEnvironment.h"
#include "MpiOp.h"
#include "MpiRequest.h"
#include "MpiStatus.h"


// TODO: prevent multiple definitions when compiled with other
// libraries that define a dummy MPI_Comm
#ifndef MPI_ENABLED
class MPI_Comm {};
#endif // MPI_ENABLED


class MpiCommunicator
{
 public:
	using StandardOp = MpiOp::StandardOp;

	//-------------------------------------------//
	//----- Constructors and Initialization -----//
	//-------------------------------------------//

	// Default constructor: initialize as a duplicate of MPI_COMM_SELF
	MpiCommunicator();

	// Initialize as a duplicate of the raw input communicator
	explicit MpiCommunicator(const MPI_Comm& communicator);

	// Copy constructor
	// FIXME:
	// - duplicating member variables appropriately
	MpiCommunicator(const MpiCommunicator& communicator);

	// Initialize as a duplicate of MPI_COMM_WORLD
	static MpiCommunicator World();

	// Initialize as a duplicate of MPI_COMM_SELF
	static MpiCommunicator Self();

	// Destructor
	~MpiCommunicator();

	// Assuming a dim-dimensional system in Cartesian space, this function
	// uses MPI_Dims_create() to determine the number of domain decomposition
	// cells that should be placed along each axis
	template<std::size_t dim>
	std::array<int,dim> calculateGridDimensions() const;


	//---------------------------------//
	//----- Communicator Settings -----//
	//---------------------------------//

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
	int rank() const { return getRank(); }

	// Returns the master rank for this communicator
	int getMasterRank() const {
		return master_rank_;
	}
	int masterRank() const {
		return getMasterRank();
	}

	// Returns 'true' if this rank is the master rank
	bool isMasterRank() {
		return getRank() == getMasterRank();
	}

	// Returns the number of ranks in this communicator
	// - Returns 1 if MPI is not enabled, since a serial code is equivalent (in a sense)
	//   to an MPI run with one rank
	int getSize() const;
	int size() const        { return this->getSize(); }
	int getNumRanks() const { return this->getSize(); }

	bool isSerial() const {
		return ( this->getSize() == 1 );
	}

	// Shortcuts to environment status functions
	static bool is_mpi_initialized() {
		return MpiEnvironment::is_initialized();
	}
	static bool is_mpi_enabled() {
		return MpiEnvironment::is_enabled();
	}


	//------------------------------//
	//----- Misc. MPI Routines -----//
	//------------------------------//

	// TODO abort

	// Test for an incoming message
	// - Blocking
	void probe(const int source, const int tag, MpiStatus& status) const;
	// - Non-blocking
	bool Iprobe(const int source, const int tag, MpiStatus& status) const;


	//-------------------------------------//
	//----- MPI Datatype Registration -----//
	//-------------------------------------//

	// TODO: Public interface for the registration of new types



	//------------------------------------------------//
	//----- Point-to-Point Communication Methods -----//
	//------------------------------------------------//


	//----- send -----//

	// Blocking send
	template<typename T>
	void send(
		const T&  data,
		const int destination,  // rank of receiving process
		const int tag           // message identifier (should be unique)
	);


	//----- recv -----//

	// TODO: variable message length

	// Blocking receive
	template<typename T>
	void recv(
		T&         data,
		const int  source,
		const int  tag,
		MpiStatus& status
	);


	//----- Isend -----//

	// Non-blocking send
	//
	// - Assumes:
	//   - 'data.size()' is the number of elements to send
	template<typename T>
	void Isend(
		const T&    data,
		const int   destination,
		const int   tag,
		MpiRequest& request
	);


	//----- Irecv -----//

	// TODO: variable message length

	// Non-blocking receive
	//
	// - Assumes:
	//   - 'data.size()' is correct for the incoming message
	template<typename T>
	void Irecv(
		T&          data,
		const int   source,
		const int   tag,
		MpiRequest& request
	);


	//--------------------------------------------//
	//----- Collective Communication Methods -----//
	//--------------------------------------------//

	// Wrapper for MPI_Barrier()
	void barrier();


	//----- Bcast -----//

	// Broadcast 'data' from rank 'root' to all ranks
	// - Note that 'root' retains a copy of the data
	// - Assumes:
	//   - 'data.size()' is correct on all ranks
	template<typename T>
	void bcast(
		T&        data,
		const int root
	);

	// TODO bcastWithUnknownCount


	//----- Scatter -----//

	// Scatters chunks of data from 'root' to each rank
	// - Assumes:
	//   - 'send_count' is known on each rank (used to resize 'recv_data')
	//   - if N = send_data.size(), then recv_data.size() = N/num_ranks
	template<typename T, typename Container>
	void scatter(
		const T&  send_data,   // data to scatter (size = send_count*num_ranks)
		const int send_count,  // number of data elements to send to each rank
		// Output
		Container& recv_data,  // resized to 'size_count'
		const int  root
	);


	//----- Scatterv -----//

	// Scatters a variable amount of data to each rank
	// - 'send_offsets[j]' is the offset in 'send_data' where the data to rank 'j' begins
	// - Assumes:
	//   - 'recv_data' is properly sized on each rank
	//      - On rank 'r':  recv_data.size() == send_counts[r]
	template<typename T, typename Container>
	void scatterv(
		// Input (only relevant at 'root')
		const T&                send_data,     // data to scatter
		const std::vector<int>& send_counts,   // number of data elements to send to each rank
		const std::vector<int>& send_offsets,  // linear offsets
		// Output
		Container&              recv_data,  // size unchanged
		const int               root
	);

	// Performs scatterv, but does not assume that the receiving processes know their 'recv_count'
	template<typename T, typename Container>
	void scattervWithUnknownCounts(
		// Input (only relevant at 'root')
		const T&                send_data,     // data to scatter
		const std::vector<int>& send_counts,   // number of data elements to send to each rank
		const std::vector<int>& send_offsets,  // linear offsets
		// Output
		Container&              recv_data,  // resized as needed
		const int               root
	);


	//----- Allgather -----//

	// Gathers a set of data on all ranks
	//
	// - Assumes:
	//   - if N = data.size(), then recv_data.size() = N*num_ranks
	template<typename T, typename Container>
	void allgather(
		const T&   data,
		Container& recv_data  // container is resized as needed
	);


	//----- Allgatherv -----//

	// Gathers a variable amount of data from each rank, and distributes it to all ranks
	// - 'recv_offsets[j]' is the offset in 'recv_data' where the data from rank 'j' begins
	//
	// - Assumes:
	//   - 'recv_counts' are known on all ranks
	template<typename T, typename Container>
	void allgatherv(
		const T&                send_data,    // data to send from this rank
		const std::vector<int>& recv_counts,  // number of data elements to receive from each rank
		// Output
		Container&        recv_data,    // resized as needed
		std::vector<int>& recv_offsets  // assigned
	);

	// Performs an allgather, but does *not* assume that 'recv_counts' are known
	// - Calls allgather() first to determine the number of data elements
	//   to receive from each rank
	template<typename T, typename Container>
	void allgathervWithUnknownCounts(
		const T&          send_data,
		// Output
		Container&        recv_data,    // resized as needed
		std::vector<int>& recv_counts,  // assigned
		std::vector<int>& recv_offsets  // assigned
	);


	//----- Allreduce -----//

	template<typename T>
	void allreduce(const T& data_in, T& data_out, const MpiOp& op);

	// In place
	template<typename T>
	void allreduceInPlace(T& data, const MpiOp& op);

	// For a standard operation
	template<typename T>
	void allreduce(const T& data_in, T& data_out, const MpiOp::StandardOp& op_enum);
	template<typename T>
	void allreduceInPlace(T& data, const MpiOp::StandardOp& op_enum);

	// Shortcuts for commonly-used reductions
	template<typename T>
	void allreduceSumInPlace(T& data);


	//----------------------//
	//----- Exceptions -----//
	//----------------------//

	// TODO


	//-----------------//
	//----- Misc. -----//
	//-----------------//

	// Returns a string for prepending to print statements, which indicates
	// the current rank
	std::string getLinePrefix() const {
		return "( rank " + std::to_string(getRank()) + ") ";
	}


 private:
	// Underlying MPI communicator
	bool is_communicator_initialized_ = false;
	MPI_Comm communicator_;

	// Rank (index) of the master process
	const int master_rank_ = 0;

	// Registry with the types recognized by this communicator
	MpiDatatypeRegistry datatype_registry_;

	// Given a set of counts, constructs a set of minimal linear offsets
	// that map a set of arrays of varying length to a 1D array
	static void makeOffsetsUsingCounts(
		const std::vector<int>& counts,
		std::vector<int>&       offsets,
		int&                    total_count  // sum of all 'counts'
	);
}; // end class MpiCommunicator


//-----------------//
//----- Misc. -----//
//-----------------//


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
#endif // ifdef MPI_ENABLED

	return grid_dimensions;
}


inline 
void MpiCommunicator::makeOffsetsUsingCounts(
	const std::vector<int>& counts,
	std::vector<int>& offsets, int& total_count
) {
	int num_counts = counts.size();
	offsets.resize(num_counts);
	total_count = 0;
	for ( int r=0; r<num_counts; ++r ) {
		if ( r == 0 )
			offsets[r] = 0;
		else {
			offsets[r] = offsets[r-1] + counts[r-1];
		}
		total_count += counts[r];
	}
}


//------------------------------------------------//
//----- Point-to-Point Communication Methods -----//
//------------------------------------------------//


template<typename T>
inline
void MpiCommunicator::send(
	const T& data, const int destination, const int tag)
{
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		// send expects non-const inputs
		MpiData<T> send_data( const_cast<T&>(data), datatype_registry_ );

		MPI_Send( send_data.data(), send_data.size(), send_data.getDatatype(),
		          destination, tag, communicator_ );
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // ifdef MPI_ENABLED
}


template<typename T>
inline
void MpiCommunicator::recv(T& data, const int source, const int tag, MpiStatus& status)
{
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		MpiData<T> recv_data( const_cast<T&>(data), datatype_registry_ );

		if ( status.ignore() ) {
			MPI_Recv( recv_data.data(), recv_data.size(), recv_data.getDatatype(),
			          source, tag, communicator_, MPI_STATUS_IGNORE );
		}
		else {
			MPI_Recv( recv_data.data(), recv_data.size(), recv_data.getDatatype(),
			          source, tag, communicator_, &status.access_MPI_Status() );
			status.setDatatype( recv_data.getDatatype() );
		}
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // ifdef MPI_ENABLED
}


template<typename T>
inline
void MpiCommunicator::Isend(
	const T& data, const int destination, const int tag, MpiRequest& request)
{
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		MpiData<T> send_data( const_cast<T&>(data), datatype_registry_ );

		MPI_Isend( send_data.data(), send_data.size(), send_data.getDatatype(),
		           destination, tag, communicator_, &request.access_MPI_Request() );
		request.setActive( send_data.getDatatype() );
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // ifdef MPI_ENABLED
}


template<typename T>
inline
void MpiCommunicator::Irecv(T& data, const int source, const int tag, MpiRequest& request)
{
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		MpiData<T> recv_data( data, datatype_registry_ );

		MPI_Irecv( recv_data.data(), recv_data.size(), recv_data.getDatatype(),
		           source, tag, communicator_, &request.access_MPI_Request() );
		request.setActive( recv_data.getDatatype() );
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // ifdef MPI_ENABLED
}



//------------------------------------//
//----- Collective Communication -----//
//------------------------------------//


//----- Bcast -----//

template<typename T>
inline
void MpiCommunicator::bcast(T& data, const int root)
{
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		MpiData<T> mpi_data( data, datatype_registry_ );

		MPI_Bcast( mpi_data.data(), mpi_data.size(), mpi_data.getDatatype(),
		           root, this->communicator_ );
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // ifdef MPI_ENABLED
}


// TODO bcastWithUnknownCount


//----- Scatter -----//


template<typename T, typename Container>
void MpiCommunicator::scatter(
	const T& send_data, const int send_count,
	Container& recv_data, const int root )
{
	static_assert(is_MpiData_same_value_type<T,Container>::value, "type mismatch");
	
	MpiData<T>         mpi_send_data( const_cast<T&>(send_data), datatype_registry_ );
	MpiData<Container> mpi_recv_data( recv_data,                 datatype_registry_ );

	FANCY_DEBUG_ASSERT( (! isMasterRank()) || (mpi_send_data.size() == send_count*size()),
	                    getLinePrefix() << "size mismatch: " << mpi_send_data.size() << " != " << send_count << "*" << size() );

	const int recv_count = send_count;
	mpi_recv_data.resize(recv_count);
	
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		// MPI_Allgatherv expects non-constant send buffer
		MPI_Scatter( mpi_send_data.data(), send_count, mpi_send_data.getDatatype(),
		             mpi_recv_data.data(), recv_count, mpi_recv_data.getDatatype(),
		             root, this->communicator_ );
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // ifdef MPI_ENABLED
}


//----- Scatterv -----//


template<typename T, typename Container>
void MpiCommunicator::scatterv(
	const T& send_data, const std::vector<int>& send_counts, const std::vector<int>& send_offsets,
	Container& recv_data, const int root )
{
	static_assert(is_MpiData_same_value_type<T,Container>::value, "type mismatch");
	FANCY_DEBUG_ASSERT( (! isMasterRank()) || (send_counts.size() == static_cast<unsigned>(size())), "inconsistent size" );
	FANCY_DEBUG_ASSERT( (! isMasterRank()) || (send_counts.size() == send_offsets.size()), "size mismatch" );

	MpiData<T>         mpi_send_data( const_cast<T&>(send_data), datatype_registry_ );
	MpiData<Container> mpi_recv_data( recv_data,                 datatype_registry_ );
	
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		// MPI_Allgatherv expects non-constant send buffer
		MPI_Scatterv( mpi_send_data.data(), send_counts.data(), send_offsets.data(), mpi_send_data.getDatatype(),
		              mpi_recv_data.data(), recv_data.size(),                        mpi_recv_data.getDatatype(),
		              root, this->communicator_ );
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // ifdef MPI_ENABLED
}


template<typename T, typename Container>
void MpiCommunicator::scattervWithUnknownCounts(
	const T& send_data, const std::vector<int>& send_counts, const std::vector<int>& send_offsets,
	Container& recv_data, const int root )
{
	static_assert(is_MpiData_same_value_type<T,Container>::value, "type mismatch");
	FANCY_DEBUG_ASSERT( (! isMasterRank()) || (send_counts.size() == static_cast<unsigned>(size())), "inconsistent size" );

	// Share the appropriate count with each rank and prepare the local receive buffer
	int recv_count = 0;
	scatter( send_counts, 1, recv_count, root );
	// - Need to wrap in a temporary MpiData to get access to a universal resize() function
	MpiData<Container> mpi_recv_data( recv_data, datatype_registry_ );
	mpi_recv_data.resize( recv_count );

	scatterv(send_data, send_counts, send_offsets, recv_data, root);
}


//----- Allgather -----//


template<typename T, typename Container>
void MpiCommunicator::allgather(const T& data_in, Container& data_out)
{
	static_assert(is_MpiData_same_value_type<T,Container>::value, "type mismatch");

	MpiData<T> send_data( const_cast<T&>(data_in), datatype_registry_ );
	int send_size = send_data.size();

	MpiData<Container> recv_data( data_out, datatype_registry_ );
	int recv_size = this->getNumRanks() * send_size;
	recv_data.resize( recv_size );

#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		FANCY_ASSERT( send_data.getDatatype() == recv_data.getDatatype(), "type mismatch" );

		MPI_Allgather( send_data.data(), send_data.size(), send_data.getDatatype(),
		               recv_data.data(), send_data.size(), recv_data.getDatatype(),
		               this->communicator_ );
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // ifdef MPI_ENABLED
}


//----- Allgatherv -----//


template<typename T, typename Container>
void MpiCommunicator::allgatherv(
	const T& send_data, const std::vector<int>& recv_counts,
  Container& recv_data, std::vector<int>& recv_offsets )
{
	static_assert(is_MpiData_same_value_type<T,Container>::value, "type mismatch");

	// Set up local offsets
	int num_recv_total = 0;
	makeOffsetsUsingCounts(recv_counts, recv_offsets, num_recv_total);

	MpiData<T>         mpi_send_data( const_cast<T&>(send_data), datatype_registry_ );
	MpiData<Container> mpi_recv_data( recv_data,                 datatype_registry_ );
	recv_data.resize(num_recv_total);
	
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		// MPI_Allgatherv expects non-constant send buffer
		MPI_Allgatherv( mpi_send_data.data(), mpi_send_data.size(),                    mpi_send_data.getDatatype(),
		                mpi_recv_data.data(), recv_counts.data(), recv_offsets.data(), mpi_recv_data.getDatatype(),
		                this->communicator_ );
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // ifdef MPI_ENABLED
}


template<typename T, typename Container>
void MpiCommunicator::allgathervWithUnknownCounts(
	const T& send_data,
	Container& recv_data, std::vector<int>& recv_counts, std::vector<int>& recv_offsets )
{
	// First, gather the number of elements that each rank will send.
	int num_ranks = this->size();
	recv_counts.assign(num_ranks, 0);
	int local_block_size = send_data.size();
	allgather(local_block_size, recv_counts);

	// Share all data
	allgatherv(send_data, recv_counts, recv_data, recv_offsets);
}


//----- Allreduce -----//


template<typename T>
inline
void MpiCommunicator::allreduce(const T& data_in, T& data_out, const MpiOp& op)
{
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		MpiData<T> mpi_data_in ( const_cast<T&>(data_in), datatype_registry_ );
		MpiData<T> mpi_data_out( data_out,                datatype_registry_ );

		FANCY_ASSERT( mpi_data_in.getDatatype() == mpi_data_out.getDatatype(), "type mismatch" );
		FANCY_ASSERT( mpi_data_in.size()        == mpi_data_out.size(),        "size mismatch" );

		MPI_Allreduce( mpi_data_in.data(), mpi_data_out.data(), mpi_data_out.size(), mpi_data_out.getDatatype(),
									 op.get_MPI_Op(), this->communicator_ );
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // ifdef MPI_ENABLED
}


template<typename T>
inline
void MpiCommunicator::allreduceInPlace(T& data, const MpiOp& op)
{
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		MpiData<T> mpi_data( data, datatype_registry_ );

		MPI_Allreduce( MPI_IN_PLACE, mpi_data.data(), mpi_data.size(), mpi_data.getDatatype(),
		               op.get_MPI_Op(), this->communicator_ );
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // ifdef MPI_ENABLED
}


template<typename T>
inline
void MpiCommunicator::allreduce(const T& data_in, T& data_out, const MpiOp::StandardOp& op_enum)
{
	using value_type = typename MpiDataTraits<T>::value_type;
	allreduce(data_in, data_out, datatype_registry_.mapStandardMpiOp<value_type>(op_enum));
}


template<typename T>
inline
void MpiCommunicator::allreduceInPlace(T& data, const MpiOp::StandardOp& op_enum) {
	using value_type = typename MpiDataTraits<T>::value_type;
	allreduceInPlace( data, datatype_registry_.mapStandardMpiOp<value_type>(op_enum) );
}


template<typename T>
inline
void MpiCommunicator::allreduceSumInPlace(T& data)
{
	allreduceInPlace( data, MpiOp::StandardOp::Sum );
}


#endif // ifndef MPI_COMMUNICATOR_H
