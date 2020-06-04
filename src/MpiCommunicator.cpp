// MpiCommunicator.cpp
// - Mainly implementations of non-templated wrappers around MPI functions

#include "MpiCommunicator.h"
//#include "MpiDatatype.h"

// Useful typedefs
//using MpiStatus = MpiCommunicator::MpiStatus;

#ifndef MPI_ENABLED
/*
// Dummy MPI_Init and MPI_Finalize for when MPI isn't available
void MPI_Init(int* argc, char** argv[]) {
	// Suppress "unused arguments" warnings
	(void) argc; (void) argv;
};

// Dummy MPI_Finalize
void MPI_Finalize() {};
*/
#endif /* MPI_ENABLED */


//-------------------------------------------//
//----- Constructors and Initialization -----//
//-------------------------------------------//


MpiCommunicator::MpiCommunicator()
{
#ifdef MPI_ENABLED
	//do_MPI_Init_if_not_initialized();
	setCommunicator(MPI_COMM_SELF);
	registerDefaultMpiDatatypes();
#endif /* MPI_ENABLED */
}


MpiCommunicator::MpiCommunicator(const MPI_Comm& communicator)
{
#ifdef MPI_ENABLED
	//do_MPI_Init_if_not_initialized();
	setCommunicator(communicator);
	registerDefaultMpiDatatypes();
#else
	// Suppress "unused arguments" warnings
	(void) communicator;
#endif /* MPI_ENABLED */
}


void MpiCommunicator::registerDefaultMpiDatatypes()
{
	// First, register MPI primitives. These will be used to build derived MPI_Datatypes.
	registerPrimitiveMpiDatatypes();

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

	registerType< std::array<std::array<double,1>,1> >();
	registerType< std::array<std::array<double,2>,2> >();
	registerType< std::array<std::array<double,3>,3> >();
}


// Registers all predefined MPI_Datatypes in 'mpi_datatype_map_'
void MpiCommunicator::registerPrimitiveMpiDatatypes()
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


// FIXME Copy other member variables
MpiCommunicator::MpiCommunicator(const MpiCommunicator& communicator)
{
#ifdef MPI_ENABLED
	setCommunicator(communicator.communicator_);
#else
	// Suppress "unused arguments" warnings
	(void) communicator;
#endif /* MPI_ENABLED */
}


bool MpiCommunicator::is_mpi_enabled()
{
	return MpiEnvironment::is_enabled();
}


bool MpiCommunicator::is_mpi_initialized()
{
	return MpiEnvironment::is_initialized();
}


MpiCommunicator::~MpiCommunicator() 
{
#ifdef MPI_ENABLED
	// Deallocate the duplicated communicator
	MPI_Comm_free(&communicator_);
#endif /* MPI_ENABLED */
}

// Returns MPI_COMM_WORLD if MPI is enabled, else returns a dummy MPI_Comm
MPI_Comm MpiCommunicator::get_mpi_comm_world()
{
#ifdef MPI_ENABLED
	return MPI_COMM_WORLD;
#else
	return MPI_Comm();
#endif
}

// Returns MPI_COMM_SELF if MPI is enabled, else returns a dummy MPI_Comm
MPI_Comm MpiCommunicator::get_mpi_comm_self()
{
#ifdef MPI_ENABLED
	return MPI_COMM_SELF;
#else
	return MPI_Comm();
#endif
}

// Definition of static variable
const MpiData MpiCommunicator::mpi_in_place;


//-------------------------------------------------//
//----- Get/Set Functions for MpiCommunicator -----//
//-------------------------------------------------//


void MpiCommunicator::setCommunicator(const MPI_Comm& communicator)
{
#ifdef MPI_ENABLED
	if ( is_mpi_initialized() ) {
		if ( is_communicator_initialized_ ) {
			// Free the old communicator
			MPI_Comm_free(&communicator_);
		}

		MPI_Comm_dup(communicator, &communicator_);
		is_communicator_initialized_ = true;
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}

#else
	// Set dummy communicator
	communicator_ = communicator;

	/*
	// Suppress "unused function" warnings
	int argc_dummy = 0;
	char** argv_dummy = nullptr;
	MPI_Init(&argc_dummy, &argv_dummy);
	MPI_Finalize();
	*/
#endif /* MPI_ENABLED */
}


void MpiCommunicator::setCommunicator(void* communicator_ptr)
{
	if ( communicator_ptr != nullptr ) {
		setCommunicator( *(static_cast<MPI_Comm*>(communicator_ptr)) );
	}
	else {
		throw std::runtime_error("communicator pointer is null");
	}
}


int MpiCommunicator::getRank() const
{
	int rank = 0;
#ifdef MPI_ENABLED
	if ( is_mpi_initialized() ) { 
		MPI_Comm_rank(communicator_, &rank);
	}
	else {
		throw MpiEnvironment::MpiUninitializedException(); 
	}
#endif /* MPI_ENABLED */
	return rank;
}


int MpiCommunicator::getSize() const
{
	int size = 1;
#ifdef MPI_ENABLED
	if ( is_mpi_initialized() ) { 
		MPI_Comm_size(communicator_, &size);
	}
	else {
		throw MpiEnvironment::MpiUninitializedException(); 
	}
#endif /* MPI_ENABLED */
	return size;
}


//------------------------------------------------//
//----- Point-to-Point Communication Methods -----//
//------------------------------------------------//


// Wrapper for MPI_Send
void MpiCommunicator::send(
	const MpiData& data, const int destination, const int tag)
{
#ifdef MPI_ENABLED
	if ( is_mpi_initialized() ) {
		// MPI_Send expects non-const arguments
		auto mpi_data = const_cast<MpiData&>(data);
		MPI_Send(mpi_data.data_ptr, mpi_data.size, mpi_data.data_type, //.get_MPI_Datatype(),
		         destination, tag, this->communicator_);
	}	
	else { 
		throw MpiEnvironment::MpiUninitializedException(); 
	}
#else
	(void) data; (void) destination; (void) tag;
	throw MpiEnvironment::MpiDisabledException();
#endif /* MPI_ENABLED */
}


// Wrapper for MPI_Recv
void MpiCommunicator::recv(
	MpiData& data, const int source, const int tag, MpiStatus& status)
{
#ifdef MPI_ENABLED
	if ( is_mpi_initialized() ) {
		if ( status.ignore() ) {
			// Ignore status
			MPI_Recv(data.data_ptr, data.size, data.data_type, //.get_MPI_Datatype(),
							 source, tag, this->communicator_, MPI_STATUS_IGNORE);
		}
		else {
			MPI_Recv(data.data_ptr, data.max_size, data.data_type, //.get_MPI_Datatype(),
							 source, tag, this->communicator_, &status.access_MPI_Status());
		}
	}
	else {
		throw MpiEnvironment::MpiUninitializedException(); 
	}	
#else
	(void) data; (void) source; (void) tag; (void) status;
	throw MpiEnvironment::MpiDisabledException();
#endif /* MPI_ENABLED */
}


// Wrapper for MPI_Isend
void MpiCommunicator::Isend(
	const MpiData& data, const int destination, const int tag, MpiRequest& request)
{
#ifdef MPI_ENABLED
	if ( is_mpi_initialized() ) {
		MPI_Isend(data.data_ptr, data.size, data.data_type, //.get_MPI_Datatype(),
		          destination, tag, this->communicator_, &request.access_MPI_Request());
	}
	else {
		throw MpiEnvironment::MpiUninitializedException(); 
	}	
#else
	(void) data; (void) destination; (void) tag; (void) request;
	throw MpiEnvironment::MpiDisabledException();
#endif /* MPI_ENABLED */
}


// Wrapper for MPI_Irecv
void MpiCommunicator::Irecv(
	MpiData& data, const int source, const int tag, MpiRequest& request)
{
#ifdef MPI_ENABLED
	if ( is_mpi_initialized() ) {
		MPI_Irecv(data.data_ptr, data.size, data.data_type, //.get_MPI_Datatype(),
		          source, tag, this->communicator_, &request.access_MPI_Request());
	}
	else {
		throw MpiEnvironment::MpiUninitializedException(); 
	}	
#else
	(void) data; (void) source; (void) tag; (void) request;
	throw MpiEnvironment::MpiDisabledException();
#endif /* MPI_ENABLED */
}


//------------------------------------//
//----- Collective Communication -----//
//------------------------------------//


// Wrapper for MPI_Barrier
void MpiCommunicator::barrier() 
{
#ifdef MPI_ENABLED
	if ( is_mpi_initialized() ) {
		MPI_Barrier(this->communicator_);
	}
	else { 
		throw MpiEnvironment::MpiUninitializedException(); 
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif /* MPI_ENABLED */
}


// Wrapper for MPI_Bcast
void MpiCommunicator::bcast(MpiData& data, const int root)
{
#ifdef MPI_ENABLED
	if ( is_mpi_initialized() ) {
		MPI_Bcast(data.data_ptr, data.size, data.data_type, //.get_MPI_Datatype(),
		          root, this->communicator_);
	}
	else { 
		throw MpiEnvironment::MpiUninitializedException(); 
	}
#else
	(void) data; (void) root;
	throw MpiEnvironment::MpiDisabledException();
#endif /* MPI_ENABLED */
}


// Wrapper for MPI_Allreduce
void MpiCommunicator::allreduce(
	const MpiData& data_in, MpiData& data_out, const MpiOp& op)
{
#ifdef MPI_ENABLED
	if ( is_mpi_initialized() ) {
		if ( &data_in == &mpi_in_place ) {
			// Perform allreduce in place
			MPI_Allreduce(MPI_IN_PLACE, data_out.data_ptr, data_out.size, data_out.data_type, //.get_MPI_Datatype(),
			              op.mpi_op_, this->communicator_);
		}
		else {
			// Perform allreduce and place result in output buffer
			// - MPI_Allreduce expects non-constant inputs
			auto data_in_ref = const_cast<MpiData&>(data_in);
			if ( data_in_ref.data_type == data_out.data_type and
			     data_in_ref.size      == data_out.size 
			) {
				MPI_Allreduce(data_in_ref.data_ptr, data_out.data_ptr, data_out.size, data_out.data_type, //.get_MPI_Datatype(),
											op.mpi_op_, this->communicator_);
			}
			else {
				throw std::runtime_error("allreduce: data type and/or size mismatch");
			}
		}
	}
	else { 
		throw MpiEnvironment::MpiUninitializedException(); 
	}
#else
	(void) data_in; (void) data_out; (void) op;
	throw MpiEnvironment::MpiDisabledException();
#endif /* MPI_ENABLED */
}


// Wrapper around underlying MPI_Allgather
void MpiCommunicator::allgather(
	const MpiData& send_data,
	MpiData& recv_data)
{
#ifdef MPI_ENABLED
	if ( is_mpi_initialized() ) {
		// Check input
		if ( send_data.data_type != recv_data.data_type ) {
			throw std::runtime_error("Error in MPI allgather: data type mismatch");
		}

		// MPI_Allgather expects non-const inputs
		auto send_data_ref = const_cast<MpiData&>(send_data);

		MPI_Allgather(send_data_ref.data_ptr, send_data_ref.size, send_data_ref.data_type, //.get_MPI_Datatype(),
		              recv_data.data_ptr,     send_data_ref.size, send_data_ref.data_type, //.get_MPI_Datatype(),
		              this->communicator_);
	}
	else {
		throw MpiEnvironment::MpiUninitializedException(); 
	}
#else
	(void) send_data; (void) recv_data;
	throw MpiEnvironment::MpiDisabledException();
#endif /* MPI_ENABLED */
}


void MpiCommunicator::allgatherv(
		const MpiData& send_data, const std::vector<int>& recv_counts, const std::vector<int>& recv_offsets,
		MpiData& recv_data)
{
#ifdef MPI_ENABLED
	if ( is_mpi_initialized() ) {
		// MPI_Allgatherv expects non-constant send buffer
		auto send_data_ref = const_cast<MpiData&>(send_data);
		MPI_Allgatherv( send_data_ref.data_ptr, send_data_ref.size, send_data_ref.data_type,
		                recv_data.data_ptr, recv_counts.data(), recv_offsets.data(), recv_data.data_type, 
		                this->communicator_ );
	}
	else { 
		throw MpiEnvironment::MpiUninitializedException(); 
	}
#else
	(void) data; (void) recv_offsets; (void) recv_counts; (void) recv_data;
	throw MpiEnvironment::MpiDisabledException();
#endif /* MPI_ENABLED */
}


void MpiCommunicator::Iallgatherv(
		const MpiData& data, const std::vector<int>& recv_counts, const std::vector<int>& recv_offsets,
		MpiData& recv_data, MpiRequest& request)
{
#ifdef MPI_ENABLED
	if ( is_mpi_initialized() ) {
		// MPI_Allgatherv expects non-constant send buffer
		auto data_ref = const_cast<MpiData&>(data);
		MPI_Iallgatherv( data_ref.data_ptr, data_ref.size, data_ref.data_type,
		                 recv_data.data_ptr, recv_counts.data(), recv_offsets.data(), recv_data.data_type,
		                 this->communicator_, &request.access_MPI_Request() );
	}
	else { 
		throw MpiEnvironment::MpiUninitializedException(); 
	}
#else
	(void) data; (void) recv_offsets; (void) recv_counts; (void) recv_data;
	throw MpiEnvironment::MpiDisabledException();
#endif /* MPI_ENABLED */
}
