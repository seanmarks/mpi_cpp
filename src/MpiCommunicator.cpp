// Written by Sean M. Marks (https://github.com/seanmarks)

#include "MpiCommunicator.h"

//-------------------------------------------//
//----- Constructors and Initialization -----//
//-------------------------------------------//


MpiCommunicator::MpiCommunicator()
{
#ifdef MPI_ENABLED
	setCommunicator(MPI_COMM_SELF);
#endif // ifdef MPI_ENABLED
}


MpiCommunicator MpiCommunicator::World()
{
#ifdef MPI_ENABLED
	return MpiCommunicator(MPI_COMM_WORLD);
#else
	return MpiCommunicator();
#endif // ifdef MPI_ENABLED
}


MpiCommunicator MpiCommunicator::Self()
{
#ifdef MPI_ENABLED
	return MpiCommunicator(MPI_COMM_SELF);
#else
	return MpiCommunicator();
#endif // ifdef MPI_ENABLED
}

MpiCommunicator::MpiCommunicator(const MPI_Comm& communicator)
{
#ifdef MPI_ENABLED
	setCommunicator(communicator);
#else
	// Suppress "unused arguments" warnings
	(void) communicator;
#endif // ifdef MPI_ENABLED
}


// FIXME Copy other member variables
MpiCommunicator::MpiCommunicator(const MpiCommunicator& communicator)
{
#ifdef MPI_ENABLED
	setCommunicator(communicator.communicator_);
#else
	// Suppress "unused arguments" warnings
	(void) communicator;
#endif // ifdef MPI_ENABLED
}


MpiCommunicator::~MpiCommunicator()
{
#ifdef MPI_ENABLED
	// Deallocate the duplicated communicator
	MPI_Comm_free(&communicator_);
#endif // ifdef MPI_ENABLED
}


void MpiCommunicator::probe(const int source, const int tag, MpiStatus& status) const
{
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		if ( status.ignore() ) {
			MPI_Probe(source, tag, communicator_, MPI_STATUS_IGNORE);
		}
		else {
			MPI_Probe(source, tag, communicator_, &status.access_MPI_Status());
		}
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // MPI_ENABLED
}


bool MpiCommunicator::Iprobe(const int source, const int tag, MpiStatus& status) const
{
	bool message_ready = false;

#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		int flag = 0;
		if ( status.ignore() ) {
			MPI_Iprobe(source, tag, communicator_, &flag, MPI_STATUS_IGNORE);
		}
		else {
			MPI_Iprobe(source, tag, communicator_, &flag, &status.access_MPI_Status());
		}
		message_ready = static_cast<bool>(flag);
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // MPI_ENABLED

	return message_ready;	
}


//-------------------------------------------------//
//----- Get/Set Functions for MpiCommunicator -----//
//-------------------------------------------------//


void MpiCommunicator::setCommunicator(const MPI_Comm& communicator)
{
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
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
#endif // ifdef MPI_ENABLED
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
	if ( MpiEnvironment::is_initialized() ) {
		MPI_Comm_rank(communicator_, &rank);
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#endif // ifdef MPI_ENABLED
	return rank;
}


int MpiCommunicator::getSize() const
{
	int size = 1;
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		MPI_Comm_size(communicator_, &size);
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#endif // ifdef MPI_ENABLED
	return size;
}



//------------------------------------//
//----- Collective Communication -----//
//------------------------------------//


void MpiCommunicator::barrier()
{
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		MPI_Barrier(this->communicator_);
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();
	}
#else
	throw MpiEnvironment::MpiDisabledException();
#endif // ifdef MPI_ENABLED
}
