#include "MpiRequest.h"


MpiRequest::~MpiRequest()
{
	if ( is_active_ ) {
		MpiStatus ignore_status = MpiStatus::Ignore();
		wait( ignore_status );
	}
}


void MpiRequest::wait(MpiStatus& status)
{
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		if ( is_active_ ) {
			// Wait until the request is complete
			if ( status.ignore() ) {
				MPI_Wait(&request_, MPI_STATUS_IGNORE);
			}
			else {
				status.setDatatype(datatype_);  // ensure status is properly prepared
				MPI_Wait(&request_, &status.access_MPI_Status());
			}
			is_active_ = false;
		}
	}
	else {
		throw MpiEnvironment::MpiUninitializedException(); 
	}
#else
	(void) status;
	throw MpiEnvironment::MpiDisabledException();
#endif // MPI_ENABLED
}


bool MpiRequest::test(MpiStatus& status)
{
	bool is_completed = false;

#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		FANCY_ASSERT( is_active_, "improper use of MPI_Test for inactive request object" );

		int flag = 0;
		if ( status.ignore() ) {
			MPI_Test(&request_, &flag, MPI_STATUS_IGNORE);
		}
		else {
			status.setDatatype(datatype_);  // ensure status is properly prepared
			MPI_Test(&request_, &flag, &status.access_MPI_Status());
		}
		is_completed = static_cast<bool>(flag);
	}
	else {
		throw MpiEnvironment::MpiUninitializedException(); 
	}
#else
	(void) status;
	throw MpiEnvironment::MpiDisabledException();
#endif // MPI_ENABLED

	return is_completed;
}
