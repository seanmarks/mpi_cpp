#include "MpiRequest.h"

void MpiRequest::wait(MpiStatus& status)
{
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		if ( status.ignore() ) {
			MPI_Wait(&request_, MPI_STATUS_IGNORE);
		}
		else {
			MPI_Wait(&request_, &status.access_MPI_Status());
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
		int flag = 0;
		if ( status.ignore() ) {
			MPI_Test(&request_, &flag, MPI_STATUS_IGNORE);
		}
		else {
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
