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
#endif /* MPI_ENABLED */
}
