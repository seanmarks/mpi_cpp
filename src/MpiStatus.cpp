#include "MpiStatus.h"

int MpiStatus::getCount(const MPI_Datatype& data_type) const
{
	FANCY_ASSERT( ! ignore(), "improper use of ignored status" );

	int count = 0;
#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		// Perform const_cast since MPI_Get_count() can expect a non-const MPI_Status
		MPI_Get_count(const_cast<MPI_Status*>(&status_), data_type, &count);
	}
	else {
		throw MpiEnvironment::MpiUninitializedException();  
	}
#else
	(void) data_type;
	throw MpiEnvironment::MpiDisabledException();
#endif /* MPI_ENABLED */
	return count;
}
