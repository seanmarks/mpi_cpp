#include "MpiStatus.h"

int MpiStatus::getCount() const
{
	FANCY_ASSERT( ! ignore(),                     "improper use of ignored status" );
	FANCY_ASSERT( datatype_ != MPI_DATATYPE_NULL, "uninitialized datatype" );

	int count = 0;

#ifdef MPI_ENABLED
	if ( MpiEnvironment::is_initialized() ) {
		// Perform const_cast since MPI_Get_count() may expect a non-const MPI_Status
		MPI_Get_count(const_cast<MPI_Status*>(&status_), datatype_, &count);
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
