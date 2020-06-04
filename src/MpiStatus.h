// MpiStatus - wrapper around MPI_Status

#pragma once
#ifndef MPI_STATUS_H
#define MPI_STATUS_H

#include "Assert.h"
#include "MpiDatatype.h"
#include "MpiEnvironment.h"

#ifndef MPI_ENABLED
class MPI_Status {};
#endif // MPI_ENABLED

// Wrapper around MPI_Status
// - Note: MPI_STATUS_IGNORE is a named constant of unspecified type, not 'MPI_Status'
class MpiStatus {
 public:
	MpiStatus(bool ignore = false):
		ignore_(ignore) {}

	MpiStatus(const MPI_Status& status):
		status_(status) {}

	// To get an MpiStatus object that behaves as MPI_STATUS_IGNORE, use this constructor
	static MpiStatus Ignore() {
		bool ignore_status = true;
		return MpiStatus(ignore_status);
	}

	// Get the number of elements transferred
	// - Wrapper for private method that deals with actual MPI_Datatype
	template<typename T>
	int getCount() const;

	const MPI_Status& get_MPI_Status() const { return status_; };
	MPI_Status&       access_MPI_Status()    { return status_; };

	// Whether this status behaves as MPI_STATUS_IGNORE
	bool ignore() const {
		return ignore_;
	}

	void setIgnore(const bool ignore) {
		ignore_ = ignore;
	}


 private:
	// Underlying MPI Status
	MPI_Status status_;

	// Whether this instance should be treated like MPI_STATUS_IGNORE
	bool ignore_ = false;  

	// Calls MPI_Get_count on status_ to determine the number
	// of elements received
	// TODO update to MpiDatatype?
	int getCount(const MPI_Datatype& data_type) const;
};


template<typename T>
int MpiStatus::getCount() const 
{
	// TODO: update to MpiDatatype
	return getCount( MpiDatatype::map_primitive_MPI_Datatype<T>() );
}

#endif // ifndef MPI_STATUS_H
