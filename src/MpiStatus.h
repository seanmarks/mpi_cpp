// MpiStatus - wrapper around MPI_Status

#pragma once
#ifndef MPI_STATUS_H
#define MPI_STATUS_H

#include "Assert.h"
#include "MpiDatatype.h"
#include "MpiEnvironment.h"


#ifndef MPI_ENABLED
class MPI_Status
{
 public:
	int count      = 0;
	int cancelled  = 0;
	int MPI_SOURCE = 0;
	int MPI_TAG    = 0;
	int MPI_ERROR  = 0;
};
#endif // MPI_ENABLED


// Wrapper around MPI_Status
// - Note: MPI_STATUS_IGNORE is a named constant of unspecified type, not 'MPI_Status'
class MpiStatus {
 public:
	MpiStatus(bool ignore = false):
		ignore_(ignore) {}

	// To get an MpiStatus object that behaves as MPI_STATUS_IGNORE, use this constructor
	static MpiStatus Ignore() {
		bool ignore_status = true;
		return MpiStatus(ignore_status);
	}

	// Get the number of elements transferred
	// - Wrapper for private method that deals with actual MPI_Datatype
	template<typename T>
	int getCount() const;

	// Whether this status behaves as MPI_STATUS_IGNORE
	bool ignore() const {
		return ignore_;
	}
	void setIgnore(const bool ignore) {
		ignore_ = ignore;
	}

	// Get status properties
	bool isCancelled() const {
		FANCY_ASSERT( ! ignore(), "improper use of ignored status" );
		int flag = 1;
#ifdef MPI_ENABLED
		MPI_Test_cancelled(&status_, &flag);
#endif // ifdef MPI_ENABLED
		return static_cast<bool>(flag);
	}
	int getSource() const {
		FANCY_ASSERT( ! ignore(), "improper use of ignored status" );
		return status_.MPI_SOURCE;
	}
	int getTag() const {
		FANCY_ASSERT( ! ignore(), "improper use of ignored status" );
		return status_.MPI_TAG;
	}
	int getError() const {
		FANCY_ASSERT( ! ignore(), "improper use of ignored status" );
		return status_.MPI_ERROR;
	}

	// Access underlying struct
	const MPI_Status& get_MPI_Status() const { return status_; };
	MPI_Status&       access_MPI_Status()    { return status_; };

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
	// FIXME: What about other registered Datatypes?
	return getCount( MpiDatatype::map_primitive_MPI_Datatype<T>() );
}

#endif // ifndef MPI_STATUS_H
