// MpiStatus - wrapper around MPI_Status
// - Written by Sean M. Marks (https://github.com/seanmarks)
//
// NOTES
// - Note: MPI_STATUS_IGNORE is a named constant, not an MPI_Status
//
// DEVELOPMENT (TODO)
// - Convert to template based on the associated message?
//   - Will have to be done if MpiRequest is converted to a template class

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


class MpiStatus {
 public:
	// Default: don't ignore this status
	MpiStatus(bool ignore = false) noexcept:
		ignore_(ignore) {}

	// To get an MpiStatus object that behaves as MPI_STATUS_IGNORE, use this constructor
	// - An alternative would have been to use a static const MpiStatus to represent MPI_STATUS_IGNORE,
	//   but many functions require a mutable reference as an argument.
	static MpiStatus Ignore() noexcept {
		bool ignore_status = true;
		return MpiStatus(ignore_status);
	}

	// Get the number of elements transferred
	// - Throws if the the status is being ignored, or the datatype is null
	int getCount() const;

	// Whether this status behaves as MPI_STATUS_IGNORE
	bool ignore() const {
		return ignore_;
	}
	void setIgnore(const bool ignore) {
		ignore_ = ignore;
	}

	// Get status properties
	bool isCancelled() const;
	int getSource()    const;
	int getTag()       const;
	int getError()     const;

	// Access underlying struct
	const MPI_Status& get_MPI_Status() const { return status_; };
	MPI_Status&       access_MPI_Status()    { return status_; };

	// Sets the datatype associated with this status
	// - Necessary for proper use of getCount()
	void setDatatype(const MPI_Datatype& datatype) {
		datatype_ = datatype;
	}

	const MPI_Datatype& getDatatype() const {
		return datatype_;
	}

 private:
	// Underlying MPI Status
	MPI_Status status_;

	// Whether this instance should be treated like MPI_STATUS_IGNORE
	bool ignore_ = false;  

	MPI_Datatype datatype_ = MPI_DATATYPE_NULL;
};


inline
bool MpiStatus::isCancelled() const {
	FANCY_ASSERT( ! ignore(), "improper use of ignored status" );
	int flag = 1;
#ifdef MPI_ENABLED
	MPI_Test_cancelled(&status_, &flag);
#endif // ifdef MPI_ENABLED
	return static_cast<bool>(flag);
}


inline
int MpiStatus::getSource() const {
	FANCY_ASSERT( ! ignore(), "improper use of ignored status" );
	return status_.MPI_SOURCE;
}


inline
int MpiStatus::getTag() const {
	FANCY_ASSERT( ! ignore(), "improper use of ignored status" );
	return status_.MPI_TAG;
}


inline
int MpiStatus::getError() const {
	FANCY_ASSERT( ! ignore(), "improper use of ignored status" );
	return status_.MPI_ERROR;
}


/*
template<typename T>
int MpiStatus::getCount() const 
{
	FANCY_ASSERT( ! ignore(), "improper use of ignored status" );

	// FIXME: General MpiDatatype
	return getCount( MpiDatatype::map_primitive_MPI_Datatype<T>() );
}
*/

#endif // ifndef MPI_STATUS_H
