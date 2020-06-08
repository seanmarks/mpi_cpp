// MpiRequest - wraps MPI_Request
// - Written by Sean M. Marks (https://github.com/seanmarks)
//
// DEVELOPMENT (TODO)
// - Convert to template based on the associated message?
//   - Would more closely link the request with the data it represents
//     and prevent some misuses of requests intended for use with different types

#pragma once
#ifndef MPI_REQUEST_H
#define MPI_REQUEST_H

#include "MpiDatatype.h"
#include "MpiEnvironment.h"
#include "MpiStatus.h"

class MpiCommunicator;

#ifndef MPI_ENABLED
class MPI_Request {};
#endif // MPI_ENABLED

class MpiRequest {
 public:
	MpiRequest() {}

	MpiRequest(const MPI_Request& request):
		request_(request) {}

	// Finalize the request, if still active
	// - Prevents dangling requests
	~MpiRequest();

	// It is probably not safe to copy requests
	MpiRequest(const MpiRequest&) = delete;

	// Wrapper for MPI_Wait
	void wait(MpiStatus& status);

	// Returns true if the request has been completed and results are available
	// - The user must still subsequently call 'wait()' to wrap up the request
	bool test(MpiStatus& mpi_status);

	const MPI_Request& get_MPI_Request() const { return request_; };
	MPI_Request&       access_MPI_Request()    { return request_; };

	// Indicates whether this request represents a pending communication
	// - True  --> need to finalize request (extract info of interest, then 'wait()')
	// - False --> nothing to do
	bool isActive() {
		return is_active_;
	}

	// Call to indicate that this request represents a pending communication
	// of the given datatype
	void setActive(const MPI_Datatype& datatype) {
		is_active_ = true;
		datatype_  = datatype;
	}

	const MPI_Datatype& getDatatype() const {
		return datatype_;
	}

 private:
	// Underlying MPI Request
	MPI_Request request_;

	bool is_active_ = false;

	MPI_Datatype datatype_  = MPI_DATATYPE_NULL;
};

#endif // ifndef MPI_REQUEST_H
