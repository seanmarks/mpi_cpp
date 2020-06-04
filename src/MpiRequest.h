// MpiRequest - wraps MPI_Request

#pragma once
#ifndef MPI_REQUEST_H
#define MPI_REQUEST_H

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

	// Wrapper for MPI_Wait
	void wait(MpiStatus& mpi_status);

	// TODO: MPI_Probe, MPI_Test, ...

	const MPI_Request& get_MPI_Request() const { return request_; };
	MPI_Request&       access_MPI_Request()    { return request_; };

 private:
	// Underlying MPI Request
	MPI_Request request_;

	bool is_active_ = false;  // TODO

	// Ptr to the communicator this request is associated with
	const MpiCommunicator* comm_ptr_ = nullptr;  // TODO
};

#endif // ifndef MPI_REQUEST_H
