// MpiEnvironment: Check the whether MPI is enabled, and initialize/finalize it
// TODO:
// - Move to static MpiEnvironment class?
// - Add overloads for init() that take arguments?

#ifndef MPI_ENVIRONMENT_H
#define MPI_ENVIRONMENT_H

// Turn on MPI_ENABLED when compiling for PLUMED with MPI
#if ( defined(__PLUMED_HAS_MPI) && ! defined(MPI_ENABLED) )
#define MPI_ENABLED
#endif

#ifdef MPI_ENABLED
#include <mpi.h>
#endif /* MPI_ENABLED */

#include <exception>
#include <stdexcept>


namespace MpiEnvironment {

// Query the status of the MPI environment
bool is_enabled();
bool is_initialized();

// Wrappers around MPI_Init/MPI_Finalize
void initialize();
void finalize();

// Alias for initialize()
void init();


//----- MPI Exceptions -----//

class MpiDisabledException : public std::exception {
 public:
	MpiDisabledException() {}
	const char* what() const noexcept override {
		return "MPI is not enabled";
	};
};

class MpiUninitializedException : public std::exception {
 public:
	MpiUninitializedException() {}
	const char* what() const noexcept override {
		return "MPI has not been initialized";
	};
};

} // end namespace MpiEnvironment


#endif /* MPI_ENVIRONMENT_H */
