
#include <iostream>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "MpiEnvironment.h"
#include "MpiCommunicator.h"

int main(int argc, char* argv[]) 
{
	// Start MPI
	MpiEnvironment::initialize();

	std::unordered_map<std::type_index, std::string> type_map;
	//std::map<std::type_index, std::string> type_map;
	type_map[std::type_index(typeid(int))] = "int";
	type_map[std::type_index(typeid(double))] = "double";

	int i = 10;
	std::cout << "i is " << type_map[std::type_index(typeid(i))] << '\n';

	//
	{
		MpiCommunicator comm( MpiCommunicator::get_mpi_comm_world() );
		int my_rank = comm.getRank();
		int num_ranks = comm.getSize();

		// Prefix output with rank
		std::string prefix = "(rank " + std::to_string(my_rank) + ") ";

		std::cout << prefix << "Testing allreduce_sum_in_place() for a single number\n";
		int sum = 1;
		comm.allreduceInPlace(sum, MpiOp::StandardOp::Sum);
		if ( sum != num_ranks ) {
			std::stringstream err_ss;
			err_ss << prefix << "ERROR: sum is " << sum << " (should be = num_ranks = " << num_ranks << ")\n";
			throw std::runtime_error( err_ss.str() );
		}

		sum = 0;
		for ( int j=0; j<num_ranks; ++j ) {
			sum += j;
		}

		std::cout << prefix << "Testing allreduce_sum_in_place() for an array\n";
		const int DIM = 3;
		std::array<int,DIM> arr;  arr.fill(my_rank);
		comm.allreduceInPlace(arr, MpiOp::StandardOp::Sum);
		for ( int d=0; d<DIM; ++d ) {
			if ( arr[d] != sum ) {
				std::cerr << prefix << "ERROR: arr[" << d << "] = " << arr[d] << " (should be " << sum << ")\n";
			}
			/*
			else {
				std::cout << prefix << "Success!\n";
			}
			*/
		}

		std::cout << prefix << "Testing allreduce_sum_in_place() for a vector\n";
		std::vector<double> vec(DIM, my_rank);
		comm.allreduceInPlace(vec, MpiOp::StandardOp::Sum);
		for ( int d=0; d<DIM; ++d ) {
			if ( vec[d] != sum ) {
				std::cerr << prefix << "ERROR: vec[" << d << "] = " << arr[d] << " (should be " << sum << ")\n";
			}
		}

		std::cout << prefix << "Testing allreduce_sum_in_place() for a complex number\n";
		std::complex<double> z(my_rank, my_rank);
		comm.allreduceInPlace(z, MpiOp::StandardOp::Sum);
		if ( z.real() != sum ) {
			std::cerr << prefix << "ERROR: Re(z) is " << z.real() << " (should be " << sum << ")\n";
		}
		if ( z.imag() != sum ) {
			std::cerr << prefix << "ERROR: Im(z) is " << z.imag() << " (should be " << sum << ")\n";
		}

		// Test Isend
		int destination = my_rank + 1;
		if ( destination == num_ranks ) { destination = 0; }
		int tag = my_rank;
		MpiRequest isend_request;
		comm.Isend(my_rank, destination, tag, isend_request);
		std::cout << prefix << "Sending message to rank " << destination << " with tag " << tag << "\n";

		// Test Irecv
		int number;
		int source = my_rank - 1;
		if ( source < 0 ) { source = num_ranks - 1; }
		MpiRequest irecv_request;
		comm.Irecv(number, source, source, irecv_request);
		std::cout << prefix << "Receiving message from rank " << source << " with tag " << source << "\n";

		MpiStatus mpi_status_ignore{ MpiStatus::Ignore() };
		isend_request.wait( mpi_status_ignore );
		irecv_request.wait( mpi_status_ignore );
		//comm.recv(number, source, source);

		std::cout << prefix << "I received " << number << " from rank " << source << "\n";

		std::cout << prefix << "Done\n";

		comm.barrier();
	}

	// Finish MPI
	MpiEnvironment::finalize();
}
