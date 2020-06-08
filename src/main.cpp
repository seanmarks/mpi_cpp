
#include <iostream>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "MpiEnvironment.h"
#include "MpiCommunicator.h"

#include "MpiData.h"
#include "MpiDataTraits.h"
#include "MpiDatatype.h"


int main(int argc, char* argv[])
{
	// Start MPI
	MpiEnvironment::initialize();

	std::unordered_map<std::type_index, std::string> type_map;
	type_map[std::type_index(typeid(int))]    = "int";
	type_map[std::type_index(typeid(double))] = "double";
	type_map[std::type_index(typeid(double))] = "float";

	/*
	int i = 10;
	std::cout << "i is " << type_map[std::type_index(typeid(i))] << '\n';
	*/

	//
	{
		MpiCommunicator comm = MpiCommunicator::World();
		const int my_rank     = comm.getRank();
		const int num_ranks   = comm.getSize();
		const int master_rank = comm.getMasterRank();
		const bool is_master  = comm.isMasterRank();

		FANCY_ASSERT( num_ranks >= 2, "these tests cannot be done without at least " << num_ranks << " ranks" );

		// Prefix output with rank
		std::string prefix = "(rank " + std::to_string(my_rank) + ") ";

		int sum = 0;
		for ( int j=0; j<num_ranks; ++j ) {
			sum += j;
		}
		const int sum_of_ranks = sum;

		/*
		//MpiData mpi_data_scalar(scalar, comm);  // with MpiComm input
		//MpiData mpi_data_scalar(scalar, MpiDatatype::map_primitive_MPI_Datatype<int>());  // with MPI_Datatype input
		std::cout << "mpi_data_scalar.getSize()     = " << mpi_data_scalar.getSize()    << std::endl;
		std::cout << "mpi_data_scalar.getCapacity() = " << mpi_data_scalar.getCapacity() << std::endl;
		comm.barrier();
		*/

		//MpiData mpi_data_vec(vec, comm);  // with MpiComm input
		//MpiData mpi_data_vec(vec, MpiDatatype::map_primitive_MPI_Datatype<double>());  // with MPI_Datatype input
		/*
		std::cout << "mpi_data_vec.getSize()     = " << mpi_data_vec.getSize()    << std::endl;
		std::cout << "mpi_data_vec.getCapacity() = " << mpi_data_vec.getCapacity() << std::endl;
		comm.barrier();
		*/


		//----- Test send/recv -----//

		if ( is_master ) {
			std::cout << "Test send/recv" << std::endl;
		}
		comm.barrier();

		int scalar     = my_rank;
		int scalar_tag = 0;

		for ( int r=0; r<num_ranks; ++r ) {
			// Rank r sends a message to r+1
			int source = r;
			int dest = r + 1;
			if ( dest >= num_ranks ) { dest = 0; }

			int scalar_recv;
			if ( my_rank == source ) {
				comm.send( scalar, dest, scalar_tag );
			}
			else if ( my_rank == dest ) {
				MpiStatus status;
				comm.recv( scalar_recv, source, scalar_tag, status );

				// Check number of values received
				int count = status.getCount();
				FANCY_ASSERT( count == 1,
				              prefix << " received " << count << " values, expected 1" );

				// Check value received
				FANCY_ASSERT( scalar_recv == source,
				              prefix << " received " << scalar_recv << ", expected " << source );
			}
		}


		//----- Test Isend/Irecv -----//

		if ( is_master ) {
			std::cout << "Test Isend/Irecv" << std::endl;
		}
		comm.barrier();

		int len = 10;
		std::vector<double> vec(len, my_rank);
		int vec_tag = scalar_tag + 1;

		// Each rank sends a message to my_rank+1 and receives a message from my_rank-1 (with wrap-around)
		int dest_Isend = my_rank + 1;
		if ( dest_Isend >= num_ranks ) { dest_Isend = 0; }
		int source_Irecv = my_rank - 1;
		if ( source_Irecv < 0 ) { source_Irecv = num_ranks-1; }

		MpiRequest scalar_Isend_request;
		comm.Isend( scalar, dest_Isend, scalar_tag, scalar_Isend_request );

		MpiRequest scalar_Irecv_request;
		int scalar_Irecv;
		comm.Irecv( scalar_Irecv, source_Irecv, scalar_tag, scalar_Irecv_request );

		// Wait for data
		MpiStatus ignore_status = MpiStatus::Ignore();
		scalar_Irecv_request.wait( ignore_status );

		FANCY_ASSERT( scalar_Irecv == source_Irecv,
		              prefix << " received " << scalar_Irecv << ", expected " << source_Irecv );

		// Clean up send request
		MpiStatus status;
		scalar_Isend_request.wait( status );


		//----- Test bcast -----//

		if ( is_master ) {
			std::cout << "Test bcast" << std::endl;
		}
		comm.barrier();

		// Rank 1 sends a unique array to all other ranks
		int bcast_root = 1;

		double value          = 13.0*my_rank;
		double expected_value = 13.0*bcast_root;

		constexpr int N_DIM = 3;
		std::array<double, N_DIM> arr;
		arr.fill(value);

		comm.bcast( arr, bcast_root );

		for ( int d=0; d<N_DIM; ++d ) {
			FANCY_ASSERT( arr[d] == expected_value,
			              prefix << " received arr[" << d << "]=" << arr[d] << ", expected " << expected_value );
			//std::cout << arr[d] << std::endl;
		}


		//----- Test allreduce -----//

		if ( is_master ) {
			std::cout << "Test allreduce" << std::endl;
		}

		//std::cout << prefix << "Testing allreduce_sum_in_place() for a single number\n";
		sum = 1;
		comm.allreduceInPlace(sum, MpiOp::StandardOp::Sum);
		FANCY_ASSERT( sum == num_ranks,
		              prefix << "sum is " << sum << ", expected sum = num_ranks = " << num_ranks );


		//std::cout << prefix << "Testing allreduce_sum_in_place() for an array\n";
		arr.fill(my_rank);
		comm.allreduceInPlace(arr, MpiOp::StandardOp::Sum);
		for ( int d=0; d<N_DIM; ++d ) {
			FANCY_ASSERT( arr[d] == sum_of_ranks,
			              prefix << " arr[" << d << "]=" << arr[d] << ", expected " << sum_of_ranks );
		}

		//std::cout << prefix << "Testing allreduce_sum_in_place() for a vector\n";
		vec.assign(len, my_rank);
		comm.allreduceInPlace(vec, MpiOp::StandardOp::Sum);
		for ( int i=0; i<len; ++i ) {
			FANCY_ASSERT( vec[i] == sum_of_ranks,
			              prefix << "vec[" << i << "] = " << vec[i] << ", expected " << sum_of_ranks );
		}

		//std::cout << prefix << "Testing allreduce_sum_in_place() for a complex number\n";
		std::complex<double> z(my_rank, my_rank);
		comm.allreduceInPlace(z, MpiOp::StandardOp::Sum);
		FANCY_ASSERT( z.real() == sum_of_ranks,
		              prefix << "ERROR: z.real() is " << z.real() << ", expected " << sum_of_ranks );
		FANCY_ASSERT( z.imag() == sum_of_ranks,
		              prefix << "ERROR: z.imag() is " << z.imag() << ", expected " << sum_of_ranks );


		//----- Test allgather -----//

		if ( is_master ) {
			std::cout << "Test allgather" << std::endl;
		}
		comm.barrier();

		const double my_rank_d = my_rank;
		vec.resize(0);  // allgather() should resize as needed
		comm.allgather(my_rank_d, vec);

		FANCY_ASSERT( vec.size() == static_cast<unsigned>(num_ranks), "bad size: " << vec.size() );
		for ( int r=0; r<num_ranks; ++r ) {
			FANCY_ASSERT( vec[r] == r,
				            prefix << "vec[" << r << "] = " << vec[r] << ", expected " << r );
		}


		//----- Test allgatherv -----//

		if ( is_master ) {
			std::cout << "Test allgatherv" << std::endl;
		}
		comm.barrier();

		// Each rank 'r' sends a vector with 'r' pieces of data
		vec.assign(my_rank, my_rank_d);
		std::vector<int> recv_counts(num_ranks), recv_offsets(num_ranks);
		for ( int r=0; r<num_ranks; ++r ) {
			if ( r == 0 ) {
				recv_offsets[r] = 0;
			}
			else {
				recv_offsets[r] = recv_offsets[r-1] + recv_counts[r-1];
			}
			recv_counts[r] = r;

			// Ascending values, starting with local rank
			vec[r] = my_rank_d + r;
		}

		std::vector<double> buffer;
		comm.allgatherv( vec, recv_counts, buffer, recv_offsets );

		FANCY_ASSERT( recv_offsets.size() == recv_counts.size(), "bad size" );
		for ( int r=0; r<num_ranks; ++r ) {
			int offset = recv_offsets[r];
			int count  = recv_counts[r];
			FANCY_ASSERT( count == r, "bad size" );
			for ( int i=0; i<count; ++i ) {
				FANCY_ASSERT( buffer[offset+i] == r + i,
				              prefix << "buffer(" << r << "," << i << ") = " << buffer[offset+1] << ","
				              << " expected " << r+i );
			}
		}

		// Same as above, but with auto-determined counts
		recv_counts.clear();
		comm.allgathervWithUnknownCounts( vec, buffer, recv_counts, recv_offsets );
		FANCY_ASSERT( recv_counts.size() == static_cast<unsigned>(num_ranks), "bad size" );
		for ( int r=0; r<num_ranks; ++r ) {
			int offset = recv_offsets[r];
			int count  = recv_counts[r];
			FANCY_ASSERT( count == r, "bad size" );
			for ( int i=0; i<count; ++i ) {
				FANCY_ASSERT( buffer[offset+i] == r + i,
				              prefix << "buffer(" << r << "," << i << ") = " << buffer[offset+i] << ","
				              << " expected " << r+i );
			}
		}


		//----- Test scatterv -----//

		if ( is_master ) {
			std::cout << "Test scatterv" << std::endl;
		}
		comm.barrier();

		// Master rank sends 'r' pieces of data to each rank 'r'
		vec.resize(0);
		std::vector<int> send_counts(num_ranks), send_offsets(num_ranks);
		for ( int r=0; r<num_ranks; ++r ) {
			if ( r == 0 ) {
				send_offsets[r] = 0;
			}
			else {
				send_offsets[r] = send_offsets[r-1] + send_counts[r-1];
			}
			send_counts[r] = r;

			for ( int i=0; i<r; ++i ) {
				vec.push_back(i);
			}
		}

		// Send data
		int scatterv_root = master_rank;
		if ( ! is_master ) {
			vec.clear();  // non-master ranks don't get to keep their data
		}
		comm.scatterv(vec, send_counts, send_offsets, buffer, scatterv_root);

		// Check results
		FANCY_ASSERT( buffer.size() == static_cast<unsigned>(my_rank), "bad size" );
		for ( int i=0; i<my_rank; ++i ) {
			FANCY_ASSERT( buffer[i] == i,
			              prefix << "buffer[" << i << "] = " << buffer[i] << "," << " expected " << i );
		}


		/*
		const double my_rank_d = my_rank;
		vec.resize(0);  // allgather() should resize as needed
		comm.allgather(my_rank_d, vec);

		FANCY_ASSERT( vec.size() == static_cast<unsigned>(num_ranks), "bad size: " << vec.size() );
		for ( int r=0; r<num_ranks; ++r ) {
			FANCY_ASSERT( vec[r] == r,
				            prefix << "vec[" << r << "] = " << vec[r] << ", expected " << r );
		}
		*/

		/*
		// Misc. MPI op testing
		std::array<std::array<int,N_DIM>,N_DIM> matrix;
		std::cout << prefix << "sizeof(matrix) = " << sizeof(matrix) << " (" << sizeof(matrix)/(N_DIM*N_DIM) << ")\n";
		//MpiOp mpi_op_1(mpi_sum_contiguous<double>, true );
		//std::function<void(double*,double*,int*,MPI_Datatype*)> my_func = mpi_sum_contiguous<double>;
		//std::function<void(double*,double*,int*,MPI_Datatype*)> my_func( mpi_sum_contiguous<double> );
		//MpiOp mpi_op_1(my_func, true);
		//MpiOp mpi_op_1(std::function<void(double*,double*,int*,MPI_Datatype*)>(mpi_sum_contiguous<double>), true );
		//MpiOp mpi_op_2;
		*/


		/*
		// Testing MPI type extent functions

		MPI_Aint extent;
		MPI_Type_extent(this->mpi_datatype_, &extent);
		std::cout << " REGISTERED EXTENT " << extent << "\n";

		int entry = 0;
		const auto& map = MpiCommunicator::mpi_datatype_map_;
		for ( auto it = map.begin(); it != map.end(); ++it ) {
			++entry;

			// Handle to underlying type
			MPI_Datatype& data_type = const_cast<MPI_Datatype&>( it->second.get_MPI_Datatype() );

			// Get extent (i.e. number of bytes)
			MPI_Aint extent;
			MPI_Type_extent(data_type, &extent);

			std::cout << "Entry " << entry << ": extent " << extent << "\n";
			//std::cout << "Entry: " << std::string( it->second.get_MPI_Datatype()->name ) << "\n";

			MPI_Aint extent_int, extent_float, extent_double, extent_char;
			MPI_Type_extent(MPI_INT,    &extent_int);
			MPI_Type_extent(MPI_FLOAT,  &extent_float);
			MPI_Type_extent(MPI_DOUBLE, &extent_double);
			MPI_Type_extent(MPI_CHAR,   &extent_char);
			std::cout << "  DEBUG: extent_int = " << extent_int << ",  extent_float = " << extent_float << ", "
			          << "extent_double = " << extent_double << ", extent_char " << extent_char << "\n";
		}
		*/

		if ( is_master ) {
			std::cout << "\nTest: Success!" << std::endl;
		}
	}

	// Finish MPI
	MpiEnvironment::finalize();


	return 0;
}
