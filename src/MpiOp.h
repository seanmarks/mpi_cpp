// MpiOp - Wrapper around MPI_Op
// - Handles allocation and deallocation
//
// - FIXME: moving/copying

#pragma once
#ifndef MPI_OP_H
#define MPI_OP_H

#include <functional>
#include <sstream>
#include <type_traits>
#include <unordered_map>

#include "MpiDatatype.h"
#include "MpiEnvironment.h"
#include "utils.h"


#ifndef MPI_ENABLED
class MPI_Op {};
class MPI_User_function {};
//class MPI_Datatype;
#endif


class MpiOp 
{
 public:
	// A user-defined std::function that operates on type T
	template <typename T> using MpiUserFunction = std::function<void(T*,T*,int*,MPI_Datatype*)>;

	// Function pointer corresponding to the above std::function
	template <typename T> using MpiUserFunctionPtr = void(*)(T*,T*,int*,MPI_Datatype*);

	// Standard MPI operations
	enum class StandardOp {
		Null, 
		Max, Min, Sum, Product, 
		Land, Band, Lor, Bor, Lxor, Bxor, 
		Minloc, Maxloc, 
		Replace
	};

	// In C++14, this can be removed because std::hash supports enum classes
	// Source: https://stackoverflow.com/questions/18837857/cant-use-enum-class-as-unordered-map-key
	// - User "Daniel"
	struct EnumClassHash {
		template <typename T>
		std::size_t operator()(T t) const {
				return static_cast<std::size_t>(t);
		}
	};
	using StandardOpsMap = std::unordered_map<StandardOp, MpiOp, EnumClassHash>;

	MpiOp();

	// Construct an MpiOp for the indicated type and function, and register it with the MPI core
	template<typename T>
	MpiOp(MpiUserFunction<T> const& user_function, const bool is_commutative );

	// Copy constructor 
	MpiOp(const MpiOp& mpi_op);

	// Copy assignment operator 
	MpiOp& operator=(const MpiOp& mpi_op);

	~MpiOp();

	const MPI_Op& get_MPI_Op() const {
		return mpi_op_;
	}

	// Get a string corresponding to the operation
	static const std::string& getName(const StandardOp& op);

 private:
	// Underlying MPI_Op struct
	MPI_Op mpi_op_;

	// Store the variables necessary to re-create the MPI_Op when the
	// object is copied
	MPI_User_function* user_function_ptr_ = nullptr;
	bool is_commutative_ = false;

	static const std::unordered_map<StandardOp, std::string, EnumClassHash> standard_mpi_op_names_;
};



// Construct an MpiOp for the indicated type and function, and register it with the MPI core
template<typename T>
MpiOp::MpiOp(MpiUserFunction<T> const& user_function, const bool is_commutative ):
	user_function_ptr_(nullptr),
	is_commutative_(is_commutative)
{
	// Get a pointer to the underlying function pointer
	const MpiUserFunctionPtr<T>* user_function_ptr = user_function.template target< MpiUserFunctionPtr<T> >();
	if ( user_function_ptr == nullptr ) {
		throw std::runtime_error("Unable to extract function ptr from std::function in MpiOp constructor");
	}

	// Cast to MPI types
	user_function_ptr_ = (MPI_User_function*)(*user_function_ptr);
	
#ifdef MPI_ENABLED
	MPI_Op_create(user_function_ptr_, is_commutative_, &mpi_op_);
#endif // ifdef MPI_ENABLED
}


#endif // ifndef MPI_OP_H
