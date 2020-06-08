// Written by Sean M. Marks (https://github.com/seanmarks)

#include "MpiOp.h"


MpiOp::MpiOp(): 
	user_function_ptr_(nullptr),
	is_commutative_(false)
{}


MpiOp::MpiOp(const MpiOp& mpi_op)
{
	this->user_function_ptr_ = mpi_op.user_function_ptr_;
	this->is_commutative_ = mpi_op.is_commutative_;
#ifdef MPI_ENABLED
	if ( this->user_function_ptr_ != nullptr ) {
		MPI_Op_create(user_function_ptr_, is_commutative_, &mpi_op_);
	}
#endif // ifdef MPI_ENABLED
}


MpiOp& MpiOp::operator=(const MpiOp& mpi_op)
{
	if ( &mpi_op != this ) {
		this->user_function_ptr_ = mpi_op.user_function_ptr_;
		this->is_commutative_ = mpi_op.is_commutative_;
#ifdef MPI_ENABLED
		if ( this->user_function_ptr_ != nullptr ) {
			MPI_Op_create(user_function_ptr_, is_commutative_, &mpi_op_);
		}
#endif // ifdef MPI_ENABLED
	}
	return *this;
}


MpiOp::~MpiOp()
{
#ifdef MPI_ENABLED
	if ( user_function_ptr_ != nullptr ) {
		MPI_Op_free(&mpi_op_);
	}
#endif // ifdef MPI_ENABLED
};


const std::unordered_map<MpiOp::StandardOp, std::string, MpiOp::EnumClassHash> 
	MpiOp::standard_mpi_op_names_ =  {
	{ StandardOp::Null,    "Null"    },
	{ StandardOp::Max,     "Max"     },
	{ StandardOp::Min,     "Min"     },
	{ StandardOp::Sum,     "Sum"     },
	{ StandardOp::Product, "Product" },
	{ StandardOp::Land,    "Land"    },
	{ StandardOp::Band,    "Band"    },
	{ StandardOp::Lor,     "Lor"     },
	{ StandardOp::Bor,     "Bor"     },
	{ StandardOp::Lxor,    "Lxor"    },
	{ StandardOp::Bxor,    "Bxor"    },
	{ StandardOp::Minloc,  "Minloc"  },
	{ StandardOp::Maxloc,  "Maxloc"  },
	{ StandardOp::Replace, "Replace" }
};


const std::string& MpiOp::getName(const MpiOp::StandardOp& op) {
	const auto it = standard_mpi_op_names_.find(op);
	if ( it != standard_mpi_op_names_.end() ) {
		return it->second;
	}
	else {
		std::stringstream err_ss;
		err_ss << "Error in " << FANCY_FUNCTION << "\n"
					 << "  A standard MPI operation does not have a registered name."
		          << " This should never happen.\n";
		throw std::runtime_error( err_ss.str() );
	}
}
