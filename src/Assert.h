// Assert: quick exception throwing with assert-like syntax and informative messages

#ifndef ASSERT_H
#define ASSERT_H

#include <exception>
#include <sstream>
#include <string>
#include <stdexcept>

#include "utils.h"

// 'message' is very flexible: you can pass it anything that can be
// handled by a std::stringstream
//   e.g. FANCY_ASSERT(true, "this is " << " true");
#define FANCY_ASSERT(test,message) if (not (test)) { \
		std::stringstream err_ss; \
		err_ss << "assertion failed in " FANCY_FUNCTION "\n" \
		       << "  " << message << "\n" \
		       << "  test: " STRINGIFY(test) "\n"; \
  	throw std::runtime_error( err_ss.str() ); \
	}

#endif // ifndef ASSERT_H
