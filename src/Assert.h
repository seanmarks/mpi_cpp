// Assert: quick exception throwing with assert-like syntax and informative messages

#ifndef ASSERT_H
#define ASSERT_H

#include <exception>
#include <sstream>
#include <string>
#include <stdexcept>

// Convert 'x' to a string using arcane preprocessor tricks
#define STRINGIFY(x) #x
#define TO_STRING(x) STRINGIFY(x)

// Prints the file name and line number where it's expanded
#define LOCATION_IN_SOURCE __FILE__ ":" TO_STRING(__LINE__)

// Wrap text macros in std::string for greater flexibility
#define LOCATION_IN_SOURCE_STRING std::string(LOCATION_IN_SOURCE)
#define PRETTY_FUNCTION_STRING std::string(__PRETTY_FUNCTION__) 

// Prints the "pretty" name of the function along in which the macro is expanded,
// along with the associated file name and line number
#define FANCY_FUNCTION "\"" + PRETTY_FUNCTION_STRING + "\" " \
		"(" + LOCATION_IN_SOURCE_STRING + ")"

// "Fancy assert" - quick exception checking
// - 'message' is very flexible: you can pass it anything that can be
//   handled by a std::stringstream. For example:
//      FANCY_ASSERT( i > 0,
//                    "unexpected value: i = " << i );
//
#define FANCY_ASSERT(test,message) if (not (test)) {                \
		std::stringstream err_ss;                                       \
		err_ss << "assertion failed\n"                                  \
		       << "  function:  " << PRETTY_FUNCTION_STRING << "\n"     \
		       << "  where:     " << LOCATION_IN_SOURCE_STRING << "\n"  \
		       << "  message:   " << message << "\n"                    \
		       << "  test:      " << STRINGIFY(test) << std::endl;      \
  	throw std::runtime_error( err_ss.str() );                       \
	}

// Fancy asserts that are only run in debug mode
#ifndef NDEBUG
#  define FANCY_DEBUG_ASSERT(test,message) FANCY_ASSERT((test),message)
#else
#  define FANCY_DEBUG_ASSERT(test,message)
#endif // ifndef DEBUG

#endif // ifndef ASSERT_H
