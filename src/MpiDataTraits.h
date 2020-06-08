// MpiDataTraits - helper class for constructing MpiData buffers from different types
//
// - NOTES
//   - It might seem funny at first to have a class whose only job is to call
//     the member functions of another class. For example, the template specialization
//     for a std::vector supports a 'size' function that just calls 'size' on the
//     input vector. However, this construction is necessary for a general, extensible
//     interface for creating MpiData from an arbitrary type.
//     - For example, while a primitive type such as a 'double' does not have any
//       member functions (let alone 'size'), its MpiDataTraits specialization provides
//       the interface that MpiData (and related classes) expect.
//
//   - Several overloads are required to construct an MpiBuffer for individual elements
//     of a given type (e.g. a single std::array or std::complex)
//     - This is distinct from how the *datatype* of a single element is registered
//       (e.g. a single std::array as an MPI_Type_contiguous)
//
// FIXME
// - Variable-length buffers for recv/Irecv
//   - resize / resizeToCapacity
//   - finalize() ?
//   - Implement as a derived class of MpiDataTraits?

#pragma once
#ifndef MPI_DATA_TRAITS_H
#define MPI_DATA_TRAITS_H

#include <array>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "Assert.h"


// Primitive types
template<typename T>
struct MpiDataTraits
{
	static_assert(std::is_arithmetic<T>::value, "invalid type");

	using size_type     = std::size_t;
	using value_type    = T;
	using reference     = T&;
	using pointer       = T*;
	using const_pointer = const T*;

	static size_type size(const T& value)     { return 1; }
	static size_type capacity(const T& value) { return 1; }

	static T*       data(T& value)       { return &value; }
	static const T* data(const T& value) { return &value; }

	static void resize(T& value, const size_type new_size) {
		FANCY_ASSERT( new_size == 1, "invalid size" );
	}

	static void resizeToCapacity(T& value) {
		return;
	}
};


// std::vector
template<typename T, typename A>
struct MpiDataTraits<std::vector<T,A>>
{
	static_assert(std::is_arithmetic<T>::value, "invalid type");

	using Vector = std::vector<T,A>;
	using size_type     = std::size_t;  // typename Vector::size_type;
	using value_type    = T;            // typename Vector::value_type;
	using reference     = T&;           // typename Vector::reference;
	using pointer       = T*;           // typename Vector::pointer;
	using const_pointer = const T*;     // typename Vector::const_pointer;


	static size_type size(const Vector& vec)     { return vec.size();     }
	static size_type capacity(const Vector& vec) { return vec.capacity(); }

	static T*       data(Vector& vec)       { return vec.data(); }
	static const T* data(const Vector& vec) { return vec.data(); }

	// When a variable-length buffer is used to receive data, it should max out
	// its size beforehand, and then shrink later
	static void resize(Vector& vec, const size_type new_size) {
		vec.resize( new_size );
	}
	static void resizeToCapacity(Vector& vec) {
		vec.resize( vec.capacity() );
	}
};


// std::array
template<typename T, std::size_t N>
struct MpiDataTraits<std::array<T,N>>
{
	static_assert(std::is_arithmetic<T>::value, "invalid type");
	static_assert(N > 0, "invalid size");

	using Array = std::array<T,N>;
	using size_type     = std::size_t;
	using value_type    = Array;
	using reference     = Array&;
	using pointer       = Array*;
	using const_pointer = const Array*;

	static size_type size(const Array& array)     { return 1; }
	static size_type capacity(const Array& array) { return 1; }

	static T*       data(Array&       array) { return array.data(); }
	static const T* data(const Array& array) { return array.data(); }

	static void resize(Array& array, const size_type new_size) {
		FANCY_ASSERT( new_size == 1, "invalid size" );
	}

	static void resizeToCapacity(Array& array) {
		return;
	}
};


// Matrix-like type (std::array of std::array)
template<typename T, std::size_t NR, std::size_t NC>
struct MpiDataTraits<std::array<std::array<T,NC>, NR>>
{
	static_assert(std::is_arithmetic<T>::value, "invalid type");
	static_assert(NR > 0, "invalid size");
	static_assert(NC > 0, "invalid size");

	using Matrix        = std::array<std::array<T,NC>, NR>;
	using size_type     = std::size_t;
	using value_type    = Matrix;
	using reference     = Matrix&;
	using pointer       = Matrix*;
	using const_pointer = const Matrix*;

	static size_type size(const Matrix& matrix)     { return 1; }
	static size_type capacity(const Matrix& matrix) { return 1; }

	// Return pointers to the start of the first *inner* array
	static T*       data(Matrix&       matrix) { return matrix[0].data(); }
	static const T* data(const Matrix& matrix) { return matrix[0].data(); }

	static void resize(Matrix& matrix, const size_type new_size) {
		FANCY_ASSERT( new_size == 1, "invalid size" );
	}

	static void resizeToCapacity(Matrix& matrix) {
		return;
	}
};


// Complex
template<typename T>
struct MpiDataTraits<std::complex<T>>
{
	static_assert(std::is_arithmetic<T>::value, "invalid type");

	using Complex       = std::complex<T>;
	using size_type     = std::size_t;
	using value_type    = Complex;
	using reference     = Complex&;
	using pointer       = Complex*;
	using const_pointer = const Complex*;

	static size_type size(const Complex& value)     { return 1; }
	static size_type capacity(const Complex& value) { return 1; }

	static Complex*       data(Complex& value)       { return &value; }
	static const Complex* data(const Complex& value) { return &value; }

	static void resize(Complex& value, const size_type new_size) {
		FANCY_ASSERT( new_size == 1, "invalid size" );
	}

	static void resizeToCapacity(Complex& value) {
		return;
	}
};


#endif // ifndef MPI_DATA_TRAITS_H
