#include "Foo.h"

Foo::Foo():
	i_(0)
{}


Foo::Foo(const int i):
	i_(i)
{}


int Foo::get_i() const 
{ 
	return i_; 
}
