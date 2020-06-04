#ifndef FOO_H
#define FOO_H

class Foo
{
 public:
	Foo();
	Foo(const int i);

	int get_i() const;

 private:
	int i_;
};

#endif /* FOO_H */
