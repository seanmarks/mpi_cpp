#ifndef BAR_H
#define BAR_H

#include <string>
#include <memory>
#include <unordered_map>

class Foo;

class Bar
{
 public:
	Bar();
	Bar(const int i);

	int get_i() const;

 private:
	std::unordered_map<int, std::unique_ptr<Foo>> map_;
};

#include "Foo.h"

#endif /* BAR_H */
