/*
	Ideas

	Use MPI_Datatype 
	MPI_Type_contiguous
	MPI_Type_commit
	MPI_Type_free

	Use a GenericFactory to statically register the mapping from C++ types
	to
*/

/*
	Sources
		https://stackoverflow.com/questions/9859390/use-data-type-class-type-as-key-in-a-map
		https://stackoverflow.com/questions/8682582/what-is-type-infobefore-useful-for

*/


struct compare {
	bool operator ()(const type_info* a, const type_info* b) const {
		return a->before(*b);
	}
};

std::map<const type_info*, std::string, compare> m;

void f() {
    m[&typeid(int)] = "Hello world";
}


// TODO switch to type_index 
// - wraps type_info* and provides comparison operators
// https://stackoverflow.com/questions/9859390/use-data-type-class-type-as-key-in-a-map

#include <typeindex>
#include <typeinfo>
#include <unordered_map>

// TODO Use 'map' instead?
typedef std::unordered_map<std::type_index, int> tmap;

int main()
{
    tmap m;
    m[std::type_index(typeid(main))] = 12;
    m[std::type_index(typeid(tmap))] = 15;
}
