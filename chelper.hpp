#include <vector>

#include <iostream>

using std::cout;
using std::endl;



template<typename T>
std::vector<double> linspace(T start, T stop, int num = 50, bool endpoint = true)
{
  double step = 0;
  
  if (endpoint) {
    step = (stop - start) / (static_cast<double>(num)-1);
  }
  else {
    step = (stop - start) / (static_cast<double>(num));
  }
  std::vector<double> ls(num);
  for (int i = 0; i < num; ++i)
  {
    ls[i] = start + step * i;
  }
  return ls;
}
