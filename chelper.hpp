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
  for (int i = 0; i < num; ++i) {
    ls[i] = start + step * i;
  }
  return ls;
}

template<typename T>
std::vector<T> partition(T& range, int num)
{
  int partSize = range.size() / num;
  std::vector<T> partitions;
  for (int i = 0; i < num; i++) {
    if (i==num-1)
      partitions.emplace_back(range.begin()+(i*partSize), range.end());
    else
      partitions.emplace_back(range.begin()+(i*partSize), range.begin()+((i+1)*partSize));
    
  }
  return partitions;
    
}

#include <cmath>

double basisfunction(double radius)
{
  const double cutoff = 0.4;
  const double shape  = 8;

  if (radius > cutoff)
    return 0;
  else
    return std::exp( - std::pow(shape*radius, 2));
}

double testfunction(double a) { return std::pow(a, 5) - std::pow(a, 4) + std::pow(a, 3) - std::pow(a, 2) + 1; }

