#include <vector>

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{
  double start = static_cast<double>(start_in);
  double end   = static_cast<double>(end_in);
  double num   = static_cast<double>(num_in);
  double delta = (end - start) / (num - 1);

  std::vector<double> linspaced(num - 1);
  for (int i=0; i < num; ++i)
  {
    linspaced[i] = start + delta * i;
  }
  linspaced.push_back(end);
  return linspaced;
}
