#include "project_to_OWL_ball.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

std::vector<double>
evaluate_prox(std::vector<double> w, std::vector<double> lambdas, double epsilon)
{
  std::vector<double> w_out(w.size());

  evaluateProx(w, lambdas, epsilon, w_out, false);

  return w_out;
}

PYBIND11_MODULE(_slope, m)
{
  m.def("sorted_l1_proj", &evaluate_prox, "Project onto the SL1 norm ball");
}
