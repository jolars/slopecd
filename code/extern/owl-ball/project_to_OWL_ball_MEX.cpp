//  project_to_OWL_ball_MEX.cpp

// Copyright 2015, Damek Davis

// This file is part of OWL Ball Toolbox version 1.0.
//
//    The OWL Ball Toolbox is free software: you can redistribute it
//    and/or  modify it under the terms of the GNU General Public License
//    as published by the Free Software Foundation, either version 3 of
//    the License, or (at your option) any later version.
//
//    The SLOPE Toolbox is distributed in the hope that it will
//    be useful, but WITHOUT ANY WARRANTY; without even the implied
//    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//    See the GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with the OWL Ball Toolbox. If not, see
//    <http://www.gnu.org/licenses/>.

#include "project_to_OWL_ball_MEX.h"
#include "mex.h"
#include "project_to_OWL_ball.h"
#include <string.h>

/* The gateway function */
void
mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{

  /* check for proper number of arguments */
  if (nrhs != 4) {
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs", "Four inputs required.");
  }

  /* make sure the first input argument is type double */
  if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble",
                      "Vector to project must be type double.");
  }

  /* make sure the first and second input argument is not a sparse vector */
  if (mxIsSparse(prhs[0]) || mxIsSparse(prhs[1])) {
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble",
                      "Inputs cannot be sparse vectors");
  }

  /* check that number of columns in first input argument is 1 */
  if (mxGetN(prhs[0]) != 1) {
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector",
                      "Vector to project must be a column vector.");
  }

  /* make sure the second input argument is type double */
  if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble",
                      "Weight vector must be type double.");
  }

  /* check that number of columns in second input argument is 1 */
  if (mxGetN(prhs[1]) != 1) {
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector",
                      "Weight vector be a column vector.");
  }

  /* make sure the third input argument is scalar */
  if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) ||
      mxGetNumberOfElements(prhs[2]) != 1) {
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar",
                      "Input epsilon must be a scalar.");
  }

  char *input_buf, *output_buf;
  size_t buflen;

  if (mxIsChar(prhs[3]) != 1)
    mexErrMsgIdAndTxt("MATLAB:revord:inputNotString",
                      "Input must be a string.");

  /* input must be a row vector */
  if (mxGetM(prhs[3]) != 1)
    mexErrMsgIdAndTxt("MATLAB:revord:inputNotVector",
                      "Input must be a row vector.");

  /* get the length of the input string */
  buflen = (mxGetM(prhs[3]) * mxGetN(prhs[3])) + 1;

  /* copy the string data from prhs[0] into a C string input_ buf.    */
  input_buf = mxArrayToString(prhs[3]);

  if (input_buf == NULL)
    mexErrMsgIdAndTxt("MATLAB:revord:conversionFailed",
                      "Could not convert input to string.");

  bool sorted_and_positive;
  if (strcmp(input_buf, "true") == 0) {
    sorted_and_positive = true;
  } else if (strcmp(input_buf, "false") == 0) {
    sorted_and_positive = false;
  } else {
    mexErrMsgIdAndTxt("MATLAB:NotTrueOrFalse",
                      "Argument must be true or false.");
  }

  /* create a pointer to the real data in the input matrix  */
  /* get dimensions of the input matrix */
  size_t nb_rows;
  nb_rows = mxGetM(prhs[0]);
  std::vector<double> z(mxGetPr(prhs[0]),
                        mxGetPr(prhs[0]) +
                          nb_rows * sizeof(mxGetPr(prhs[0])) / sizeof(double));
  std::vector<double> w(mxGetPr(prhs[1]),
                        mxGetPr(prhs[1]) +
                          nb_rows * sizeof(mxGetPr(prhs[1])) / sizeof(double));
  double epsilon;

  /* get the value of the scalar input  */
  epsilon = mxGetScalar(prhs[2]);
  //    sorted_and_positive = mxGetScalar(prhs[3]);

  /* create the output matrix */
  plhs[0] = mxCreateDoubleMatrix((mwSize)nb_rows, 1, mxREAL);

  /* get a pointer to the real data in the output matrix */
  double* outMatrix; /* output matrix */
  outMatrix = mxGetPr(plhs[0]);

  /* call the computational routine */

  std::vector<double> x(nb_rows, 0);
  evaluateProx(z, w, epsilon, x, sorted_and_positive);
  for (size_t i = 0; i < z.size(); i++) {
    outMatrix[i] = x[i];
  }
}
