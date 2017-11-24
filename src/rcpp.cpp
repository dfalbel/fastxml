#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]


// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::export]]
NumericVector timesTwo(NumericVector x) {
  return x * 2;
}



// [[Rcpp::export]]
int useAuto() {
  auto val = 42;		// val will be of type int
  return val;
}