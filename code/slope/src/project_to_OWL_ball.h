//
//  project_to_OWL_ball.h
//
//
// Copyright 2015, Damek Davis
//
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

#ifndef ____project_to_OWL_ball__
#define ____project_to_OWL_ball__

#include <algorithm>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index_container.hpp>
#include <cmath>
#include <iterator>
#include <list>
#include <string>
#include <vector>

using boost::multi_index_container;
using namespace boost::multi_index;

// Group structure
struct group
{
  //    Ratio comparint between group and following group.
  double r;
  //    The sum of the z and w elements from this group. Holds onto these
  //    instead of the average.
  double z_sum;
  double w_sum;

  //    Upper and lower bounds on the group elements.
  size_t min_idx;
  size_t max_idx;

  group(double r_,
        double z_sum_,
        double w_sum_,
        size_t min_idx_,
        size_t max_idx_)
    : r(r_)
    , z_sum(z_sum_)
    , w_sum(w_sum_)
    , min_idx(min_idx_)
    , max_idx(max_idx_)
  {
  }

  //    Function returning the number of elements in the group.
  double nb_elements() const { return max_idx - min_idx + 1; }
};

/* Container for groups ordered with repect to
 (1) the minimum element in the group and (2)
 the ratio between the current group and the next group.
 */
typedef multi_index_container<
  group,
  indexed_by<ordered_unique<BOOST_MULTI_INDEX_MEMBER(group, size_t, min_idx)>,
             ordered_non_unique<BOOST_MULTI_INDEX_MEMBER(group, double, r)>>>
  group_container;

// Compare struct for sorting one vector with repect to the ordering of another.
struct comparator
{
  comparator(std::vector<double>& v)
    : lookup(v)
  {
  }
  bool operator()(unsigned int a, unsigned int b)
  {
    return std::abs(lookup[a]) > std::abs(lookup[b]);
  }
  std::vector<double>& lookup;
};

// Helper struct to modify group_containers in place.
struct change_group_ratio
{
  change_group_ratio(const double r_)
    : r(r_)
  {
  }

  void operator()(group& g) { g.r = r; }

private:
  double r;
};

// Main function
void
evaluateProx(std::vector<double>& z_in,
             std::vector<double>& w,
             double epsilon,
             std::vector<double>& x_out,
             bool sorted_and_positive);

// Tool for resorting vectors.
void
preprocess_vector(std::vector<double>& z_in,
                  std::vector<double>& z,
                  std::vector<int>& sign_vec,
                  std::vector<size_t>& order);

// Put all the elements into the group_container.
void
initialize_group_container(group_container& gc,
                           std::vector<double>& z,
                           std::vector<double>& w);

// Projection onto a simplex.
void
project_to_simplex(std::vector<double>& x, group_container& gc, double epsilon);

// Tool for outputing the solution.
void
set_x_val(std::vector<double>& x, group_container& gc, double lambda);

// Helper function to combine the last groups in the data structure.
void
combine_end_groups(group_container& gc,
                   group_container::iterator& it_by_idx,
                   double& zw,
                   double& w2,
                   size_t n);

// Main function that performs group averaging.
void
update_group_container(group_container& gc,
                       double& zw,
                       double& w2,
                       double lambda);

#endif /* defined(____project_to_OWL_ball__) */
