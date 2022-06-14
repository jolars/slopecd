//
//  project_to_OWL_ball.cpp
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

#include "project_to_OWL_ball.h"

void
evaluateProx(std::vector<double>& z_in,
             std::vector<double>& w,
             double epsilon,
             std::vector<double>& x_out,
             bool sorted_and_positive)
{

  //    Initialize an sort vector if necessary.
  std::vector<double> z(z_in.size());
  std::vector<size_t> order(z_in.size());
  std::vector<int> sign_vec(z_in.size());

  if (sorted_and_positive) {
    //        This copies the entire vector :(.
    z = z_in;
  } else {
    preprocess_vector(z_in, z, sign_vec, order);
  }
  std::vector<double> x(z.size());

  //    Create the group container data structure.
  group_container gc;
  initialize_group_container(gc, z, w);

  /*
   zw = <z, w>.
   w2 = ||w||^2.
   zw_0 = <z_0, w> where (z_0)_i = z_i if z_i is not z_n and is 0 otherwise.
   w2_0 = ||w_0||^2 where (w_0)_i = w_i if z_i is not z_n and is 0 otherwise.
   lambda = (zw - epsilon)/w2.
   lambda_0 = (zw_0 - epsilon)/w2_0.
   */
  double zw_0 = 0;
  double zw = 0;
  double w2 = 0;
  double w2_0 = 0;
  double lambda_0;
  double lambda_1;

  for (group_container::iterator itr = gc.begin(); itr != gc.end(); itr++) {
    double nb_elements = (*itr).nb_elements();
    zw += (*itr).z_sum * (*itr).w_sum / (nb_elements);
    w2 += (*itr).w_sum * (*itr).w_sum / (nb_elements);
  }

  //    Now the main loop; will always terminate in at most n steps.
  size_t count = 0;
  while (count < z.size()) {
    //        End the program if z is already in the the ball.
    if (zw <= epsilon) {
      x = z;
      break;
    }
    //        Simple formula for obtaining zw_0 and w2_0 from zw and w2_0
    double nb_elements = (*gc.rbegin()).nb_elements();
    zw_0 = zw - ((*gc.rbegin()).z_sum * (*gc.rbegin()).w_sum) / nb_elements;
    w2_0 = w2 - ((*gc.rbegin()).w_sum * (*gc.rbegin()).w_sum) / nb_elements;

    lambda_0 = (zw_0 - epsilon) / w2_0;
    lambda_1 = (zw - epsilon) / w2;

    //        Now we get the minimum ratio.
    double r = (*gc.get<1>().begin()).r;
    //        If the ratio is infinite, we project onto the simplex.
    if (r == std::numeric_limits<double>::infinity()) {
      project_to_simplex(x, gc, epsilon);
      break;
    }

    //        Now we get the value of the variables on the last group.
    group_container::iterator itr_by_idx = gc.end();
    itr_by_idx--;
    double z_last = (*itr_by_idx).z_sum;
    double w_last = (*itr_by_idx).w_sum;
    size_t last_group_start = (*itr_by_idx).min_idx;

    //        Begin the 6 part test.
    if (lambda_1 > r) {
      //            Do averaging. We know lambda^* >= lambda_0 > r
      update_group_container(gc, zw, w2, lambda_1);
    } else {
      if (z_last - lambda_1 * w_last >= 0) {
        //                Finish
        set_x_val(x, gc, lambda_1);
        break;
      } else {
        if (lambda_0 > r) {
          //                    Do averaging. We know lambda^* >= lambda_0 > r
          update_group_container(gc, zw, w2, lambda_0);
        } else {
          //                    Now we get the value of the variables on the
          //                    second to last group.
          itr_by_idx--;
          double z_second_last = (*itr_by_idx).z_sum;
          double w_second_last = (*itr_by_idx).w_sum;
          /*
                          We keep iterating until we find k with z_k - lambda_0
             w_k >= 0. All the k that we find will have optimal value 0. Thus,
             we cannot have every element of the vector satisfy z_k - lambda_0
             w_k < 0, or we will not satisfy <w,x^*> = epsilon.
           */
          while ((itr_by_idx != gc.begin()) &&
                 (z_second_last - lambda_0 * w_second_last < 0)) {
            itr_by_idx--;
            z_second_last = (*itr_by_idx).z_sum;
            w_second_last = (*itr_by_idx).w_sum;
          }
          itr_by_idx++;
          z_second_last = (*itr_by_idx).z_sum;
          w_second_last = (*itr_by_idx).w_sum;
          size_t second_last_group_start = (*itr_by_idx).min_idx;

          if (last_group_start == second_last_group_start) {
            //                        If the last group is the only that
            //                        satisfies z_k - lambda_0 w_k < 0, we can
            //                        stop.
            set_x_val(x, gc, lambda_0);
            break;
          } else {
            //                        Otherwise, we need to merge all the end
            //                        groups.
            combine_end_groups(gc, itr_by_idx, zw, w2, z.size() - 1);
          }
        }
      }
    }
    count++;
  }

  if (sorted_and_positive) {
    //        This copies the entire vector :(.
    x_out = x;
  } else {
    //        If the vector was not sorted, we need to invert the permutation in
    //        order and recover x^*
    x_out.reserve(x.size());
    std::vector<size_t> inverse(z.size());
    for (size_t it = 0; it < x.size(); it++) {
      inverse[order[it]] = it;
    }
    for (size_t it = 0; it < x.size(); it++) {
      x_out[it] = sign_vec[it] * x[inverse[it]];
    }
  }
}

void
preprocess_vector(std::vector<double>& z_in,
                  std::vector<double>& z,
                  std::vector<int>& sign_vec,
                  std::vector<size_t>& order)
{
  //    We just sort z_in in order of descending absolute value and maintain the
  //    sorting permutation and the absolute value.
  for (size_t it = 0; it < z_in.size(); it++) {
    order[it] = it;
    sign_vec[it] = (z_in[it] >= 0) ? 1 : -1;
  }

  comparator cmp(z_in);

  std::sort(order.begin(), order.end(), cmp);

  for (size_t it = 0; it < z_in.size(); it++) {
    z[it] = sign_vec[order[it]] * z_in[order[it]];
  }
}

void
initialize_group_container(group_container& gc,
                           std::vector<double>& z,
                           std::vector<double>& w)
{

  double z_next = 0;
  double w_next = 0;
  double nb_el = 0;
  double z_sum = 0;
  double w_sum = 0;
  size_t max_idx = 0;
  size_t min_idx = 0;
  double r = 0;
  //    Start at the last group and insert it into the group container.
  for (int itr = (int)z.size() - 1; itr >= 0; itr--) {
    //        Initialize the sum over the entire group.
    z_sum = z[itr];
    w_sum = w[itr];
    //        Last element of group.
    max_idx = itr;
    //        Keep adding elements into group while they are equal to z_{itr}
    while ((itr >= 1) && (z[itr] == z[itr - 1])) {
      itr--;
      z_sum += z[itr];
      w_sum += w[itr];
    }
    //        Last element of group.
    min_idx = itr;
    nb_el = max_idx - min_idx + 1;

    if (max_idx == z.size() - 1) {
      //            If it's the last group, give it value infinity for ratio.
      r = std::numeric_limits<double>::infinity();
    } else {
      //            Do standard ratio.
      r = (z_sum / nb_el - z_next) / (w_sum / nb_el - w_next);
    }
    //        Keep the sum over the group to the right in order to compute r.
    z_next = z_sum / nb_el;
    w_next = w_sum / nb_el;
    //        Insert the group into the group container.
    gc.insert(gc.begin(), group(r, z_sum, w_sum, min_idx, max_idx));
  }
}

void
project_to_simplex(std::vector<double>& x, group_container& gc, double epsilon)
{
  //    Standard algorithm to project a sorted vector onto the simplex.
  double k = 0;
  double lambda = 0;
  double z_sum = 0;
  double w_0 = (*gc.begin()).w_sum / ((*gc.begin()).nb_elements());
  double scaled_eps = epsilon / w_0;

  /*
      Recall that if two components are equal in z (i.e. in the same group)
      then they are equal in the solution.  We use this to make the projection
     faster.
   */
  for (group_container::iterator it_by_idx = gc.begin(); it_by_idx != gc.end();
       it_by_idx++) {
    k += (*it_by_idx).nb_elements();
    double z_val = (*it_by_idx).z_sum;
    lambda = (z_sum + z_val - scaled_eps) / k;
    if (lambda < z_val / (*it_by_idx).nb_elements()) {
      z_sum += z_val;
    } else {
      k -= (*it_by_idx).nb_elements();
      break;
    }
  }
  lambda = (z_sum - scaled_eps) / k;

  for (group_container::iterator it_seq = gc.begin(); it_seq != gc.end();
       it_seq++) {
    double x_val =
      std::max((*it_seq).z_sum / (*it_seq).nb_elements() - lambda, (double)0);
    for (size_t it_idx = (*it_seq).min_idx; it_idx <= (*it_seq).max_idx;
         it_idx++) {
      x[it_idx] = x_val;
    }
  }
}

void
set_x_val(std::vector<double>& x, group_container& gc, double lambda)
{

  for (group_container::iterator it_by_idx = gc.begin(); it_by_idx != gc.end();
       it_by_idx++) {
    double x_val =
      std::max((*it_by_idx).z_sum - lambda * (*it_by_idx).w_sum, (double)0) /
      ((*it_by_idx).nb_elements());
    for (size_t it_idx = (*it_by_idx).min_idx; it_idx <= (*it_by_idx).max_idx;
         it_idx++) {
      x[it_idx] = x_val;
    }
  }
}

void
combine_end_groups(group_container& gc,
                   group_container::iterator& it_by_idx,
                   double& zw,
                   double& w2,
                   size_t n)
{

  double z_sum_add_to_G = (*it_by_idx).z_sum;
  double w_sum_add_to_G = (*it_by_idx).w_sum;
  zw -= (*it_by_idx).z_sum * (*it_by_idx).w_sum / (*it_by_idx).nb_elements();
  w2 -= (*it_by_idx).w_sum * (*it_by_idx).w_sum / (*it_by_idx).nb_elements();
  it_by_idx++;

  //    Loop to merge the final components of the group_container.
  while (it_by_idx != gc.end()) {

    double z_sum = (*it_by_idx).z_sum;
    double w_sum = (*it_by_idx).w_sum;
    double nb_elements = (*it_by_idx).nb_elements();
    zw -= z_sum * w_sum / nb_elements;
    w2 -= w_sum * w_sum / nb_elements;
    z_sum_add_to_G += z_sum;
    w_sum_add_to_G += w_sum;
    //        Erases the element and increases the pointer by 1.
    it_by_idx = gc.erase(it_by_idx);
  }

  it_by_idx--;
  group g_start = (*it_by_idx);
  //    This is the size of the vector.
  g_start.max_idx = n;
  double nb_e1 = g_start.nb_elements();
  g_start.z_sum += z_sum_add_to_G;
  g_start.w_sum += w_sum_add_to_G;
  zw += g_start.z_sum * g_start.w_sum / nb_e1;
  w2 += g_start.w_sum * g_start.w_sum / nb_e1;
  gc.replace(it_by_idx, g_start);

  //    Update the ratio of the group before g_start.
  if (it_by_idx != gc.begin()) {
    it_by_idx--;
    group gL = (*it_by_idx);
    double nb_eL = gL.nb_elements();
    gL.r = ((gL.z_sum / nb_eL) - (g_start.z_sum / nb_e1)) /
           ((gL.w_sum / nb_eL) - (g_start.w_sum / nb_e1));
    gc.replace(it_by_idx, gL);
  }
}

void
update_group_container(group_container& gc,
                       double& zw,
                       double& w2,
                       double lambda)
{
  //    Data structure to combine groups.
  std::list<group> group_stack;
  double z_sum = 0;
  double w_sum = 0;
  double nb_elements = 0;

  //    While the minimum group ratio is less than the input lambda...
  while ((*gc.get<1>().begin()).r <= lambda) {
    //        Get a view to move between adjacent groups.
    group_container::iterator it_by_idx = gc.project<0>(gc.get<1>().begin());
    z_sum = (*it_by_idx).z_sum;
    w_sum = (*it_by_idx).w_sum;
    nb_elements = (*it_by_idx).nb_elements();
    //        We remove the sum from the current zw and w2 variables.
    zw -= z_sum * w_sum / nb_elements;
    w2 -= w_sum * w_sum / nb_elements;
    group g_start = (*it_by_idx);

    //        Loop forward
    //        Here we delete the group of the found index, and get a pointer to
    //        the next index.
    double r_prev = g_start.r;
    it_by_idx = gc.erase(it_by_idx);

    //        Keep moving to the right until the ratio is bigger than lambda, or
    //        we reach the end of the vector.
    while ((r_prev <= lambda) && (it_by_idx != gc.end())) {
      //            Update the right endpoint of group
      g_start.max_idx = (*it_by_idx).max_idx;
      //            Update the partial sum variables
      z_sum = (*it_by_idx).z_sum;
      w_sum = (*it_by_idx).w_sum;
      nb_elements = (*it_by_idx).nb_elements();
      zw -= z_sum * w_sum / nb_elements;
      w2 -= w_sum * w_sum / nb_elements;
      g_start.z_sum += z_sum;
      g_start.w_sum += w_sum;
      //            Remove group from container and move to the right one group
      //            (moving done automatically)
      r_prev = (*it_by_idx).r;
      it_by_idx = gc.erase(it_by_idx);
    }

    if (it_by_idx != gc.begin()) {
      //            Decrement pointer to point to left of original group.
      it_by_idx--;
      //            Keep moving to the right until the ratio is bigger than
      //            lambda, or we reach the end of the vector.
      while ((*it_by_idx).r <= lambda) {

        //                Update the left endpoint of group
        g_start.min_idx = (*it_by_idx).min_idx;
        //                Update the partial sum variables
        z_sum = (*it_by_idx).z_sum;
        w_sum = (*it_by_idx).w_sum;
        nb_elements = (*it_by_idx).nb_elements();
        zw -= z_sum * w_sum / nb_elements;
        w2 -= w_sum * w_sum / nb_elements;
        g_start.z_sum += z_sum;
        g_start.w_sum += w_sum;
        //                Remove group from container and move to the left one
        //                group.
        it_by_idx = gc.erase(it_by_idx);
        if (it_by_idx == gc.begin()) {
          break;
        } else {
          it_by_idx--;
        }
      }
    }

    //        Push the group onto the stack to deal with it later.
    group_stack.push_front(g_start);
    if (gc.empty()) {
      break;
    }
  }

  //    Insert the elements of the group stack back into the group_container.
  //    Note that the values of $r$ are not up to date!!!
  gc.insert(group_stack.begin(), group_stack.end());

  double z_1;
  double w_1;
  double r;
  //    Now we update zw, w2, and the ratios
  for (std::list<group>::iterator it = group_stack.begin();
       it != group_stack.end();
       it++) {
    //        Update zw and w2
    nb_elements = (*it).nb_elements();
    zw += (*it).z_sum * (*it).w_sum / nb_elements;
    w2 += (*it).w_sum * (*it).w_sum / nb_elements;

    //        Find the position where the current element lives in the group
    //        container
    group_container::iterator it_by_idx = gc.find((*it).min_idx);

    z_1 = (*it_by_idx).z_sum / (*it_by_idx).nb_elements();
    w_1 = (*it_by_idx).w_sum / (*it_by_idx).nb_elements();
    it_by_idx++;

    //        compute right ratios. (we ignore last group because it will
    //        already have r = inf.)
    if (it_by_idx != gc.end()) {
      r = (z_1 - (*it_by_idx).z_sum / (*it_by_idx).nb_elements()) /
          (w_1 - (*it_by_idx).w_sum / (*it_by_idx).nb_elements());
    } else {
      r = std::numeric_limits<double>::infinity();
    }
    it_by_idx--;

    //        Modify in place the group ratio.
    gc.modify(it_by_idx, change_group_ratio(r));

    //        If we're not at the beginning we update the ratio to the right of
    //        the current element.
    if (it_by_idx != gc.begin()) {
      it_by_idx--;
      r = ((*it_by_idx).z_sum / (*it_by_idx).nb_elements() - z_1) /
          ((*it_by_idx).w_sum / (*it_by_idx).nb_elements() - w_1);
      gc.modify(it_by_idx, change_group_ratio(r));
    }
  }
}
