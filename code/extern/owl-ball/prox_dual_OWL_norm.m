% Copyright 2015, Damek Davis

% This file is part of OWL Ball Toolbox version 1.0.
%
%    The OWL Ball Toolbox is free software: you can redistribute it
%    and/or  modify it under the terms of the GNU General Public License
%    as published by the Free Software Foundation, either version 3 of
%    the License, or (at your option) any later version.
%
%    The SLOPE Toolbox is distributed in the hope that it will
%    be useful, but WITHOUT ANY WARRANTY; without even the implied
%    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%    See the GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with the OWL Ball Toolbox. If not, see
%    <http://www.gnu.org/licenses/>.

function [ x ] = prox_dual_OWL_norm( z, tau, gamma, sorted_and_positive )
%project_to_OWL_ball. Compute the projection onto the OWL ball
%   Input:
%   z: Vector input to prox
%   tau: A strictly positive and decreasing sequence of real numbers 
%   gamma: Prox stepsize
%   sorted: String (true/false) indicated whether z is sorted in nonincreasing order and
%   positive.
% 
%   Output:
%   x: projection of z onto OWL ball.

if gamma <= 0
    error('Stepsize must be positive');
end
if any( tau <= 0) 
    error('Weights must be strictly positive');
end
% if ~isreal(tau)
%    error('weights must be real numbers'); 
% end
w = zeros(length(z), 1);
w(1) = 1/tau(1);
w(2:end) = (1./tau(2:end)) - (1./tau(1:(end-1)));
x = z - gamma*project_to_OWL_ball_MEX((1/gamma)*z, w, 1, sorted_and_positive);

end

