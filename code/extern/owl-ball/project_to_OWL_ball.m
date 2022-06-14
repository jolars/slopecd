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

function [ x ] = project_to_OWL_ball( z, w, epsilon, sorted_and_positive )
%project_to_OWL_ball. Compute the projection onto the OWL ball
%   Input:
%   z: Vector to project
%   w: Sorted vector of weights. 
%   epsilon: Radius of OWL ball
%   sorted: String (true/false) indicated whether z is sorted in nonincreasing order and
%   positive.
% 
%   Output:
%   x: projection of z onto OWL ball.

x = project_to_OWL_ball_MEX(z, w, epsilon, sorted_and_positive);

end

