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

n_set = [1000, 10000, 100000,1000000];
density_set = [1, .5, .25, .10];
n_runs = 100;
lambda_1 = 1e-3;
lambda_2 = 1e-5;
vals = zeros(length(density_set), length(n_set));
for ii = 1:length(n_set)
   n = n_set(ii);
   w = lambda_1 + lambda_2*(n - (1:n)');
   for jj = 1:length(density_set);
       density = density_set(jj);
       disp(['Starting: Density number ', num2str(jj), ' out of ' num2str(length(density_set)),...
           ' and n_set number ', num2str(ii), ' out of ', num2str(length(n_set))]);
       for kk = 1:n_runs
           z = full(sprandn(n, 1, density));
           tic 
           x = project_to_OWL_ball_MEX(z, w, 1, 'false');
           vals(jj, ii) = vals(jj, ii) + toc/n_runs;
       end
   end
end

save('OWLtimings.mat', 'vals', 'n_set', 'lambda_1', 'lambda_2', 'n_runs');