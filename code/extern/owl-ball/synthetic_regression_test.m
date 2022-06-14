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

clear all; close all; clc;
% d_set = [5, 10];
d_set = [1];
infosetDRS = [];
optssetDRS = [];
infosetFBSLS = [];
optssetFBSLS = [];
infosetFBS = [];
optssetFBS = [];
infosetFISTA = [];
optssetFISTA = [];
data = [];
data_elements = [];
MAX_ITER = 10000;
for outer = 1:length(d_set)
    disp(['Outer loop: ', num2str(outer)]);
    %% OSCAR test
    % Generate data

    d = d_set(outer);
    x_true = [zeros(150*d, 1); 3*ones(50*d, 1); zeros(250*d, 1);...
        -4*ones(50*d, 1); zeros(250*d, 1); 6*ones(50*d, 1); zeros(200*d, 1)];
    n = 1000*d;

    mu = zeros(1,n);
    sigma = eye(1000*d);
    for i=1:n
        for j=1:n
            sigma(i,j) = .8^(abs(i-j));
        end;  
    end;

    A = mvnrnd(mu,sigma,n);
    for i=1:n
      A(:,i) = (A(:,i) - mean(A(:,i)))/std(A(:,i));
    end;

    b = normrnd(A*x_true,sqrt(.01));
    b = (b-mean(b))/std(b);

    lambda_1 = 1e-3;
    lambda_2 = 1e-5;
    w = lambda_1 + lambda_2*(n - (1:n)');
    data.A = A;
    data.b = b;
    data.w = w;
    data_elements = [data_elements, data];

%     Just one epsilon for now.
    eps_set = [sort(abs(x_true), 'descend')'*w];
    for eps_idx = 1:length(eps_set)
        epsilon = eps_set(eps_idx);
    %% DRS
    gamma_set = [1];
%     gamma_set = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    for j = 1:length(gamma_set)
        disp(['DRS Test: ', num2str(j)]);
        [sol,info, opts] = OWL_constrained_regression(A,b,w,epsilon,...
                         'algorithm', 4,...
                         'printfrequency', 100, ...
                         'maxitr', MAX_ITER, ...
                          'stoppingtolerance', 0, ...
                         'opsplittingstepsize', gamma_set(j),... 
                          'truesolution', x_true, ...
                         'saveprogress', 1);
        info.d = d;
        info.epsilon = epsilon;
        infosetDRS = [infosetDRS, info];
        optssetDRS = [optssetDRS, opts];
    end

    %% FBS        
    disp('FBS Test.');
    [sol,info, opts] = OWL_constrained_regression(A,b,w,epsilon,...
                     'algorithm', 1,...        
                     'printfrequency', 100, ...
                     'maxitr', MAX_ITER, ...
                     'stoppingtolerance', 0,...
                     'truesolution', x_true,...
                     'saveprogress', 1);
    info.d = d;
    info.epsilon = epsilon;
    infosetFBS = [infosetFBS, info];
    optssetFBS = [optssetFBS, opts];

    %% FBS with line search 
     disp(['FBS Linesearch Test: ', num2str(j)]);
    [sol,info, opts] = OWL_constrained_regression(A,b,w,epsilon,...
                     'algorithm', 3,...
                     'printfrequency', 100, ...
                     'maxitr', MAX_ITER, ...
                     'stoppingtolerance', 0,...
                     'truesolution', x_true,...
                     'saveprogress', 1);
    info.d = d;
    info.epsilon = epsilon;
%         info.A = A;
    infosetFBSLS = [infosetFBSLS, info];
    optssetFBSLS = [optssetFBSLS, opts];
%     
        %% FISTA        
    disp('FISTA Test.');
    [sol,info, opts] = OWL_constrained_regression(A,b,w,epsilon,...
                     'algorithm', 2,...        
                     'printfrequency', 100, ...
                     'maxitr', MAX_ITER, ...
                     'objstoppingtolerance', 0,...
                     'truesolution', x_true,...
                     'saveprogress', 1, ...
                     'acceleration', 1);
    info.d = d;
    info.epsilon = epsilon;
    infosetFISTA = [infosetFISTA, info];
    optssetFISTA = [optssetFISTA, opts];

    end
end
save('synthetic_example_test', 'infosetDRS', 'infosetFBS', 'infosetFBSLS', 'infosetFISTA', 'data_elements', 'd_set');