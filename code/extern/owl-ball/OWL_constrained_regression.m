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

function [sol, info, opts ] = OWL_constrained_regression( A, b, w, epsilon, varargin )
% MODEL:
%   minimize (1/2)||Ax - b||^2
%   subject to w_1 |x|_1 + w_2 |x|_2 + ... + |x|_n <= epsilon
%
% where:
% A: the measurement matrix.
% b: the observed vector. 
% w: a vector of weights for the OWL norm.
% |x|_i: the $i$th largest component of x in absolute value.
% epsilon: the radius of the ball.
%
% INPUT:
% A: Measurement matrix
% b: the observed vector. 
% w: a vector of weights for the OWL norm.
% epsilon: the radius of the ball.
% varargin: (optional) options%
%
% OUTPUT
% x:    Final iterate
% info: information structure
%       info.itr:   number of total iterations
%       info.fpr_hist:  history of fixed-point residual
%       info.obj: Objective value history
%       info.obj_erg_std: Standard ergodic objective value
%       info.obj_erg_new: New ergodic obejctive value
% opts: options used
%% ------------------------------------------------------------
% Precompute frequently used quantities  
%--------------------------------------------------------------
AtA = A' * A;
Atb = A' * b;
n = size(A, 2);
%% ------------------------------------------------------------
% Set parameters to their default values  
%--------------------------------------------------------------
opts.gamma = [];     % operator splitting step size
opts.splttol = 1e-6;   % splitting stopping tolerance
opts.truesol = [];  % true solution
opts.maxitr = 10000;    % total iterations
opts.printfrequency = 0;
opts.saveprogress = false;
opts.lambda = 1;%2/(4 - 1.99);
opts.linesearch = 0;
opts.relerrtol = 0;
opts.accl = 0;
opts.objtol = 0;
opts.algorithm = 0;
% Initilization
z = zeros(n, 1);
x_prev = z;
t = 1;

%% Check for required input arguments
if (nargin-length(varargin)) ~= 4
    error('Wrong number of required input arguments');
end

%% ------------------------------------------------------------
% Set parameters to user specified values
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
    error('Options should be given in pairs');
else
    for itr=1:2:(length(varargin)-1)
        switch lower(varargin{itr})
            case 'algorithm'
%                 1 = Forward-Backward splitting (FBS) 
%                 2 = FISTA
%                 3 = Forward-backward splitting (FBS) with line search
%                 from http://www.optimization-online.org/DB_FILE/2015/03/4804.pdf
%                 4 = Douglas-Rachford splitting.
                if ismember(varargin{itr+1}, 1:4)
                   opts.algorithm = varargin{itr+1}; 
                else
                    error('Choose algorithm numbered 1 through 4');
                end
            case 'acceleration'
                opts.accl = varargin{itr+1};
            case 'overrelaxationstepsize'
                opts.lambda = varargin{itr+1};
            case 'relerrtol'
                opts.relerrtol = varargin{itr+1};  
            case 'opsplittingstepsize'
                opts.gamma = varargin{itr+1}; 
            case 'objstoppingtolerance'
                opts.objtol = varargin{itr+1}; 
            case 'stoppingtolerance'
                opts.splttol = varargin{itr+1};
            case 'truesolution'
                opts.truesol = varargin{itr+1};
                norm_truesol = norm(opts.truesol(:));
                opts.objective = objective(A, b, opts.truesol);
            case 'maxitr'
                opts.maxitr = varargin{itr+1};
            case 'initialpoint'
                z = varargin{itr+1}(:);
            case 'printfrequency'
                opts.printfrequency = varargin{itr+1};
            case 'saveprogress'
                opts.saveprogress = varargin{itr+1};
            otherwise
                error(['Unrecognized option: ''' varargin{itr} '''']);
        end;
    end;
end

if opts.saveprogress
    info.fpr_hist = zeros(opts.maxitr,1);
    info.obj = zeros(opts.maxitr,1);
    info.time = zeros(opts.maxitr,1);
    if ~isempty(opts.truesol)
        info.relerr_hist = zeros(opts.maxitr,1);    
        info.relobjerr_hist = zeros(opts.maxitr,1);
    end
else
    info = [];
end

if isempty(opts.gamma)
    switch opts.algorithm
        case 1 
    %         Standard FBS            
                tic;
                beta = 1/norm(AtA);
                info.norm_time = toc;
                opts.gamma = 1.99*beta;
        case 2
    %         Accelerated FBS            
                tic;
                beta = 1/norm(AtA);
                info.norm_time = toc;
                opts.gamma = beta;
        case 3
    %         FBS with line search 
            opts.gamma = 1;
        case 4
    %         DRS 
            opts.gamma = 1;
    end;
end

if opts.algorithm == 4
    IpAtA = eye(n) + opts.gamma* (AtA);
end

tic;
rho = 1;
%% Main iterations
for itr = 1:opts.maxitr
    switch opts.algorithm
        case 1 
%         Standard FBS            
            z_prev = z;
            z = project_to_OWL_ball(z - opts.gamma *(AtA*z - Atb), w, epsilon, 'false');
        case 2
%         Accelerated FBS            
            z_prev = z;
            x = project_to_OWL_ball(z - opts.gamma *(AtA*z - Atb), w, epsilon, 'false');
            t_new = (1 + sqrt(1 + 4*t^2))/2;
            z = x + ((t - 1)/t_new)*(x - x_prev);
            x_prev = x;
            t = t_new;          
        case 3
%         FBS with line search              
            z_prev = z;
            x_B = project_to_OWL_ball_MEX(z, w, epsilon, 'false');
            v = (x_B - z) - opts.gamma *(AtA*z - Atb);
            Av = A*v;
            if Av == 0;
                rho = 1;
            else 
                rho = (1/opts.gamma)*norm(v)^2/norm(Av)^2;
            end
            z = z + rho * v; 
        case 4
%         DRS            
            z_prev = z;
           	x_g = IpAtA\(z+opts.gamma*Atb);
            x_f = project_to_OWL_ball(2*x_g - z, w, epsilon, 'false');
            z = z + opts.lambda*(x_f - x_g);
    end;
    info.time(itr) = toc;
    %% print/save progress
    if (opts.printfrequency && (mod(itr,opts.printfrequency)==0)) || opts.saveprogress
        % compute relative fixed-point residual
        FPR = (1/rho)*norm(z - z_prev)/(1+ norm(z));
        % compute root mean squared error
        obj = objective(A, b, z);
        % compute relative solution error, if true solution is given
        if ~isempty(opts.truesol)
            relerr = norm(z-opts.truesol)/norm_truesol;
            relobjerr = abs(obj - opts.objective)/(1+abs(opts.objective));
        end
        % print progress
        if opts.printfrequency && (mod(itr,opts.printfrequency)==0)
            fprintf('itr=%d\tfpr=%4.2e\tobj=%4.2e', itr, FPR, obj);
            if ~isempty(opts.truesol); fprintf('\trelerr=%4.2e\trelobjerr=%4.2e', relerr, relobjerr); end
            fprintf('\n');
        end
        % save progress
        if opts.saveprogress
            info.fpr_hist(itr) = FPR;
            info.obj(itr) = obj;
            if ~isempty(opts.truesol)
                info.relerr_hist(itr) = relerr;
                info.relobjerr_hist(itr) = relobjerr;
            end
        end
        %% stopping check
        if FPR < opts.splttol && itr > 1
            sol = z;
            break;
        end
        if obj < opts.objtol && itr > 1
            sol = z;
            break;
        end
        if ~isempty(opts.truesol) 
            if relerr < opts.relerrtol && itr > 1
                sol = z;
                break;
            end
        end
    end
end
    sol = z;
    info.sol = sol;

end

function x = objective(A, b, x)
    x = (1/2)*sum((A*x - b).^2);
end
