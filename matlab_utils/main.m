clear; clc; close all;

indx;
% 
%% system
dt = 1e-3;
dt2 = dt^2;
dt3 = dt^3;
m = 0.05;
finger_length = 7.5e-2;
A =[
    1., 0., 0., dt, 0., 0., dt2 / m, 0., 0.;
    0., 1., 0., 0., dt, 0., 0., dt2 / m, 0.;
    0., 0., 1., 0., 0., dt, 0., 0., dt2 / m;
    0., 0., 0., 1., 0., 0., dt / m, 0., 0.;
    0., 0., 0., 0., 1., 0., 0., dt / m, 0.;
    0., 0., 0., 0., 0., 1., 0., 0., dt / m;
    0., 0., 0., 0., 0., 0., 1., 0., 0.;
    0., 0., 0., 0., 0., 0., 0., 1., 0.;
    0., 0., 0., 0., 0., 0., 0., 0., 1.;
];
B = [
    dt3 / m, 0, 0;
    0, dt3 / m, 0;
    0, 0, dt3 / m;
    dt2 / m, 0, 0;
    0, dt2 / m, 0;
    0, 0, dt2 / m;
    dt, 0, 0;
    0, dt, 0;
    0, 0, dt;
];
[nx,nu] = size(B);

umin = [-50, -50, -50];     
umax = [50, 50, 50];
xmin = [-0.05, -0.05, -0.05, -1, -1, -1, -1, -1, -1]; 
xmax = [0.05, 0.05, 0.05, 1, 1, 1, 1, 1, 1];

%% FORCES multistage form
% assume variable ordering zi = [ui; xi] for i=1...N

% dimensions
model.nStates = length(index.states);
model.nInputs = length(index.inputs);
model.N     = 100;   % horizon length
model.nh = 1;
model.nvar  = model.nStates + model.nInputs;    % number of variables
model.neq   = model.nStates;    % number of equality constraints
model.dt = dt;
model.index = index; 
model.A = A;
model.B = B;
model.npar = index.parameters;

% objective 
model.objective = @(z, param) cost_function(z, param, model);
% model.objectiveN = @(z,params) cost_function(z, params, model);

% equalities
model.eq = @(z) finger_model(z, model);             
model.E =  [zeros(model.nStates, model.nInputs), eye(model.nStates)];

% initial state
model.xinitidx = model.index.init; % Index of state

% inequalities
model.ineq = @(z) eval_const(z, model);
model.lb = [ umin,    xmin  ];
model.ub = [ umax,    xmax  ];
model.hl = [-finger_length^2];
model.hu = [finger_length^2];

%% Generate FORCES solver
codeoptions = getOptions('FORCESNLPsolver');
codeoptions.maxit = 200;    % Maximum number of iterations
codeoptions.printlevel = 2; % Use printlevel = 2 to print progress (but not for timings)
codeoptions.optlevel = 2;   % 0: no optimization, 1: optimize for size, 2: optimize for speed, 3: optimize for size & speed
codeoptions.cleanup = 1;

% generate code
FORCES_NLP(model, codeoptions);

% %% simulate
x1 = [0.02;0.00;0.03;0.00;0.00;0.00;0.00;0.00;0.00];
kmax = 3000;
X = zeros(9,kmax+1); X(:,1) = x1;
U = zeros(3,kmax);
problem.x0 = zeros(model.N*model.nvar,1);
params = zeros(index.parameters, model.N);
params(index.XTarget, :) = 1e-2 ;
params(index.YTarget, :) = 1e-2 ;
params(index.ZTarget, :) = 1e-2 ;
params(index.position_weight, :) = 10;
params(index.input_weight, :)=0.0001;
params = reshape(params, [model.N*index.parameters,1]);
for k = 1:kmax
    problem.xinit = X(:,k);
    problem.all_parameters = params; 
    [solverout,exitflag,info] = FORCESNLPsolver(problem);
    if( exitflag == 1 )
        U(:,k) = solverout.x001(1:3);
        solvetime(k) = info.solvetime;
        iters(k) = info.it;
    else
        error('Some problem in solver');
    end
    X(:,k+1) = model.eq( [U(:,k); X(:,k)] )';
end
% 
% %% plot
figure; clf;
subplot(2,1,1); grid on; title('states'); hold on;
stairs(1:kmax,X(:,1:kmax)');
subplot(2,1,2);  
grid on; 
title('input'); 
hold on;
stairs(1:kmax,U(:,1:kmax)');
