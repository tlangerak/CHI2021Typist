clear; clc; close all;

indx;
% 
%% system
dt = 1.1e-2;
dt2 = dt^2;
m = 0.1;
finger_length = 7.5e-2;

A =[
    1., 0., 0., dt, 0., 0.;
    0., 1., 0., 0., dt, 0.;
    0., 0., 1., 0., 0., dt;
    0., 0., 0., 1., 0., 0.;
    0., 0., 0., 0., 1., 0.;
    0., 0., 0., 0., 0., 1.;
];
B = [
    0.5*dt2 / m, 0, 0;
    0, 0.5*dt2 / m, 0;
    0, 0, 0.5*dt2 / m;
    dt / m, 0, 0;
    0, dt / m, 0;
    0, 0, dt / m;
];
[nx,nu] = size(B);
max_acceration = 8;
umin = [-10, -10, -10];     
umax = [10, 10, 10];
xmin = [-0.1, -0.1, -0.1, -10, -10, -10]; 
xmax = [0.1, 0.1, 0.1, 10, 10, 10];

%% FORCES multistage form
% assume variable ordering zi = [ui; xi] for i=1...N

% dimensions
model.nStates = length(index.states);
model.nInputs = length(index.inputs);
model.N = 50;   % horizon length
% model.nh = 1;
model.nvar  = model.nStates + model.nInputs;    % number of variables
model.neq   = model.nStates;    % number of equality constraints
model.dt = dt;
model.index = index; 
model.A = A;
model.B = B;
model.npar = index.parameters;

% objective 
model.objective = @(z, param) cost_function(z, param, model);
model.objectiveN = @(z,params) 10*cost_function(z, params, model);

% equalities
model.eq = @(z) finger_model(z, model);             
model.E =  [zeros(model.nStates, model.nInputs), eye(model.nStates)];

% initial state
model.xinitidx = model.index.init; % Index of state

% inequalities
% model.ineq = @(z) eval_const(z, model);
model.lb = [ umin,    xmin  ];
model.ub = [ umax,    xmax  ];
% model.hl = [-finger_length^2];
% model.hu = [finger_length^2];

%% Generate FORCES solver
codeoptions = getOptions('FORCESNLPsolver');
codeoptions.maxit = 10000;    % Maximum number of iterations
codeoptions.printlevel = 1; % Use printlevel = 2 to print progress (but not for timings)
codeoptions.optlevel = 0;   % 0: no optimization, 1: optimize for size, 2: optimize for speed, 3: optimize for size & speed
codeoptions.cleanup = 1;
codeoptions.BuildSimulinkBlock = 0;
codeoptions.overwrite = 1;

% generate code
FORCES_NLP(model, codeoptions);


%% simulate
figure; clf;
for width = 1:2
    for height = 1:2
        width
        height
        
        wx = 1e-2;
        wy = width*1e-2;
        mx = 0e-2;
        my = 6e-2;
        sy = height*1e-2;
        x1 = [0.0;0.0;sy;0.00;0.00;0.00];
        kmax = 300;
        emax = 15;
        X = zeros(6,kmax+1); X(:,1) = x1;
        U = zeros(3,kmax);
        Xend = zeros(2, emax);
        problem.x0 = zeros(model.N*model.nvar,1);
        params = zeros(index.parameters, model.N);
        params(index.XTarget, :) = mx ;
        params(index.YTarget, :) = my ;
        params(index.ZTarget, :) = 0 ;
        params(index.position_weight_xy, :) = 1;
        params(index.input_weight_xy, :)=1;
        params(index.position_weight_z, :) = 1;
        params(index.input_weight_z, :)=1;
        params(index.y_radius, :) = wy;
        params(index.x_radius, :)= wx;
        params = reshape(params, [model.N*index.parameters,1]);
        
        for e = 1:emax
            for k = 1:kmax
                problem.xinit = X(:,k);
                problem.all_parameters = params; 
                [solverout,exitflag,info] = FORCESNLPsolver(problem);
                if( exitflag == 1 )
                    desired_force = solverout.x01(1:3);
                    U(:,k) =   [step_sample(desired_force(1),dt); 
                                step_sample(desired_force(2),dt); 
                                step_sample(desired_force(3),dt)];
                    solvetime(k) = info.solvetime;
                    iters(k) = info.it;
                else
                    exitflag
                    error('Some problem in solver');
                end
                
                X(:,k+1) = model.eq( [U(:,k); X(:,k)] )';
                if X(3, k+1) < 0
                    Xend(:, e)  = X(1:2, k+1);
                    break
                end
            end
        end
        subplot(5,5, width+(height-1)*5); grid on; hold on; 
        x = [mx-wx, mx-wx, mx+wx, mx+wx, mx-wx];
        y = [my+wy, my-wy, my-wy, my+wy, my+wy];
        plot(x, y, 'b-', 'LineWidth', 0.5);
        hold on;
        scatter(Xend(1,:), Xend(2,:))
        xlim([mx-0.05, mx+0.05]);
        ylim([my-0.05, my+0.05]);
    end
end


%% plot
% figure; clf;
% subplot(3,1,1); grid on; title('position'); hold on;
% stairs((1:k)*dt,X(1:3,1:k)');
% subplot(3,1,2);  
% grid on; 
% title('vel'); 
% hold on;
% stairs((1:k)*dt,X(4:6,1:k)');
% subplot(3,1,3);  
% grid on; 
% title('input'); 
% hold on;
% stairs((1:k)*dt,U(:,1:k)');
