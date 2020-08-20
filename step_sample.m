function f = step_sample(desired_force, dt)
    desired_impulse = desired_force * dt;
    max_motor_units = 10;
    average_activation = 0.1;
    average_impulse = average_activation * dt;
%     a = desired_force / (max_motor_units * average_activation);
%     mu = max_motor_units*a;
%     sigma = sqrt(max_motor_units*abs(a)*abs(1-abs(a)));

    a = desired_impulse / (max_motor_units * average_impulse);
    mu = max_motor_units*a;
    sigma = sqrt(max_motor_units*abs(a)*abs(1-abs(a)));

    f = normrnd(mu, sigma)*average_impulse/dt;
end
