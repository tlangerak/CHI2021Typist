function f = cost_function(z, param, model)
    index=model.index;  
    w_pos = param(index.position_weight);
    w_input = param(index.input_weight);
    x = [z(index.X); z(index.Y); z(index.Z)];
    u = [z(index.dfX); z(index.dfY); z(index.dfZ)];
    x_target = [param(index.XTarget); param(index.YTarget); param(index.ZTarget)];
    et = (x-x_target)' * (x-x_target);
    eu = u' * u;
    f = w_pos * et + w_input * eu;
end
