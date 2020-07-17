function f = cost_function(z, param, model)
    index=model.index;  
    w_pos_xy = param(index.position_weight_xy);
    w_input_xy = param(index.input_weight_xy);
    w_pos_z = param(index.position_weight_z);
    w_input_z = param(index.input_weight_z);
    
    x = [z(index.X); z(index.Y)];
    x_target = [param(index.XTarget); param(index.YTarget)];
    error_pos_z = (z(index.Z)-param(index.ZTarget))^2;
    
    u_xy = [z(index.fX); z(index.fY)];
    error_input_xy = u_xy' * u_xy;
    error_input_z = z(index.fZ)^2;
%     
    radius_x = param(index.x_radius);
    radius_y = param(index.y_radius);
    d_radius = sqrt((x(1)-x_target(1))^2 / radius_x^2 + (x(2)-x_target(2))^2 / radius_y^2) - 1;
    error_pos_xy = if_else((d_radius) >= 0, d_radius^2, 0);
    
%     d_y = (x(2) - x_target(2))^2-radius_y^2;
%     error_pos_xy = max(0,d_y);
%     error_pos_xy = 0;
    f = w_pos_xy * error_pos_xy +w_pos_z*error_pos_z +w_input_xy * error_input_xy + w_input_z*error_input_z;   
end
