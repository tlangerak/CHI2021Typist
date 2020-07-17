function f = finger_model(z, model)
    % z = [x,y,z,dx,dy,dz,fx,fy,fz | dfx,dfy,dfz]
    index=model.index;
    
    x = [
        z(index.X) 
        z(index.Y) 
        z(index.Z) 
        z(index.dX) 
        z(index.dY) 
        z(index.dZ) 
        ];
    u = [
        z(index.fX)
        z(index.fY)
        z(index.fZ)
        ];
        
    ft= model.A*x +  model.B*u;
    f=ft;
end
