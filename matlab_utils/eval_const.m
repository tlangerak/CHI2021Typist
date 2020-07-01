function h = eval_const(z, model)
    index = model.index;
    h = [z(index.X)^2+z(index.Y)^2+z(index.Z)^2];
end
