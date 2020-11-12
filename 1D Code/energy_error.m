function er = energy_error(u_ex,un,u0,T,dt,ys)

[x,w] = gauss(10); w = w/2; x = (x+1)/2; w = w(:)';

un_tmp = [0; un; 0];
u0_tmp = [0; u0; 0];

er = 0;
for j = 1:length(ys)-1
    h = ys(j+1)-ys(j);
    tmp = u_ex(h*x+ys(j),T)-(un_tmp(j+1)-un_tmp(j))*x-un_tmp(j);
    tmp = (tmp-(u_ex(h*x+ys(j),T-dt)-(u0_tmp(j+1)-u0_tmp(j))*x-u0_tmp(j)))/dt;
    er = er+h*w*tmp.^2;
end

er = sqrt(er);