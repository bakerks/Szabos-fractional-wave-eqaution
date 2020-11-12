function [f,ys] = rhs(fn,ys,ps,hs)

if (nargin < 4)
    hs = diff(ys);
end

Nx = sum(ps)+1;
l2g = local2global(ps);

f = zeros(Nx,1);
[x,w] = gauss(2*max(ps)+1); x = (x+1)/2; w = w/2;
fx_leg = shape_fn(x,max(ps));

for el = 1:length(ps)
    for j = 1:ps(el)+1
        i = l2g(el,j);
        f(i) = f(i)+hs(el)*w'*(fx_leg(:,j).*fn(ys(el)+hs(el)*x));
    end
end

f = f(2:end-1);