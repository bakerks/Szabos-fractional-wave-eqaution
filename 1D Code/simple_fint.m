function f = simple_fint(t,g,alpha,nq)
%function f = simple_fint(t,g,alpha)

% compute (1/gamma(alpha))*int_0^t (t-s)^{alpha-1}f(s)ds

if (nargin < 4)
    nq = 20;
end
f = zeros(size(t));
[xj,wj] = gaussj(nq,alpha-1,0);
[x,w] = gauss(nq); x = (x+1)/2; w = w/2;
for j = 1:length(t)
    t1 = t(j)*3/4;
    t1d = t(j)-t1;
    f(j) = ((t1d/2)^alpha)*wj'*g((xj+1).*t1d/2+t1);
    if (t(j) > 0)        
        f(j) = f(j)+(t1/2)*w'*(g(x*t1/2).*(t(j)-t1*x/2).^(alpha-1));
        f(j) = f(j)+(t1/2)*w'*(g((x+1)*t1/2).*(t(j)-t1*(x+1)/2).^(alpha-1));
    end
end

f = (1/gamma(alpha))*f;