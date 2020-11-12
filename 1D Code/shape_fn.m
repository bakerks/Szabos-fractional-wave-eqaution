function [f,fd,L] = shape_fn(x,p)

f = zeros(length(x),p+1);
fd = f;

L = f;

L(:,1) = ones(length(x),1);
L(:,2) = 2*x-1;

%construct Legendre polynomials on [0,1]
for j = 2:p
    L(:,j+1) = ((2*j-1)/j)*(2*x-1).*L(:,j)-((j-1)/j)*L(:,j-1);
end


%construct integrated Legendre polynomials
f(:,1) = 1-x;
f(:,2) = x;
fd(:,1) = -ones(size(x));
fd(:,2) = ones(size(x));

for j = 3:p+1    
    f(:,j) = (1/sqrt(4*j-6))*(L(:,j)-L(:,j-2));
    fd(:,j) = 2*sqrt(j-3/2)*L(:,j-1);
end