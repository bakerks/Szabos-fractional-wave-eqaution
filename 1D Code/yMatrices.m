function [M,A,iA,jA,vA] = yMatrices(ys,ps,alpha,hs)
% function [M,A] = yMatrices(ys,ps,alpha)
% returns mass and stiffness matrices for the weighted inner product. The input variables denote 
% ys = (0=ys(1)), ys(2), ...,ys(end) = Y and ps(j) the polynomial degree on interval [ys(j),
% ys(j+1)]
% note that to obtain Dirichlet boundary condition at ys(1), remove the
% first row and column of the matrices, whereas to obtain the Dir BC at
% y(end), remove the final row and column.

if (nargin < 4)
    hs = diff(ys);
end

% number of dofs
Nl = length(ps);
l2g = local2global(ps);

%[xj,wj] = gaussj(max(10,2*ps(1)+5),0,alpha); xj = (xj+1)/2; wj = wj*2^(-alpha-1); %gauss-jacobi on interval [0,1]
[xj,wj] = gaussj(max(ps(1)+1,5),0,alpha);xj = (xj+1)/2; wj = wj*2^(-alpha-1); %gauss-jacobi on interval [0,1]

[fxj,fxjd] = shape_fn(xj,max(ps));

%[x,w] = gauss(max(2*max(ps)+1,10)); x = (x+1)/2; w = w/2; % gauss on interval [0,1]
[x,w] = gauss(max(2*max(ps)+1,10)); x = (x+1)/2; w = w/2; % gauss on interval [0,1]
[fx,fxd] = shape_fn(x,max(ps));


% there will be this many nonzeros with repeats
%nz = sum((ps(1:end-1)+1).^2)+ps(end)^2;
nz = sum((ps+1).^2);
iM = zeros(nz,1); jM = iM; vM = iM; 
iA = iM; jA = iM; vA = iM;

ind = 1;
%first element where we do G-J
%hy = ys(2)-ys(1);
hy = hs(1);
for l = 1:ps(1)+1
   for k = 1:l
       i = l2g(1,l); j = l2g(1,k);
              
       fxl = fxj(:,l);
       fxld = fxjd(:,l);                     
       fxk = fxj(:,k);
       fxkd = fxjd(:,k);       
       
       iM(ind) = i; jM(ind) = j; vM(ind) = (hy^(1+alpha))*wj'*(fxl.*fxk); 
       iA(ind) = i; jA(ind) = j; vA(ind) = (hy^(-1+alpha))*wj'*(fxld.*fxkd); ind = ind+1;
       
       if (i ~= j)
            iM(ind) = j; jM(ind) = i; vM(ind) = vM(ind-1); 
            iA(ind) = j; jA(ind) = i; vA(ind) = vA(ind-1); ind = ind+1;
       end
   end
end

%remaining elements

for el = 2:Nl
    %hy = ys(el+1)-ys(el);
    hy = hs(el);
    for l = 1:ps(el)+1
        for k = 1:l                        
            i = l2g(el,l);
            j = l2g(el,k);
                        
            fxl = fx(:,l);
            fxld = fxd(:,l);           
                  
            fxk = fx(:,k);
            fxkd = fxd(:,k);            
            ya = ((1-x)*ys(el)+x*ys(el+1)).^alpha;
            iM(ind) = i; jM(ind) = j; vM(ind) = hy*w'*(ya.*fxl.*fxk); 
            iA(ind) = i; jA(ind) = j; vA(ind) = (1/hy)*w'*(ya.*fxld.*fxkd); ind = ind+1;            
            if (i ~= j)
                iM(ind) = j; jM(ind) = i; vM(ind) = vM(ind-1);
                iA(ind) = j; jA(ind) = i; vA(ind) = vA(ind-1); ind = ind+1;
            end
        end
    end
end

M = sparse(iM,jM,vM);

A = sparse(iA,jA,vA);