%Solves szabos equation using standard CQ for gamma in (-1,1) with or
%without correction terms using the schemes given in equations 3.9 and 3.8
%respectivly for nonsmooth u

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set parameters
p=2; %(1 bdf1) (2 bdf2) (3 trapez) used within CQ

c = 1; %removes correction if c=0

gam = 0.75; %order of FD in (-1,1)
cgam = ceil(gam);

%controls additional terms required for higher order derivatives
if (gam<0)
    X=0;
elseif (gam>0)
    X=1;
end

a0 = 1; %attenuation coeff >0
a = -a0*(4/pi)*gamma(-gam-1)*gamma(gam+2)*cos((pi*(gam+1))/2); %constant in front of FD

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Set up exact solution

%define z0 - defined in Remark 2.1
if (cgam==0)
    z0 = -a/gamma(3-gam);
elseif (cgam==1)
    z0 = -2*a/gamma(4-gam);
end

%time part:

%Example: (used to make Figure4)
U = @(t) 1+t+t.^2+z0*t.^(2+ cgam-gam);
Ud = @(t) 1+2*t+z0*(2+cgam - gam)*t.^(1+cgam-gam);
Udd = @(t) 2+z0*(2+cgam-gam)*(1+cgam-gam)*t.^(cgam-gam);

%time and space combined:
u_ex = @(x,t) U(t)*sin(pi*x);
u_t_ex = @(x,t) Ud(t)*sin(pi*x);
u_tt_ex = @(x,t) Udd(t)*sin(pi*x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Define the source term f

%exact fractional derivative part:
if (cgam == 0)
    Ufd = @(t) 1/(gamma(1-gam))*t.^(-gam) ;
elseif (cgam == 1)
    Ufd = @(t) 2/(gamma(2-gam))*t.^(1-gam);
end

fdd = @(t) Udd(t);
fx = @(t) pi^2 *U(t);
fcd = @(t) a*Ufd(t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Define initial contitions

U0 = @(x) u_ex(x,0);

V0 = @(x) u_t_ex(x,0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%create objects for finite elements

if ~exist('err_max','var')
    err_max = [];
    err_end = [];
    M = 25/2;
end

M = M*2;
ys = (0:M)/M; ps = ones(1,M); %ys are spatial partitioning

%make mass and stiffness matrices:
[B,A] = yMatrices(ys,ps,0);
B = B(2:end-1,2:end-1); %crop these matrices for dirichlet BCs
A = A(2:end-1,2:end-1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%state time stepping variables

T =1; N = 6*M;  dt =T/N;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%scale initial conditions and f's space part

u0 = B\rhs(U0,ys,ps);
v0 = B\rhs(V0,ys,ps);
f_space = rhs(@(x) sin(pi*x),ys,ps);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Give u1 with the exact solution

U1 = @(x) u_ex(x,dt);
u1 = B\rhs(U1,ys,ps);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Gather convolution weights

weights = convw_gen(N,T,p-1, @(s) s.^(gam));

omega0 = weights(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%precalculate correction terms

    ns = 0:N; %positions
    wn1 = 0*ns';
    
    if (gam<0)
        wn1 = wn1;
    elseif (gam>0)
        for s = 0:N
            for j = 0:s
                wn1(s+1) = wn1(s+1) + weights(s-j+1)*(j);
            end
        end
        wn1 = (dt^(-gam)*ns'.^(1-gam))./gamma(2-gam) - wn1;
    end


    if (gam>0)
        wn0 = - wn1;
    elseif (gam<0)
        wn0 = (dt.*ns)'.^(-gam)./gamma(1-gam);
        for s = 0:N
            for j = 0:s
                wn0(s+1) = wn0(s+1) - weights(s-j+1);
            end
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create storage for vs

v_store = zeros(length(v0),N);
v_store(:,1) = v0;

%make a placeholder for v1
v1 = zeros(length(v0),1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initiate error

err = zeros(N-1,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Time stepping

for j = 2:N 

    %to do the first step differently (since we dont v1 yet)
    if (j==2)
        D=0;
    else 
        D=1;
    end
    
    %creating CQ approximation
    CQ = zeros(size(v0));
    
    for k = 0:j-2
        CQ = CQ+weights(j-1-k+1)*(v_store(:,k+1)- X*v0);
    end
    
    %time stepping scheme
    if (cgam ==0)
        f_term = (2 + pi^2*U((j-1)*dt)+ 2*a/gamma(2-gam)*((j-1)*dt)^(1-gam) + a*z0*gamma(3-gam)/gamma(2-2*gam) *((j-1)*dt)^(1-2*gam) )*f_space;
    elseif (cgam==1)
        f_term =  (2+pi^2*U((j-1)*dt)+a*z0*gamma(4-gam)/gamma(3-2*gam) *((j-1)*dt)^(2-2*gam))*f_space;
    end
    
    res = 2*u1 - u0 - a*dt^2*CQ + dt^2*(B\(-A*u1+f_term)); 
    res = res+a*dt^2*X*omega0*v0+a*dt*.5*omega0*u0- a*c*dt^2*wn0(j)*v0 - a*c*D*dt^2*wn1(j)*v1+a*c*dt*0.5*(1-D)*wn1(j)*u0; 
   
    
    %create u_{n+1}
    u_new = (1/(1+a*dt*.5*omega0+a*c*0.5*(1-D)*wn1(j)*dt))*res;

    %create new v and add to storage
    v_new = .5*(u_new-u0)/dt;
    v_store(:,j) = v_new;
    
    %reassign u_n and u_{n-1}
    u0 = u1; u1 = u_new;
    
    %work out error at this step
    err(j-1) = energy_error(u_ex,u_new,u0,j*dt,dt,ys);
    
    %in first step we need to replace v1 with its approximated value
    if (j==2)
       v1 = v_store(:,2);
    end

end

%update error calculations
err_max(end+1) = max(err);
err_end(end+1) = err(end);
    
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Determine the expected convergence rate

if (c==0)
    if (gam<0)
        expected_slope = min(1,1-gam);
    elseif (gam>0)
        expected_slope = 2-gam;
     end
elseif (c==1)
    if (gam<0)
        expected_slope = min(1-gam,2);
    elseif (gam>0)
        expected_slope = 2-gam;
    end
end

%Calculate convergence rate 

slope_max = log2(err_max(1:end-1)./err_max(2:end));
slope_end = log2(err_end(1:end-1)./err_end(2:end));


%Update stores

if ~exist('dx','var')
    dx = 1/M;
else
    dx = [dx 1/M];
end

if ~exist('dt_store','var')
    dt_store = dt;
else
    dt_store = [dt_store dt];
end

%Generate Plot

loglog(dx,dx.^expected_slope,'--')
hold on
loglog(dx,err_max)
xlabel('dx')
ylabel('Energy Error')
title(['Loglog plot of dx vs Maximum energy error for \gamma = ',num2str(gam)])
legend(['Expected: O(dx^{',num2str(expected_slope),'})']','Max error')
    
    