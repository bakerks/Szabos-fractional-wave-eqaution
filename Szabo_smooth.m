%Solves szabos equation using standard CQ for gamma in (-1,1) with or
%without correction terms using the schemes given in equations 3.9 and 3.8
%respectivly for smooth u

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set parameters
p=2; %(1 bdf1) (2 bdf2) (3 trapez) used within CQ

c = 1; %removes correction if c=0

gam = -0.25; %order of FD in (-1,1)
cgam = ceil(gam);

%controls additional terms required for higher order derivatives
if (gam<0)
    X=0;
elseif (gam>0)
    X=1-c;
end

a0 = 20; %attenuation coeff >0
a = -a0*(4/pi)*gamma(-gam-1)*gamma(gam+2)*cos((pi*(gam+1))/2); %constant in front of FD

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Set up exact solution

%time part:

%Example 1: (used to make Figure1)
c1 = 24; c2 = 12;
U = @(t) sin(c1*t) + cos(c2*t);
Ud = @(t) c1*cos(c1*t) - c2*sin(c2*t);
Udd = @(t) -c1^2 * sin(c1*t) - c2^2 *cos(c2*t);

%Example 2:
% om = 24;
% U = @(t) 1+sin(om*t).^2+t.^2;
% Ud = @(t) 2*om*sin(om*t).*cos(om*t) +2*t;
% Udd = @(t) 2*(om^2)*(cos(om*t).^2 - sin(om*t).^2)+2;

%time and space combined:
u_ex = @(x,t) U(t)*sin(pi*x);
u_t_ex = @(x,t) Ud(t)*sin(pi*x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Define the source term f

%exact fractional derivative part:
if (cgam == 0)
    Ufd = @(t) simple_fint(t,Ud,-gam);
elseif (cgam == 1)
    Ufd = @(t) simple_fint(t,Udd,cgam-gam);
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

T = 1; N = 6*M;  dt =T/N;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%scale initial conditions and f's space part

u0 = B\rhs(U0,ys,ps);
v0 = B\rhs(V0,ys,ps);
f_space = rhs(@(x) sin(pi*x),ys,ps);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Approximate u1 with taylor expansion
udd0 = B\(-A*u0+(fdd(0)+fx(0))*f_space);

u1 = u0+dt*v0+.5*(dt^2)*udd0; %alternativly this could be done with the exact solution

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Gather convolution weights

weights = convw_gen(N,T,p-1, @(s) s.^(gam));

omega0 = weights(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Precalculate correction terms

wn1 = zeros(N,1);
wn0 = zeros(N,1);
ns = 0:N-1; %positions

if (cgam==0)
    wn1 = 0*ns;
elseif (cgam==1)
    for k = 0:N-1
        for j = 0:k
            wn1(k+1) = wn1(k+1) + weights(k-j+1)*(j);
        end
    end
    wn1 = (dt^(-gam)*ns'.^(1-gam))./gamma(2-gam) - wn1;
end


if (cgam==1)
    wn0 = - wn1;
    for k = 0:N-1
        for j = 0:k
             wn0(k+1) = wn0(k+1) - weights(k-j+1);
        end
    end
elseif (cgam==0)
    wn0 = (dt.*ns)'.^(-gam)./gamma(1-gam);
    for k = 0:N-1
        for j = 0:k
             wn0(k+1) = wn0(k+1) - weights(k-j+1);
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
    f_term = (fdd((j-1)*dt)+fx((j-1)*dt)+fcd((j-1)*dt))*f_space;
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
        if (Ud(0)==0)
            expected_slope = 2;
        else
            expected_slope = 1;
        end
    elseif (gam>0)
        if (Udd(0)==0)
            expected_slope = 2;
        else
            expected_slope = 2-gam;
        end 
    end
elseif (c==1)
    expected_slope = 2;
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
    
    