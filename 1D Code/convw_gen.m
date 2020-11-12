function f = convw_gen(N,T,MS, Ks)
%function f = convw_gen(N,T,MS, Ks)
%returns conv. weight \omega_j(K) for j = 0, .., N
%MS = (0 bdf1) (1 bdf2) (2 trapez)
%
% 2010 - Lehel Banjai

dt = T/N;
if (MS==0)
    gam = @(z) 1-z; %BDF1
elseif (MS==1)
    gam = @(z) .5*(1-z).*(3-z); 
elseif (MS==2)
    gam = @(z) 2*(1-z)./(1+z); %trapez
else
    fprintf('only multistep 0  1 2 possible for types of multistep\n');
    return;
end

zs = exp(-1i*2*pi*(0:4*N)/(4*N+1)).';
%lam = eps^(1/(5*N));
lam = 10^(-15/(5*N));

%note: so error for first N terms will be eps^(4/5) \approx 3e-13
tmp = real(ifft(Ks(gam(lam*zs)/dt)));
f = (lam.^(0:-1:-N).').*tmp(1:N+1);