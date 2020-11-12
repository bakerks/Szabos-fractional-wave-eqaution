function [f, ts] = cq_ms(N, T, MS, K, g)
%function [f, ts] = cq_ms(N, T, MS, K, g)
%compute f = K(\partial_t) g with the multistep method MS at t_j,
% ts = [t_0, \dots, t_N]
% j= 0, 1, \dots, N, t_j = jT/N
%MS = (0 bdf1) (1 bdf2) (2 trapez)
%example:
%Ks = @(s) s;g = @(t) sin(t).^2; 
%[f,ts] = cq_ms(100,4,1,Ks,g); semilogy(ts,abs(f-sin(2*ts)));
%
% 2010 - Lehel Banjai

gs = g((0:N)'*T/N);

w = convw_gen(N,T,MS,K);

f = zeros(N+1,1);

for n = 0:N
    f(n+1) = sum(gs(1:n+1).*w(n+1:-1:1));
end
ts = T*(0:N)'/N;