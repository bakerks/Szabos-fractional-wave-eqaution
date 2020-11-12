function l2g = local2global(ps)

Nl = length(ps);

l2g = zeros(Nl,max(ps)+1);

SC = 1;

if SC
    % first the linears except the last one which comes last.
    % bubble functions in the middle
    % to be used with static condensation
    for el = 1:Nl-1
        start = Nl+sum(ps(1:el-1))-el+1; %start for bubble functions
        l2g(el,1:(ps(el)+1)) = [el el+1 start+(1:ps(el)-1)];
    end
    el = Nl;
    start = Nl+sum(ps(1:el-1))-el+1; %start for bubble functions
    l2g(el,1:(ps(el)+1)) = [Nl start+ps(el) start+(1:ps(el)-1)];
else
    for el = 1:Nl
        l2g(el,1:(ps(el)+1)) = [1 ps(el)+1 2:ps(el)]+sum(ps(1:el-1));
    end
end