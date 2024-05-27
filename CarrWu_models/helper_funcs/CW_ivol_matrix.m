function CW_ivol_vec = CW_ivol_matrix(K_T_matrix,params,S0)

K_vec = K_T_matrix(:,1);
T_vec = K_T_matrix(:,2);

vt = params(1);
mt = params(2);
wt = params(3);
nt = params(4);
rhot = params(5);

for i=1:length(K_vec)
    
    K = K_vec(i);
    T = T_vec(i);
    k = log(K/S0);
    
    p = [(1/4).*exp(-2*nt*T).*(wt^2).*(T.^2),...
          (1-2*exp(-nt*T).*mt.*T-exp(-nt*T).*wt.*rhot.*sqrt(vt).*T),...
          -(vt+2*exp(-nt*T).*wt.*rhot.*sqrt(vt).*k+exp(-2*nt*T).*(wt^2).*(k.^2))];
      
    r = roots(p);

    CW_ivol_vec(i) = sqrt(r(r>0));

end

end