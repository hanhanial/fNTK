function CW_eval = CW_fit(params,S0,K_T_IV_matrix)

K = K_T_IV_matrix(:,1);
T = K_T_IV_matrix(:,2);
IV = K_T_IV_matrix(:,3);

k = log(K./S0);

vt = params(1);
mt = params(2);
wt = params(3);
nt = params(4);
rhot = params(5);

CW_eval = (1/4).*exp(-2*nt*T).*(wt^2).*(T.^2).*(IV.^4)+...
          (1-2*exp(-nt*T).*mt.*T-exp(-nt*T).*wt.*rhot.*sqrt(vt).*T).*(IV.^2)-...
          (vt+2*exp(-nt*T).*wt.*rhot.*sqrt(vt).*k+exp(-2*nt*T).*(wt^2).*(k.^2));

end
