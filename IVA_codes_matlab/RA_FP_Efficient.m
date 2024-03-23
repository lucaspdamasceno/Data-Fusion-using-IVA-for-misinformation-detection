% Function that estimates the parameters of a multivariate generalized
% Gaussian distribution (MGGD).
% INPUT
%   - X : pxN observations drawn from a zero mean MGGD
%   - b_fix : Fixed value of beta. If b_fix == 0 then estimate beta, else
%   not.
% OUTPUT
%   - V : Normalized covariance matrix
%   - b : Shape parameter. If b_fix==0 then beta is the estimated value of
%   the shape parameter.
%   - m : Scale parameter.
%   - alpha : adaptive choice of step
%
% Coded by Lionel Bombrun (lionel.bombrun at u-bordeaux.fr);
%          Zois Boukouvalas (zb1 at umbc.edu);
%
% References:
%
% [1] A New Riemannian Averaged Fixed Point Algorithm for Parameter Estimation of MGGD model.


function [V, b, m, alpha,kk] = RA_FP_Efficient(X, b_fix, NMaxIter)
% Get the size of the data
[p, n] = size(X);

% Use MoM to get an initial value.
%[V, b, Fail] = Moments_MGGD_MEP_X(X, b_fix);
[V, b, Fail] = MoM(X, b_fix);

if p == 1
    S = X.^2 * IV;
else
    S = dot(X,V\X);
end

tol = 100 * ones(p^2 + 1, 1);
kk = 0;
NIter = 0;
Xouter = zeros(p,p,n);
for nn = 1:n
   Xouter(:,:,nn) = X(:,nn) * X(:,nn)'; 
end

while (any(tol > 0.01) && (NIter < NMaxIter))  % Repeat until tolerance is under 1 percent for every parameter (or NIter >= 100).
    
    Stens = repmat(permute(S,[1, 3, 2]), [p,p,1]);
    
    NIter=NIter+1;
    kk = kk + 1;
    alpha = 1/(kk);
    
    if p == 1
        V_new = (b/n) * sum(S.^(b - 1) .* X.^2);
    else
		V_new = Stens .^(b-1) .* Xouter;
        V_new = sum(V_new, 3);
        V_new = (V^((1/2))) * (( (V^(-0.5)) * (V_new) * (V^(-0.5)) )^(alpha)) * (V^((1/2)));
    end
    V_new = p*V_new/trace(V_new);
    
    if p == 1
        S = X.^2 * IV;
    else
        S = dot(X,V_new\X);
    end
    
    if b_fix ~= 0
        b_new = b_fix;
    else
        
        b_new = Newton_Raphson(b, S, p, n);
    end 
    
    tol = 100 * [reshape(abs((V_new - V)./V), p^2, 1); abs((b_new - b)/b)];  % Relative change in percent
    V = V_new;
    b = b_new;
    
end

m = calc_m(b, S, p, n);

end

function m = calc_m(b, S, p, n)

F1 = sum(S.^b);
m = ((b*F1)/(p*n))^(1/b);

end