function Sigma = cov_sigma(p, rho)

Sigma = eye(p);
for i=1:p
    for j=1:p
        Sigma(i,j) = rho^(abs(i-j));
    end
end
