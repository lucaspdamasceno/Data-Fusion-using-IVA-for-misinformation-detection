% Function that estimates the parameters of a Multivariate Generalized
% Gaussian distribution (MGGD). This function is estimating the parameters
% of the scatter matrix of a MGGD based on the Fisher scoring algorithm. 
% INPUT
%   - X: data generated from MGGD distribution. pXn matrix where p denotes
%   the dimension of our space and n the number of samples.
%   - NMaxIter: Maximum number of iterations.
%   - Sigma: Scater matrix. 
% OUTPUT
%   - V: Covariance 
%	- m: Scale parameter 

% Notation is based on the paper [1]"Parameter Estimation For Multivariate 
% Generalized Gaussian Distributions" by F. Pascal, L. Bombrun, J.Y.
% Tourneret, Y. Berthoumieu. This code is a variation of the code used for
% the above paper. Also for the construction of the Fisher scoring
% algorithm we used the paper [2]"On the Geometry of Multivariate Generalized
% Gaussian Models" by Verdoolaege and Scheunders. Equations (17) and (18). 

% Code modified 
% Zois Boukouvalas, University of Maryland Baltimore County, August 2014.

function [V, b, m, NIter] = fisher_scoring_complete(X,b_fix, NMaxIter)

% Gets the dimension of the sample and of the space
[p, n] = size(X);

% Initial guess for the ML estimator.
[V, b, Fail] = MoM(X, b_fix); % Method of moments.


% Initialize the gradient of the likelihood
summa=dot(X,V\X).^b;

FX = inv(det(V))* (sum(summa)^(-p/b));

fX = zeros(p);
for i = 1:n
    fX = fX + ((summa(i))^(b - 1)) * X(:, i) * X(:, i)';
end
fX = p.*fX;

derFX = FX*((V\(fX - V))/V);
derFX = derFX/norm(derFX);

%Construct the Fisher information matrix based on equations (17) and (18)
%from [2].
%This function constructs the Fisher information matrix for given values
%of shape parameter \beta

G = zeros(p,p);
gij = 1/4 * ((p+2*b)/(p+2)-1);
gii = 1/4 * (3*(p+2*b)/(p+2)-1);
for i = 1:p
    for j = 1:p
        if i == j
            G(i,i) = gii;
        else
            G(i,j) = gij;
        end
    end
end

tol = 100 * ones(p^2 + 1, 1);
NIter = 0;

while (any(tol > 0.01) && (NIter < NMaxIter))  % Repeat until tolerance is under 1 percent for every parameter (or NIter >= 100).
    NIter = NIter + 1;
    
    V_new = V + 0.01*(G\(((FX)^((n-2)/2))*derFX));
        
    V_new = p*V_new/trace(V_new); %Normalize based on the idea from reference [1]
        
    summa=dot(X,V_new\X).^b;
    FX = inv(det(V_new))* sum(summa)^(-p/b);
    
    fX = zeros(p);
    for i = 1:n
        fX = fX + ((summa(i))^(b - 1)) * X(:, i) * X(:, i)';
    end
    fX = p.*fX;
    derFX = FX.*((V_new\(fX - V_new))/V_new);
    
    derFX = derFX/norm(derFX);
    
    if b_fix~=0
        b_new = b_fix;
    else
        b_new = Newton_Raphson(b, summa.^(1/b), p, n);
    end
               
    tol = 100 * [reshape(abs((V_new - V)./V), p^2, 1); abs((b_new - b)/b)];  % Relative change in percent
    
    V = V_new;
    b = b_new;
           
end



S = dot(X,V\X);
m = calc_m(b, S, p, n);


if NIter == NMaxIter
    disp('FS_MGGD: ATTENTION: maximum number of iterations reached!');
end

end

function m = calc_m(b, S, p, n)
% b = beta
   
    F1 = sum(S.^b);
    m = ((b*F1)/(p*n))^(1/b);
end