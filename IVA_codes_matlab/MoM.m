function [V,b, Fail] = MoM(X, b_fix)
% Same as Moments_MMD, but using an input matrix X (p x n) for the data.
% 
% Calculates estimates of the parameters beta and V of a
% multivariate generalized Gaussian pdf fit to the data in WIm, which is
% 3-band colour data, using the method of moments. The image with index
% NIm is chosen for a single wavelet subband (NS). If b_fix ~= 0 then a
% fixed shape parameter beta is used (1 for Gaussian, 0.5 for Laplace).
% Otherwise, to solve the equation for beta, the half-interval method is
% used. Col is a vector of colour indices (1 = r, 2 = g, 3 = b) indicating
% the colour bands that one actually wishes to model. This also determines
% the dimensionality of the probability space p.
%
% ATTENTION: MEP version!!

    Fail = 0;
    [p, n] = size(X);    
%     Var = zeros(p);  % Sample variance
%     for i = 1:n
%         Var = Var + X(:, i) * X(:, i)';
%     end 
%     Var = Var/n;
    Var = cov(X.');
    
    if b_fix == 0
        %         IV = inv(Var);
        varInvX = Var \ X;
        %         gamma2 = 0;  % Sample kurtosis
        %         for i = 1:n
        %             %             gamma2 = gamma2 + (X(:, i)' * IV * X(:, i))^2;
        %             gamma2 = gamma2 + (X(:, i)' * varInvX(:,i))^2;
        %         end
        gamma2 = sum(dot(X, varInvX).^2);
        gamma2 = (1/n) * gamma2 - p * (p + 2);
        
        [b, IsMaxIter] = Zero_b(gamma2, p, n);
    else
        b = b_fix;
        IsMaxIter = 0;  % No Maximum Likelihood necessary.
    end
    
    if IsMaxIter == 1
        Fail = 1;
    end
    V = (p * gamma(p/(2*b)) / (2^(1/b) * gamma((p + 2)/(2*b)))) * Var;
end


function [b, IsMaxIter] = Zero_b(gamma2, p, n)

    MaxIter = 100;
    IsMaxIter = 0;    
    R = [0.1, 5];  % Initial interval for b   
    Err = 0;
    sa = sign(g(R(1), gamma2, p, n));
    sb = sign(g(R(2), gamma2, p, n));  
    if sa == sb
        Err = 1;
        b = 1;
    else
        gb = 1;
        NIter = 0;        
        while (abs(gb) > 1e-3) && (NIter < MaxIter)
            NIter = NIter + 1;
            b = mean(R);  % Half-point
            gb = g(b, gamma2, p, n);
            if sb * gb > 0
                R(2) = b;
            elseif sb * gb < 0
                R(1) = b;
            elseif gb == 0
                break;  % Found an exact root.
            end
        end
        if NIter == MaxIter
%             disp('Moments_MGGD: maximum number of iterations reached. Results may be unreliable.');
            IsMaxIter = 1;
%            b = 0.5;
        end        
    end
end


function Res = g(b, gamma2, p, n)
% b = beta
   
    Res = p^2 * gamma(p./(2*b)) .* gamma((p + 4)./(2*b)) ...
        - (gamma((p + 2)./(2*b))).^2 * (p * (p + 2) + gamma2);
end