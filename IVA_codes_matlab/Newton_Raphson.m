% NEWTON_RAPHSON
% paramètres d'entrée :
%   - b : Ancienne estimée du paramètre 'beta'.
%   - S : Terme quadratique.
%   - p : Dimension du vecteur d'observation.
%   - n : Nombre d'observations.
% paramètre de sortie
%   - b_new : nouvelle estimée du paramètre 'beta'.

function b_new = Newton_Raphson(b, S, p, n)
Sb = S.^b;
Slog = log(S);
SbSum = sum(Sb);
SlogSbSum = sum(Slog .* Sb);

b_new = b - f(b, S, p, n, Sb, Slog, SbSum, SlogSbSum)/g(b, S, p, n, Sb, Slog, SbSum, SlogSbSum);
if(b_new<=0)
    b_new = b/2;
end
if(imag(b_new)~=0)
    b_new = b;
end
end

function Res = f(b, S, p, n, Sb, Slog, SbSum, SlogSbSum)
F1 = SlogSbSum;
F2 = SbSum;
Res = p*n*F1/(2*F2) - p*n/(2*b)*(psi(p/(2*b))+log(2)) - n - p*n/(2*b)*log(b*F2/(p*n));
Res = Res/n;
end

function Res = g(b, S, p, n, Sb, Slog, SbSum, SlogSbSum)
F1 = SbSum;
F2 = SlogSbSum;
F3 = sum(Slog.^2 .* Sb);

Res = p*n/2 * ((F1*F3 - F2^2)/(F1^2)) + p*n/(2*b^2)*(psi(p/(2*b))+log(2)) - p*n/(2*b)*(-p*psi(1,p/(2*b))/(2*b^2));
Res = Res + p*n/(2*b^2)*log(b/(p*n)) - p*n/(2*b^2) + p*n/(2*b^2)*log(F1) - p*n/(2*b)*F2/F1;
Res = Res/n;
end


