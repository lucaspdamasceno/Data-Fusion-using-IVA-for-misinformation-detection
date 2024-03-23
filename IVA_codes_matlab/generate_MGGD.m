% GENERATE_MGGD
% Cette fonction permet de g�n�rer des �chantillons selon une loi
% Multivariate Generalized Gaussian.
% param�tres d'entr�e :
%   - N : nombre de r�alisations
%   - p : dimension
%   - Sigma : matrice de covariance de taille pxp.
%   - beta : param�tre 'beta' de la loi de tau.
% param�tre de sortie :
%   - X : image de taille pxN contenant les r�alisations de la MGGD

function X = generate_MGGD(N, p, Sigma, beta)

X = transpose(RandSphere(N,p));

X = (Sigma)^(0.5)*X;


tau = (gamrnd(p/(2*beta),2,1,N)).^(1/(2*beta));
tau = repmat(tau, p, 1);


X = tau.*X;
