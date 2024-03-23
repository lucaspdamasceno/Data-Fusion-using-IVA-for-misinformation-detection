% GENERATE_MGGD
% Cette fonction permet de générer des échantillons selon une loi
% Multivariate Generalized Gaussian.
% paramètres d'entrée :
%   - N : nombre de réalisations
%   - p : dimension
%   - Sigma : matrice de covariance de taille pxp.
%   - beta : paramètre 'beta' de la loi de tau.
% paramètre de sortie :
%   - X : image de taille pxN contenant les réalisations de la MGGD

function X = generate_MGGD(N, p, Sigma, beta)

X = transpose(RandSphere(N,p));

X = (Sigma)^(0.5)*X;


tau = (gamrnd(p/(2*beta),2,1,N)).^(1/(2*beta));
tau = repmat(tau, p, 1);


X = tau.*X;
