function [S,A,X, wnit, Inv] = gen(N,K,T, density, sim)
    
    % Name of .mat file to be generated
    name = "N="+num2str(N)+"K="+num2str(K)+"T="+num2str(T)+"density="+num2str(density)+"sim="+num2str(sim);

    % Initialize K NxT matrices filled with zeros
    S=zeros(N,T,K);
    % Initialize inverse covariance N KxK matrices
    Inv=zeros(K,K,N)
    
    % Generate N random sparse positive definite precision matrix and K corresponding MVN source matrices
    for n=1:N
        rc = 0.1;
        precision = sprandsym(K, density, rc, 1);
        full(precision);
        Inv(:,:,n) = precision
        Sigma = inv(precision);
        full(Sigma);
        mu = zeros(1, K);
        S(n,:,:) = mvnrnd(mu,Sigma,T);
    
        for k = 1 : K
            S(n,:,k) = S(n,:,k) - mean(S(n,:,k));
            S(n,:,k) = S(n,:,k)/sqrt(var(S(n,:,k)));
        end
    end
    
    % Initialize K N*N random mixing matrice and K N*T matrices filled with zeros
    A = randn(N,N,K);
    X = zeros(N,T,K);
    
    % Linearly mix A, S to form "observed data"
    for k=1:K
        X(:,:,k)=A(:,:,k)*S(:,:,k);
    end

    % Generate a random first guess for W
    for kk=1:K
        wnit(:,:,kk) = rand(N);
    end
    
    % Save mat file with S,A,X,wnit data frames
    save('/Users/egzonarexhepi/Desktop/MS/Thesis/Code/IVA/SparseIVA/matFiles/' +name+ '.mat')
    
end