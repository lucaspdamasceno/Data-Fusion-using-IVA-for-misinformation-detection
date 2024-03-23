%
clear all
close all
clc
%%
N=5;
K=2;
T=10000;

S=zeros(N,T,K);

for n=1:N
    beta(n) = rand*(4-0.25)+0.25;
    rho(n) = rand*(0.7-0.5)+0.5; %Correlation within the SCV
    Sigma = cov_sigma(K, rho(n)); % scatter matirx
    %U = rand(K);
    %Sigma = U*U';
    S(n,:,:)=generate_MGGD(T, K, Sigma, beta(n))';
    %S(n,:,:)=randmv_laplace(K,T)';
    %S(n,:,:) = (Z(1:K/2,:) + 1i*Z(K/2+1:end,:)).';
end

for kk=1:K
    A(:,:,kk) = rand(N,N);
end


for k=1:K
    %A(:,:,k)=vecnorm(A(:,:,k))';
    X(:,:,k)=A(:,:,k)*S(:,:,k);
end

%
for kk=1:K
    wnit(:,:,kk) = rand(N,N);
end
%%
%}

%tic
%[W3,cost3,shapeParam3,isi3,iter] = iva_a_ggd_decp_RA_FP_P(X,'A',A,'initW',wnit);
%toc
