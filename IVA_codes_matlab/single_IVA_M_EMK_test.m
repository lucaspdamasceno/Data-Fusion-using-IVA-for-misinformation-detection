clc
clear 
close all
%% 
%MGGD sources Generation

N=300; %Sources
K=2; %Number of datasets
T=20729;
X=zeros(N,T,K);

X(:,:,1) = importdata('train_text_word2vec.mat')';
X(:,:,2) = importdata('train_img_res.mat')';

%%
%for kk=1:K
%    wnit(:,:,kk)= randn(N,N);
%end

%%

%fprintf('IVA_L')
tic
%W_IVA_L= iva_laplace(X);
toc

fprintf('IVA_L_Decp')
tic
%W_IVA_L_Decp= iva_laplace_decp(X);
toc

fprintf('IVA_L_SOS')
tic
%W_IVA_L_SOS= iva_l_sos_v1(X);
toc

fprintf('IVA_A_GGD')
tic
%W_IVA_A_GGD=iva_a_ggd_decp_RA_FP_P(X);
toc

fprintf('IVA_GGD')
tic
%W_IVA_GGD = iva_mpe_decp_v2(X);
toc


%%5
%% IVA-M-EMK
tic
W = iva_m_emk(X);
toc
%tic
%W_IVA_L= iva_laplace(X);
%toc