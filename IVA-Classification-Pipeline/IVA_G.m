%% 
%MGGD sources Generation

N=100; %Sources
K=2; %Number of datasets
T=9142;
X=zeros(N,T,K);

X(:,:,1) = readmatrix('pca_train_txt.csv')';
X(:,:,2) = readmatrix('pca_train_img_vgg.csv')';

%%
%for kk=1:K
%    wnit(:,:,kk)= randn(N,N);
%end

%%


fprintf('IVA_G')
tic
W_IVA_G=iva_second_order(X);
toc
save('W_IVA_G.mat', 'W_IVA_G')
