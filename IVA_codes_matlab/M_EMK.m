function [Lambda,PDF, r, fCst5] = M_EMK(S)                                                  
%%
%%Definition of parameters and data generation
K=size(S,1); %Dimension of space
T=size(S,2); %Number of sample size
%K=2;
%T=100000;
M=5; %Total number of measuring functions (Global+Local)
M4=4; %Total number of local measuring functions
constType=4; %Defines the type of the fourth measuring function

%beta=0.5; %Shape parameter for each of the marginals
%mu=6; %Mean of each of the marginals
%alpha=0.7; %Weight parameter for each
%paramGGD=[alpha 2*beta -mu; 1-alpha 2*beta mu];
%S = data;
%S = genMixGGDmD( paramGGD, 1, 1, K, T );
%% Define the global constraints
numCst=1;
r(numCst).func=@(X) ones(1,length(X));
r(numCst).diff=@(X) zeros(size(X));
r(numCst).int=@(X) X/2;
%
numCst=2;
r(numCst).func=@(X) ones(1,size(X,1))*X;
r1(numCst).func=@(X) X;
r(numCst).diff=@(X) ones(size(X));
r(numCst).int=@(X) (X.^2)/2;
%
numCst=3;
r(numCst).func=@(X) ones(1,size(X,1))*(X.^2);
r1(numCst).func=@(X) (X.^2);
r(numCst).diff=@(X) 2*X;
r(numCst).int=@(X) (X.^3)/3;
%
numCst=4;
if constType == 1
    r(numCst).func=@(X) ones(1,size(X,1))*(X.^4);
    r1(numCst).func=@(X) (X.^4);
    r(numCst).diff=@(X) 4*X.^3;
    r(numCst).int=@(X) (X.^5)/5;
elseif constType == 2
    r(numCst).func=@(X) ones(1,size(X,1))*(abs(X)./(1+abs(X)));
    r(numCst).diff=@(X) sign(X)./((1+abs(X)).^2);
    r(numCst).int=@(X) sign(X).*abs(X).*log(1+abs(X))+(abs(X)+1).*log(abs(X)+1)-abs(X);
elseif constType == 3
    r(numCst).func=@(X) ones(1,size(X,1))*(X.*abs(X)./(10+abs(X)));
    r(numCst).diff=@(X) abs(X).*(20+abs(X))./((10+abs(X)).^2);
    r(numCst).int=@(X) 0.5*(-20*abs(X)+X.^2+200*log(10+abs(X)));
else
    r(numCst).func=@(X) ones(1,size(X,1))*(X./(1+X.^2));
    r1(numCst).func=@(X) (X./(1+X.^2));
    r(numCst).diff=@(X) (1-X.^2)./((1+X.^2).^2);
    r(numCst).int=@(X) 0.5*log(1+X.^2);
end

%X = normalize(S,2);
X=S;
%% Preprocess the data.
%
%Define the local constraint (Gaussian Kernel)
muCst5=zeros(1,K);
sigmaCst5=eye(K);
funCst5 = mvnpdf(X',muCst5,sigmaCst5);
fCst5 = funCst5';

%% Joint pdf, including all the constraints
p=@(X,Lambda) exp(-1+Lambda(1)*r(1).func(X)+Lambda(2)*r(2).func(X)+Lambda(3)*r(3).func(X)+Lambda(4)*r(4).func(X)+Lambda(5)*fCst5);%joint PDF

%% Marginal pdf without exp(-1+lambda(1)) up to the fourth constraint
pk=@(x,Lambda) exp(Lambda(2)*r(2).func(x)+Lambda(3)*r(3).func(x)+Lambda(4)*r(4).func(x));

%% Estimate initial lambda and obtain the sample 
%averages. These Lambdas are evaluated using only global measuring functions
[Lambda, Alpha] = maxEntropyDenSI2yEstLEmD( X, M4, r, pk, r1, fCst5);

% Evaluate Lambdas using all constraints
[Lambda] = maxEntropyDenSI2yEstNImD( T, K, M4, M, r, Lambda, Alpha, constType, p, X);

A = normalization(r,Lambda,K);
PDF = p(X,Lambda);
PDF = PDF/A;
%{
%A = normalization(r,Lambda,K);
%PDF=PDF/A;
scatter3(X(1,:),X(2,:),PDF,[],PDF,'.')
figure(2)
movegui('west')
hist3(X','Nbins',[60 60],'CDataMode','auto','FaceColor','interp')
figure(3)
movegui('east')
%}
%% Plots
%Plot approximated pdf using all the constraints (not iterative)
%A = normalization(r,Lambda,K);
%f=f/A;
%{
if K == 2
%Plot 'Histogram of Generated Data'
figure;
hist3(S','Nbins',[60 60],'CDataMode','auto','FaceColor','interp')
view([74 26]) %Camera position
movegui('west')
title('Histogram of generated data')    

figure;
movegui('east')
scatter3(S(1,:),S(2,:),PDF,[],PDF,'.')
colorbar
%axis([-4,4,-4,4]) %suport set ?(x)
view([74 26]) %Camera position
title('Approximated PDF w.r.t iteration')
end

if K == 3
figure;
movegui('east')
scatter3(S(1,:),S(2,:),S(3,:),[],PDF,'.')
colorbar
%axis([-4,4,-4,4]) %suport set ?(x)
view([74 26]) %Camera position
title('Approximated PDF w.r.t iteration')
end
%}

function [Lambda, Alpha] = maxEntropyDenSI2yEstLEmD( S, M4, r, pk, r1, fCst5)
[K, T]=size(S);

B=zeros(M4-1,M4-1);
for i=1:M4-1
    for j=1:M4-1
        B(i,j)=sum(ones(1,K)*(r1(i+1).func(S).*(r1(j+1).func(S))))/T; %This is an estimation of the Jacobian, using sample averages. (No pdf yet)
    end
end

A=zeros(M4-1,1);
for i=1:M4-1
    A(i)=sum(r(i+1).func(S))/T;
end

Lambda=-B\A;
Lambda=[0;Lambda];

PDFk = @(x) pk(x,Lambda);
probIntk=quadgk(PDFk,-Inf,Inf);
Lambda(1)=1-K*log(probIntk);
Alpha=[1;A;sum(fCst5)/T];


function [LambdaNew, fConst , LambdaDelta, J, fval] = maxEntropyDenSI2yEstNImD( T, K, M4, M, r, Lambda, Alpha, constType, p, X)
maxIter=500;
tolerance = 1e-008;

Alpha(1)=K;
Lambda(end+1)=0; %Set Lambda_5=0 

LambdaOld=Lambda;
LambdaIter = zeros(M,1);

if constType == 1
    checkIdx=M-1;
else
    checkIdx=M-2;
end

%Init for The Quasi-Random Numbers and Space of Integration(SI)
dim_mea = 2^K; % dimensional measure 
numpoints_generation = 500; % number of points
numpoints = numpoints_generation/4; % number of points for each spaces 
Xn = quasirand(1,numpoints_generation,K); % quasi-random points generation for the interval [0,1]
Xt = Xn*2-1; % Xn*2 -> translation and Xn*2-1 -> dilation to cover the entire integration space [-1,1]
SI1 = zeros(numpoints,K); % Space of Integration 1 (SI1)
SI2 = zeros(numpoints,K); % Space of Integration 2 (SI2)
SI3 = zeros(numpoints,K); % Space of Integration 3 (SI3)
SI4 = zeros(numpoints,K); % Space of Integration 4 (SI4)

i = 0; % hypercubes counter for the SI1
j = 0; % hypercubes counter for the SI2
k = 0; % hypercubes counter for the SI3
l = 0; % hypercubes counter for the SI4

for n = 1 : numpoints_generation
    
     if Xt(n,1)>0 && Xt(n,2)>0
          i = i + 1;
          SI1(i,:) = Xt(n,:);      
     end   
    
    if Xt(n,1)<0 && Xt(n,2)<0
          j = j + 1;
          SI2(j,:) = Xt(n,:);
    end
        
    if Xt(n,1)>0 && Xt(n,2)<0
          k = k + 1;
          SI3(k,:) = Xt(n,:);
    end
        
    if Xt(n,1)<0 && Xt(n,2)>0
          l = l + 1;
          SI4(l,:) = Xt(n,:);
    end    
end

SI1 = [SI1 ; SI2];
SI2 = [SI3 ; SI4];

%GAUSSIAN KERNEL
muGK = mean(SI1);
sigmaGK = cov(SI1);
fGK = mvnpdf(SI1,muGK,sigmaGK);
q = fGK';

muGKy = mean(SI2);
sigmaGKy = cov(SI2);
fGKy = mvnpdf(SI2,muGKy,sigmaGKy);
qy = fGKy';
    
for iter = 2 : maxIter
    if LambdaOld(checkIdx)>0
        disp('Wrong lambda!!!');
        break;
    end 
    
    %Initializations for the Quasi-Monte Carlo Integration
    J=zeros(M,M); %This is for the Jacobian
    fConst=zeros(M,1); %For the E{r-alpha} Cost function
    
    %Evaluate the cost function E{r-alpha} for the global constraints by Quasi-Monte Carlo Integration
    
    for i=1:M4
            fval =r(i).func(SI1').*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(SI1')+LambdaOld(3)*r(3).func(SI1')+LambdaOld(4)*r(4).func(SI1')+LambdaOld(5)*q);
            fvaly =r(i).func(SI2').*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(SI2')+LambdaOld(3)*r(3).func(SI2')+LambdaOld(4)*r(4).func(SI2')+LambdaOld(5)*qy);
            QMCint = [sum(fval(1:numpoints)/numpoints),sum(fvaly(1:numpoints)/numpoints)];
            QMCintVal = dim_mea*sum(abs(QMCint));
            fConst(i) = (QMCintVal - Alpha(i)); 
    end
        
    %Evaluate the cost function for Gaussian Kernel
            fvalCst5 =q.*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(SI1')+LambdaOld(3)*r(3).func(SI1')+LambdaOld(4)*r(4).func(SI1')+LambdaOld(5)*q);
            fvalCst5y =qy.*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(SI2')+LambdaOld(3)*r(3).func(SI2')+LambdaOld(4)*r(4).func(SI2')+LambdaOld(5)*qy);
            QMCintCst5 = [sum(fvalCst5(1:numpoints)/numpoints),sum(fvalCst5y(1:numpoints)/numpoints)];
            QMCintValCst5 = dim_mea*sum(abs(QMCintCst5)); 
            fConst(5) = QMCintValCst5 - Alpha(5);
            
            fConstN = fConst/norm(fConst,1); % is the 1-norm 
                           
    %Evaluate the Jacobian matrix by Quasi-Monte Carlo Integration. 
	%This matrix is symmetrix poSI2ive definite. In this loop Jacobian 
	%corresponds to the global constraints
	
    for i=1:M4
        for j=i:M4
            fvalJ =r(i).func(SI1').*r(j).func(SI1').*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(SI1')+LambdaOld(3)*r(3).func(SI1')+LambdaOld(4)*r(4).func(SI1')+LambdaOld(5)*q);            
            fvalJy =r(i).func(SI2').*r(j).func(SI2').*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(SI2')+LambdaOld(3)*r(3).func(SI2')+LambdaOld(4)*r(4).func(SI2')+LambdaOld(5)*qy);
            QMCintJ = [sum(fvalJ(1:numpoints)/numpoints),sum(fvalJy(1:numpoints)/numpoints)];
            QMCintValJ = dim_mea*sum(abs(QMCintJ)); 
            J(i,j)= QMCintValJ;
            J(j,i)=J(i,j);
        end
    end
         
    %Last row and column of the Jacobian matrix. 
    %Corresponds to the local constraint
	
    for i=1:M4
            fvalJCst5 =r(i).func(SI1').*q.*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(SI1')+LambdaOld(3)*r(3).func(SI1')+LambdaOld(4)*r(4).func(SI1')+LambdaOld(5)*q);
            fvalJCst5y =r(i).func(SI2').*qy.*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(SI2')+LambdaOld(3)*r(3).func(SI2')+LambdaOld(4)*r(4).func(SI2')+LambdaOld(5)*qy);
            QMCintJCst5 = [sum(fvalJCst5(1:numpoints)/numpoints),sum(fvalJCst5y(1:numpoints)/numpoints)];
            QMCintValJCst5 = dim_mea*sum(abs(QMCintJCst5));
            J(i,5)=QMCintValJCst5;
            J(5,i)=J(i,5);
    end
    
    %Last element of the Jacobian 
	
            fvalCst5J =q.*q.*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(SI1')+LambdaOld(3)*r(3).func(SI1')+LambdaOld(4)*r(4).func(SI1')+LambdaOld(5)*q);
            fvalCst5Jy =qy.*qy.*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(SI2')+LambdaOld(3)*r(3).func(SI2')+LambdaOld(4)*r(4).func(SI2')+LambdaOld(5)*qy);
            QMCintCst5J = [sum(fvalCst5J(1:numpoints)/numpoints),sum(fvalCst5Jy(1:numpoints)/numpoints)];
            QMCintValCst5J = dim_mea*sum(abs(QMCintCst5J));
            J(5,5)= QMCintValCst5J;
            
    %Evaluating Lambdas
    LambdaDelta=J\fConstN;
    
    beta=1;
    if LambdaOld(checkIdx)>LambdaDelta(checkIdx)
        beta=0.99*LambdaOld(checkIdx)/LambdaDelta(checkIdx);
    end
    
    LambdaNew=LambdaOld-beta*LambdaDelta;
    LambdaIter(:,iter) = LambdaNew(:);
    if abs(1-((LambdaIter(:,iter)'*LambdaIter(:,iter-1)))/(LambdaIter(:,iter-1)'*LambdaIter(:,iter-1)))<tolerance
       break;
    end  

    
    LambdaOld=LambdaNew/sqrt(cov(LambdaNew)); %Normalized lagrange multipliers
    Cost(iter)=sum((fConstN.^2)/length(fConstN));
    
   
    %Plot approximated pdf using all the constraints (ITERATIVE MODE)
    %f = p(X,LambdaNew);
    %A = normalization(r,LambdaNew,K);
    %f=f/A;
    %figure(i);
    %movegui('east')
    %scatter3(S(1,:),S(2,:),S(3,:),[],1.8*f,'.')
    %colorbar
    %axis([-2,2,-2,2]) %suport set ?(x)
    %title(['Approximated PDF w.r.t iteration (',num2str(iter),')'])  
    
    
end
%{
figure;
semilogy(Cost,'LineWidth',2);
xlabel('Number of iterations','Interpreter','latex','FontSize',15)
ylabel('Cost','Interpreter','latex','FontSize',15)
title('Cost function w.r.t number of iterations')
%}

function c = normalization(r,Lambda,K)
%--------------------------------------------------------------------
%Compute the normalizing constant is used to reduce any probability  
%function to a probability denSI2y function with total probability of one.
%--------------------------------------------------------------------
dim_mea = 2^K;
numpoints_generation = 500;
numpoints = numpoints_generation/4;
Xn = quasirand(1,numpoints_generation,K);
Xt = Xn*2-1;
SI1 = zeros(numpoints,K); 
SI2 = zeros(numpoints,K);
SI3 = zeros(numpoints,K); 
SI4 = zeros(numpoints,K); 
i = 0;
j = 0; 
k = 0; 
l = 0; 

for n = 1 : numpoints_generation    
     if Xt(n,1)>0 && Xt(n,2)>0
          i = i + 1;
          SI1(i,:) = Xt(n,:);      
     end   
     if Xt(n,1)<0 && Xt(n,2)<0
          j = j + 1;
          SI2(j,:) = Xt(n,:);
     end       
     if Xt(n,1)>0 && Xt(n,2)<0
          k = k + 1;
          SI3(k,:) = Xt(n,:);
     end   
     if Xt(n,1)<0 && Xt(n,2)>0
          l = l + 1;
          SI4(l,:) = Xt(n,:);
     end  
end
SI1 = [SI1 ; SI2];
SI2 = [SI3 ; SI4];

muGK = mean(SI1);
sigmaGK = cov(SI1);
fGK = mvnpdf(SI1,muGK,sigmaGK);
q = fGK';

muGKy = mean(SI2);
sigmaGKy = cov(SI2);
fGKy = mvnpdf(SI2,muGKy,sigmaGKy);
qy = fGKy';

fval =exp(-1+Lambda(1)+Lambda(2)*r(2).func(SI1')+Lambda(3)*r(3).func(SI1')+Lambda(4)*r(4).func(SI1')+Lambda(5)*q);
fvaly =exp(-1+Lambda(1)+Lambda(2)*r(2).func(SI2')+Lambda(3)*r(3).func(SI2')+Lambda(4)*r(4).func(SI2')+Lambda(5)*qy);

QMCint = [sum(fval(1:numpoints)/numpoints),sum(fvaly(1:numpoints)/numpoints)];
c = dim_mea*sum(abs(QMCint));


function [y,x] = genMixGGDmD( paramGGD, b, a, K, T)
N=size(paramGGD, 1);

Beta=paramGGD(:,2);
Mu=paramGGD(:,3);
Alpha=sqrt(gamma(1./Beta)./gamma(3./Beta));

Mix=zeros(N,T);
MixT=rand(1,T);
lowThe=0;
highThe=0;
for i=1:N
    highThe=lowThe+paramGGD(i,1);
    Mix(i,:)=(MixT>=lowThe).*(MixT<highThe);
    lowThe=highThe;
end

s=zeros(N,T);
for i=1:N
    s(i,:)=ggrnd(0, Alpha(i), Beta(i), 1, T);
    s(i,:)=(s(i,:)-mean(s(i,:)))/std(s(i,:))+Mu(i);
end

s=s.*Mix;
x=sum(s,1);

x=filter(b, a, x);
x=(x-mean(x))/std(x);


y=convmtx(x,K);
y=y(end:-1:1,1:end-K+1);
y=(y-mean(y,2)*ones(1,T))./(std(y')'*ones(1,T));


function [X, R] = pre_processing(X)
% pre-processing program
[N,T] = size(X);
% remove DC
Xmean=mean(X,2);
X = X - Xmean*ones(1,T);    

% spatio pre-whitening 1 
R = X*X'/T;                 
%P = inv_sqrtmH(R);  %P = inv(sqrtm(R));
X = R\X;

