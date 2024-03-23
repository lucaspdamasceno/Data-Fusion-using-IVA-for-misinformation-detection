function M-EMK( )
clc
clear
close all
%%
%%Definition of parameters and data generation
K=2; %Dimension of space
T=1000000; %Number of sample size
M=5; %Total number of measuring functions (Global+Local)
M4=4; %Total number of local measuring functions
constType=4; %Defines the type of the fourth measuring function

beta=0.5; %Shape parameter for each of the marginals
mu=1; %Mean of each of the marginals
alpha=0.7; %Weight parameter for each
paramGGD=[alpha 2*beta -mu; 1-alpha 2*beta mu];
S = genMixGGDmD( paramGGD, 1, 1, K, T );

% Define the global constraints
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

%%
%Preprocess the data.
[X, ~] = pre_processing(S);
%%
%Define the local constraint (Gaussian Kernel)
muCst5=mean(X');
sigmaCst5=cov(X');
funCst5 = mvnpdf(X',muCst5,sigmaCst5);
fCst5 = funCst5';

%% Joint pdf, including all the constraints
p=@(X,Lambda) exp(-1+Lambda(1)+Lambda(2)*r(2).func(X)+Lambda(3)*r(3).func(X)+Lambda(4)*r(4).func(X)+Lambda(5)*fCst5);%joint PDF
%%
%Marginal pdf without exp(-1+lambda(1)) up to the fourth constraint
pk=@(x,Lambda) exp(Lambda(2)*r(2).func(x)+Lambda(3)*r(3).func(x)+Lambda(4)*r(4).func(x));
%% Estimate initial lambda and obtain the sample
%averages. These Lambdas are evaluated using only global measuring functions
[Lambda, Alpha] = maxEntropyDensityEstLEmD( X, M4, r, pk, r1, fCst5);
%%
%Evaluate Lambdas using all constraints
[Lambda, fConst] = maxEntropyDensityEstNImD( T ,K, M4, M, r, Lambda, Alpha, constType, p, X);

%% Plots
%Plot 'Histogram of Generated Data'

figure(2) 
hist3(X','Nbins',[90 90],'CDataMode','auto','FaceColor','interp')
axis([-4,4,-4,4])
movegui('west')
title('Histogram of generated data')

%Plot approximated pdf using all the constraints
A = normalization(r,Lambda,K);
figure(3)
movegui('east')
f = p(X,Lambda);
f=f/A;
X = X-0.8;
scatter3(X(2,:),X(1,:),f,[],f,'o','filled','MarkerEdgeColor','none')
title('Approximated PDF')
axis([-4,4,-4,4])
sum(abs(fConst))





function [Lambda, Alpha] = maxEntropyDensityEstLEmD( S, M4, r, pk, r1, fCst5)
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


function [LambdaNew, fConst] = maxEntropyDensityEstNImD( T,K, M4, M, r, Lambda, Alpha, constType, p, X)
maxIter=10;
tolerance = 1e-008;

Alpha(1)=K;
Lambda(end+1)=0; %Set Lambda_5=0 

LambdaOld=Lambda;

if constType == 1
    checkIdx=M-1;
else
    checkIdx=M-2;
end

%Init for The Quasi-Random Numbers
numpoints = 1000; % number of points
len = 4; % length 
Xn = quasirand(1,numpoints,K); %generating quasirandom numbers
XX = [Xn ; -Xn];   
Xny = [Xn(:,1) (-1)*Xn(:,2)]; 
YY = [Xny ; -Xny]; 

%GAUSSIAN KERNEL
muGK = mean(XX);
sigmaGK = cov(XX);
fGK = mvnpdf(XX,muGK,sigmaGK);
q = fGK';

muGKy = mean(YY);
sigmaGKy = cov(YY);
fGKy = mvnpdf(YY,muGKy,sigmaGKy);
qy = fGKy';
    
for iter = 1 : maxIter
    if LambdaOld(checkIdx)>0
        disp('Wrong lambda!!!');
        break;
    end
    
    iter 
    
    %Initializations for the Quasi-Monte Carlo Integration
    J=zeros(M,M); %This is for the Jacobian
    fConst=zeros(M,1); %For the E{r-alpha} Cost function
    
    %Evaluate the cost function E{r-alpha} for the global constraints by Quasi-Monte Carlo Integration
    
    for i=1:M4
            fval =r(i).func(XX').*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(XX')+LambdaOld(3)*r(3).func(XX')+LambdaOld(4)*r(4).func(XX')+LambdaOld(5)*q);
            fvaly =r(i).func(YY').*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(YY')+LambdaOld(3)*r(3).func(YY')+LambdaOld(4)*r(4).func(YY')+LambdaOld(5)*qy);
            QMCint = [sum(fval(1:numpoints)/numpoints),sum(fvaly(1:numpoints)/numpoints)];
            QMCintVal = len*sum(abs(QMCint));
            fConst(i) = (QMCintVal - Alpha(i)); 
    end
        
    %Evaluate the cost function for Gaussian Kernel
            fvalCst5 =q.*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(XX')+LambdaOld(3)*r(3).func(XX')+LambdaOld(4)*r(4).func(XX')+LambdaOld(5)*q);
            fvalCst5y =qy.*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(YY')+LambdaOld(3)*r(3).func(YY')+LambdaOld(4)*r(4).func(YY')+LambdaOld(5)*qy);
            QMCintCst5 = [sum(fvalCst5(1:numpoints)/numpoints),sum(fvalCst5y(1:numpoints)/numpoints)];
            QMCintValCst5 = len*sum(abs(QMCintCst5)); 
             
            fConst(5) = QMCintValCst5 - Alpha(5);
            
            fConstN = fConst/norm(fConst,1); % is the 1-norm 
                           
    %Evaluate the Jacobian matrix by Quasi-Monte Carlo Integration. 
	%This matrix is symmetrix positive definite. In this loop Jacobian 
	%corresponds to the global constraints
	
    for i=1:M4
        for j=i:M4
            fvalJ =r(i).func(XX').*r(j).func(XX').*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(XX')+LambdaOld(3)*r(3).func(XX')+LambdaOld(4)*r(4).func(XX')+LambdaOld(5)*q);            
            fvalJy =r(i).func(YY').*r(j).func(YY').*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(YY')+LambdaOld(3)*r(3).func(YY')+LambdaOld(4)*r(4).func(YY')+LambdaOld(5)*qy);
            QMCintJ = [sum(fvalJ(1:numpoints)/numpoints),sum(fvalJy(1:numpoints)/numpoints)];
            QMCintValJ = len*sum(abs(QMCintJ)); 
            J(i,j)= QMCintValJ;
            J(j,i)=J(i,j);
        end
    end
         
    %Last row and column of the Jacobian matrix. 
    %Corresponds to the local constraint
	
    for i=1:M4
            fvalJCst5 =r(i).func(XX').*q.*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(XX')+LambdaOld(3)*r(3).func(XX')+LambdaOld(4)*r(4).func(XX')+LambdaOld(5)*q);
            fvalJCst5y =r(i).func(YY').*qy.*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(YY')+LambdaOld(3)*r(3).func(YY')+LambdaOld(4)*r(4).func(YY')+LambdaOld(5)*qy);
            QMCintJCst5 = [sum(fvalJCst5(1:numpoints)/numpoints),sum(fvalJCst5y(1:numpoints)/numpoints)];
            QMCintValJCst5 = len*sum(abs(QMCintJCst5));
         
            J(i,5)=QMCintValJCst5;
            J(5,i)=J(i,5);
    end
    
    %Last element of the Jacobian 
	
            fvalCst5J =q.*q.*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(XX')+LambdaOld(3)*r(3).func(XX')+LambdaOld(4)*r(4).func(XX')+LambdaOld(5)*q);
            fvalCst5Jy =qy.*qy.*exp(-1+LambdaOld(1)+LambdaOld(2)*r(2).func(YY')+LambdaOld(3)*r(3).func(YY')+LambdaOld(4)*r(4).func(YY')+LambdaOld(5)*qy);
            QMCintCst5J = [sum(fvalCst5J(1:numpoints)/numpoints),sum(fvalCst5Jy(1:numpoints)/numpoints)];
            QMCintValCst5J = len*sum(abs(QMCintCst5J));
        
            J(5,5)= QMCintValCst5J;
            
    %Evaluating Lambdas
    LambdaDelta=J\fConstN;
    
    beta=1;
    if LambdaOld(checkIdx)>LambdaDelta(checkIdx)
        beta=0.99*LambdaOld(checkIdx)/LambdaDelta(checkIdx);
    end
       
    LambdaNew=LambdaOld-beta*LambdaDelta;
    if abs(1-((LambdaNew'*LambdaOld))/(LambdaOld'*LambdaOld))<tolerance
        break;
    end
    

    LambdaOld=LambdaNew/sqrt(cov(LambdaNew)); %Normalized lagrange multipliers
    Cost(iter)=sum((fConstN.^2)/length(fConstN));

end
figure;
semilogy(Cost);
title('Cost function w.r.t number of iterations')

function c = normalization(r,Lambda,K)
%--------------------------------------------------------------------
%Compute the normalizing constant is used to reduce any probability  
%function to a probability density function with total probability of one.
%--------------------------------------------------------------------

% Init for The Quasi-Random Numbers
numpoints = 1000; % number of points
len = 4; % length 

Xn = quasirandnumbers(1,numpoints,K); %generating quasirandom numbers
XX = [Xn ; -Xn]; % quasirandom numbers for Quadrant(2)->[x<0 y>0] and Quadrant(4)->[x>0 y<0]    
Xny = [Xn(:,1) (-1)*Xn(:,2)]; 
YY = [Xny ; -Xny]; % quasirandom numbers for Quadrant(1)->[x>0 y>0] and Quadrant(3)->[x<0 y<0]

% GAUSSIAN KERNEL
%Quadrant(2)->[x<0 y>0] and Quadrant(4)->[x>0 y<0]
muGK = mean(XX);
sigmaGK = cov(XX);
fGK = mvnpdf(XX,muGK,sigmaGK);
q = fGK';

%Quadrant(1)->[x>0 y>0] and Quadrant(3)->[x<0 y<0]
muGKy = mean(YY);
sigmaGKy = cov(YY);
fGKy = mvnpdf(YY,muGKy,sigmaGKy);
qy = fGKy';

%% Compute for Quadrant(2)->[x<0 y>0] and Quadrant(4)->[x>0 y<0]
fval =exp(-1+Lambda(1)+Lambda(2)*r(2).func(XX')+Lambda(3)*r(3).func(XX')+Lambda(4)*r(4).func(XX')+Lambda(5)*q);
%Compute for the Quadrant(2)->[x<0 y>0] and Quadrant(4)->[x>0 y<0]
fvaly =exp(-1+Lambda(1)+Lambda(2)*r(2).func(YY')+Lambda(3)*r(3).func(YY')+Lambda(4)*r(4).func(YY')+Lambda(5)*qy);

%%
QMCint = [sum(fval(1:numpoints)/numpoints),sum(fvaly(1:numpoints)/numpoints)];
c = len*sumabs(QMCint);


function y = genMixGGDmD( paramGGD, b, a, K, T )
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