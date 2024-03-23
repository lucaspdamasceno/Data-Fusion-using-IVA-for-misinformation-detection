function [cost,y,shapeParam]=comp_mpe_cost_FS(W,X,~,uncorr)

if nargin<5
    uncorr=false;
    if nargin<4
        const_log=[];
    end
end

[N,~,K]=size(X);
Y=X;
cost=0;

for k = 1:K
    Y(:,:,k) = W(:,:,k)*X(:,:,k);
    cost=cost-log(abs(det(W(:,:,k))));
end % K

shapeParam = zeros(N,1);
for n=1:N
    yn=shiftdim(Y(n,:,:)).'; % K x T
    if uncorr
        gip=dot(yn,yn);
        dcost=-K*log(rhat)+rhat*mean(sqrt(gip));
    else
        [CovN, b_hat,m] = fisher_scoring_complete(yn,0, 100);
        const_Ck=((1/2).^(K./(2.*b_hat))).*(b_hat.*gamma(K/2)./(pi^(K/2)*gamma(K./(2*b_hat))));
        const_log=-log(const_Ck);
        gip=dot(yn,CovN\yn);
        dcost_all=const_log + (K/2)*log(m) + 0.5*log(det(CovN)) + (mean(gip.^b_hat)/(2*m^(b_hat)));
        dcost=dcost_all;
        shapeParam(n)=b_hat;
    end
    cost=cost+dcost;
end
y=Y;



return