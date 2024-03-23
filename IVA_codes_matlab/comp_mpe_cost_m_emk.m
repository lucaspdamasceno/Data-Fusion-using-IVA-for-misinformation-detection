function [cost,y, Lambda, PDF, r, fCst5]=comp_mpe_cost_m_emk(W,X,~)

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

%shapeParam = zeros(N,1);
for n=1:N
    yn=shiftdim(Y(n,:,:)).'; % K x T
    %yn = reshape(Y(n,:,:),[K,10000]);
%    if uncorr
%        gip=dot(yn,yn);
%        dcost=-K*log(rhat)+rhat*mean(sqrt(gip));
%    else
        
        [Lambda,PDF, r, fCst5] = M_EMK(yn); % updates by M-EMK. 
        
        % cost function 
        
        cost_MEMK = (-1)*mean(log(PDF)); % Cost Function with respect to Lambdas
        
%       const_log=-log(const_Ck);
%       gip=dot(yn,CovN\yn);
%       dcost_all=const_log + (K/2)*log(m) + 0.5*log(det(CovN)) + (mean(gip.^b_hat)/(2*m^(b_hat)));
%       dcost=dcost_all;
%       shapeParam(n)=b_hat;
%    end
    cost=cost+cost_MEMK;
end
y=Y;

return