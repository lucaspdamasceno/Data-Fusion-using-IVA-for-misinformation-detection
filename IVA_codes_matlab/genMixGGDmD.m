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