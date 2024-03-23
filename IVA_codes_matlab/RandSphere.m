function X=RandSphere(N,dim)
% RANDSPHERE
%
% RandSphere generates uniform random points on the surface of a unit radius
% N-dim sphere centered in the origin . This script uses differents algorithms
% according to the dimensions of points:
%
%    -2D:  random generation of theta [0 2*pi]
%    -3D:  the "trig method".
%    -nD:  Gaussian distribution
%
% SYNOPSYS:
%
% INPUT:
%
%    N: integer number representing the number of points to be generated
%    dim: dimension of points, if omitted 3D is assumed as default
%
% OUTPUT:
%
%   X: Nxdim double matrix representing the coordinates of random points
%   generated
%
% EXAMPLE:
% 
%   N=1000;
%   X=RandSphere(N);
%   hold on
%   title('RandSphere')
%   plot3(X(:,1),X(:,2),X(:,3),'.k');
%   axis equal
%
% See also radnd, randi, randn, RandStream, RandStream/rand,
%              sprand, sprandn, randperm,RandStream/getDefaultStream.
%
%Authors: Luigi Giaccari,Ed Hoyle



%errors check
if nargin<1 %nargin >2 is handled by Matlab itself
    error('At least one input required')
end
if nargout~=1
    error('One and ony one output required')
end

if N<=0
    error('Negative number of points')
end
if round(N)~=N
    warning('N should be an integer. It will be rounded')
    N=round(N);
end

if nargin==1
    dim=3;%default value for dimension of points
end

switch dim
    case 3 %3D
        
        %trig method
        X=zeros(N,dim);%preallocate
        X(:,3)=rand(N,1)*2-1;%z
        t=rand(N,1)*2*pi;
        r=sqrt(1-X(:,3).^2);
        X(:,1)=r.*cos(t);%x
        X(:,2)=r.*sin(t);%y
    case 2 %2D
        
        %just a random generation of theta
        X(:,2)=rand(N,1)*2*pi;%theta: use y as temp value
        X(:,1)=cos(X(:,2));%x
        X(:,2)=sin(X(:,2));%y
    
    otherwise %nD
        
        %use gaussian distribution
        X=randn(N,dim);
        X=bsxfun(@rdivide,X,sqrt(sum(X.^2,2)));
        
end


end