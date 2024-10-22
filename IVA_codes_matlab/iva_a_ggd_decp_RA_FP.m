function [W,cost,shapePass,isi,iter] = iva_a_ggd_decp_RA_FP(X,varargin)

%% Gather Options

% build default options structure
Params=struct( ...
   'whiten',true, ... % whitening is optional
   'gradProjection',true, ... % project gradient onto orthogonal direction
   'verbose',true, ... % verbose true enables print statements
   'A',[], ... % true mixing matrices A, automatically sets verbose
   'initW',[], ... % initial estimates for demixing matrices in W
   'maxIter',2*512, ... % max number of iterations
   'terminationCriterion','ChangeInW', ... % criterion for terminating iterations: ChangeInCost, (ChangeInW)
   'termThreshold',1e-6, ... % termination threshold
   'alpha0',1.0 ... % initial step size scaling
   );

% load in user supplied options
Params=getopt(Params,varargin{:});
alphaMin=Params.termThreshold; % alpha0 will max(alphaMin,alphaScale*alpha0)
alphaScale=0.9;
supplyA=~isempty(Params.A); % set to true if user has supplied true mixing matrices
outputISI=false;

%matlabpool('open','4');

%% Create cell versions
% Faster than using the more natural 3-dimensional version

[N,T,K]=size(X); % the input format insures all datasets have equal number of samples

if Params.whiten
   [X,V]=whiten(X);
end

%% Initialize W
if ~isempty(Params.initW)
   W=Params.initW;
   if size(W,3)==1 && size(W,1)==N && size(W,2)==N
      W=repmat(W,[1,1,K]);
   end
   if Params.whiten
      for k=1:K
         W(:,:,k)=W(:,:,k)/V(:,:,k);
      end
   end
else
   W=randn(N,N,K);
end

for k=1:K
   W(:,:,k)=vecnorm(W(:,:,k)')';
end

%% When mixing matrix A supplied
% verbose is set to true
% outputISI can be computed if requested
% A matrix is conditioned by V if data is whitened
if supplyA
   % only reason to supply A matrices is to display running performance
   Params.verbose=true;
   if nargout>2
      outputISI=true;
      isi=nan(1,Params.maxIter);
   end
   if Params.whiten
      for k = 1:K
         Params.A(:,:,k) = V(:,:,k)*Params.A(:,:,k);
      end
   end
end

%% Initialize some local variables
cost=nan(1,Params.maxIter);

Y=X*0;

for iter = 1:Params.maxIter
    termCriterion=0;
   
   % Current estimated sources
   for k=1:K
      Y(:,:,k)=W(:,:,k)*X(:,:,k);
   end
   
   % Some additional computations of performance via ISI when true A is supplied
   if supplyA
      [amari_avg_isi,amari_joint_isi]=bss_isi(W,Params.A);
      if outputISI
         isi(iter)=amari_joint_isi;
      end
   end
   
   W_old=W; % save current W as W_old
   [cost(iter),~,shapeHat]=comp_mpe_cost_RA_FP(W,X);
   shapePass(:,iter) = shapeHat;
   Q=0; R=0;
   
   for n=1:N
       
       [hnk,Q,R]=decouple_trick(W,n,Q,R);
%        fprintf('\n %f \n',hnk);
       yn=squeeze(Y(n,:,:))';
       
       [Ryn, b_hat, m] = RA_FP_Efficient(yn,0,100);
       invRyn=inv(Ryn);
       gipyn=dot(yn,invRyn*yn);
       for k=1:K
           % Derivative of cost function with respect to wnk
           phi=(invRyn(k,:)*yn).*gipyn.^(b_hat-1)*b_hat*(m^(-b_hat));
           
           dW = (X(:,:,k)*phi')/T - hnk(:,k)/(W(n,:,k)*hnk(:,k));
           if Params.gradProjection
               dW=vecnorm(dW - W(n,:,k)*dW*W(n,:,k)'); % non-colinear direction normalized
           end
           W(n,:,k)=vecnorm(W(n,:,k)' - Params.alpha0*dW)';          
       end % k
   end
   
   %% Calculate termination criterion
   switch lower(Params.terminationCriterion)
      case lower('ChangeInW')
         for k=1:K
            termCriterion = max(termCriterion,max(1-abs(diag(W_old(:,:,k)*W(:,:,k)'))));
            %termCriterion = max(termCriterion,norm(W(:,:,k) - W_old(:,:,k),'fro'));
         end % k
      case lower('ChangeInCost')
         if iter==1
            termCriterion=1;
         else
            termCriterion=abs(cost(iter-1)-cost(iter))/abs(cost(iter));
         end
      otherwise
         error('Unknown termination method.')
   end
   
   %% Check the termination condition
   if termCriterion < Params.termThreshold || iter == Params.maxIter
      break;
   elseif isnan(cost(iter))
      for k = 1:K
         W(:,:,k) = eye(N) + 0.1*randn(N);
      end
      if Params.verbose
         fprintf('\n W blowup, restart with new initial value.');
      end
   elseif iter>1 && cost(iter)>cost(iter-1)
      % see if this improves convergence
      Params.alpha0=max(alphaMin,alphaScale*Params.alpha0);
   end
   %% Display Iteration Information
   if Params.verbose
      if supplyA
         fprintf('\n Step %d: W change: %f, Cost: %f, Avg ISI: %f, Joint ISI: %f',  ...
            iter, termCriterion,cost(iter),amari_avg_isi,amari_joint_isi);
      else
         fprintf('\n Step %d: W change: %f, Cost: %f', iter, termCriterion,cost(iter));
      end
   end % options.verbose
end % iter


return;

function properties = getopt(properties,varargin)
%GETOPT - Process paired optional arguments as 'prop1',val1,'prop2',val2,...
%
%   getopt(properties,varargin) returns a modified properties structure,
%   given an initial properties structure, and a list of paired arguments.
%   Each argumnet pair should be of the form property_name,val where
%   property_name is the name of one of the field in properties, and val is
%   the value to be assigned to that structure field.
%
%   No validation of the values is performed.
%%
% EXAMPLE:
%   properties = struct('zoom',1.0,'aspect',1.0,'gamma',1.0,'file',[],'bg',[]);
%   properties = getopt(properties,'aspect',0.76,'file','mydata.dat')
% would return:
%   properties =
%         zoom: 1
%       aspect: 0.7600
%        gamma: 1
%         file: 'mydata.dat'
%           bg: []
%
% Typical usage in a function:
%   properties = getopt(properties,varargin{:})

% Function from
% http://mathforum.org/epigone/comp.soft-sys.matlab/sloasmirsmon/bp0ndp$crq5@cui1.lmms.lmco.com

% dgleich
% 2003-11-19
% Added ability to pass a cell array of properties

if ~isempty(varargin) && (iscell(varargin{1}))
   varargin = varargin{1};
end;

% Process the properties (optional input arguments)
prop_names = fieldnames(properties);
TargetField = [];
for ii=1:length(varargin)
   arg = varargin{ii};
   if isempty(TargetField)
      if ~ischar(arg)
         error('Property names must be character strings');
      end
      %f = find(strcmp(prop_names, arg));
      if isempty(find(strcmp(prop_names, arg),1)) %length(f) == 0
         error('%s ',['invalid property ''',arg,'''; must be one of:'],prop_names{:});
      end
      TargetField = arg;
   else
      properties.(TargetField) = arg;
      TargetField = '';
   end
end
if ~isempty(TargetField)
   error('Property names and values must be specified in pairs.');
end

function mag=vecmag(vec,varargin)
% mag=vecmag(vec)
% or
% mag=vecmag(v1,v2,...,vN)
%
% Computes the vector 2-norm or magnitude of vector. vec has size n by m
% represents m vectors of length n (i.e. m column-vectors). Routine avoids
% potential mis-use of norm built-in function. Routine is faster than
% calling sqrt(dot(vec,vec)) -- but equivalent.
if nargin==1
   mag=sqrt(sum(vec.*conj(vec)));
else
   mag=vec.*conj(vec);
   for ii=1:length(varargin)
      mag=mag+varargin{ii}.*conj(varargin{ii});
   end
   mag=sqrt(mag);
end
return

function [uvec,mag]=vecnorm(vec)
% [vec,mag]=vecnorm(vec)
% Returns the vector normalized by 2-norm or magnitude of vector.
% vec has size n by m represents m vectors of length n (i.e. m
% column-vectors).
[n,m]=size(vec);
if n==1
   disp('vecnorm operates on column vectors, input appears to have dimension of 1')
end

uvec=zeros(n,m);
mag=vecmag(vec); % returns a 1 x m row vector
for ii=1:size(vec,1)
   uvec(ii,:)=vec(ii,:)./mag;
end
% Equivalent to: uvec=vec./repmat(mag,size(vec,1),1);

% Which implementation is optimal depends on optimality criterion (memory
% vs. speed), this version uses the former criterion.
return

function [isi,isiGrp,success,G]=bss_isi(W,A,s,Nuse)
% Non-cell inputs:
% isi=bss_isi(W,A) - user provides W & A where x=A*s, y=W*x=W*A*s
% isi=bss_isi(W,A,s) - user provides W, A, & s
%
% Cell array of matrices:
% [isi,isiGrp]=bss_isi(W,A) - W & A are cell array of matrices
% [isi,isiGrp]=bss_isi(W,A,s) - W, A, & s are cell arrays
%
% 3-d Matrices:
% [isi,isiGrp]=bss_isi(W,A) - W is NxMxK and A is MxNxK
% [isi,isiGrp]=bss_isi(W,A,s) - S is NxTxK (N=#sources, M=#sensors, K=#datasets)
%
% Measure of quality of separation for blind source separation algorithms.
% W is the estimated demixing matrix and A is the true mixing matrix.  It should be noted
% that rows of the mixing matrix should be scaled by the necessary constants to have each
% source have unity variance and accordingly each row of the demixing matrix should be
% scaled such that each estimated source has unity variance.
%
% ISI is the performance index given in Complex-valued ICA using second order statisitcs
% Proceedings of the 2004 14th IEEE Signal Processing Society Workshop, 2004, 183-192
%
% Normalized performance index (Amari Index) is given in Choi, S.; Cichocki, A.; Zhang, L.
% & Amari, S. Approximate maximum likelihood source separation using the natural gradient
% Wireless Communications, 2001. (SPAWC '01). 2001 IEEE Third Workshop on Signal
% Processing Advances in, 2001, 235-238.
%
% Note that A is p x M, where p is the number of sensors and M is the number of signals
% and W is N x p, where N is the number of estimated signals.  Ideally M=N but this is not
% guaranteed.  So if N > M, the algorithm has estimated more sources than it "should", and
% if M < N the algorithm has not found all of the sources.  This meaning of this metric is
% not well defined when averaging over cases where N is changing from trial to trial or
% algorithm to algorithm.

% Some examples to consider
% isi=bss_isi(eye(n),eye(n))=0
%
% isi=bss_isi([1 0 0; 0 1 0],eye(3))=NaN
%


% Should ideally be a permutation matrix with only one non-zero entry in any row or
% column so that isi=0 is optimal.

% generalized permutation invariant flag (default=false), only used when nargin<3
gen_perm_inv_flag=false;
success=true;

Wcell=iscell(W);
if nargin<2
   Acell=false;
else
   Acell=iscell(A);
end
if ~Wcell && ~Acell
   if ndims(W)==2 && ndims(A)==2
      if nargin==2
         % isi=bss_isi(W,A) - user provides W & A
         
         % Traditional Metric, user provided W & A separately
         G=W*A;
         [N,M]=size(G);
         Gabs=abs(G);
         if gen_perm_inv_flag
            % normalization by row
            max_G=max(Gabs,[],2);
            Gabs=repmat(1./max_G,1,size(G,2)).*Gabs;
         end
      elseif nargin==3
         % Equalize energy associated with each estimated source and true
         % source.
         %
         % y=W*A*s;
         % snorm=D*s; where snorm has unit variance: D=diag(1./std(s,0,2))
         % Thus: y=W*A*inv(D)*snorm
         % ynorm=U*y; where ynorm has unit variance: U=diag(1./std(y,0,2))
         % Thus: ynorm=U*W*A*inv(D)*snorm=G*snorm and G=U*W*A*inv(D)
         
         y=W*A*s;
         D=diag(1./std(s,0,2));
         U=diag(1./std(y,0,2));
         G=U*W*A/D; % A*inv(D)
         [N,M]=size(G);
         Gabs=abs(G);
      else
         error('Not acceptable.')
      end
      
      isi=0;
      for n=1:N
         isi=isi+sum(Gabs(n,:))/max(Gabs(n,:))-1;
      end
      for m=1:M
         isi=isi+sum(Gabs(:,m))/max(Gabs(:,m))-1;
      end
      isi=isi/(2*N*(N-1));
      isiGrp=NaN;
      success=NaN;
   elseif ndims(W)==3 && ndims(A)==3
      % IVA/GroupICA/MCCA Metrics
      % For this we want to average over the K groups as well as provide the additional
      % measure of solution to local permutation ambiguity (achieved by averaging the K
      % demixing-mixing matrices and then computing the ISI of this matrix).
      [N,M,K]=size(W);
      if M~=N
         error('This more general case has not been considered here.')
      end
      L=M;
      
      isi=0;
      GabsTotal=zeros(N,M);
      G=zeros(N,M,K);
      for k=1:K
         if nargin<=2
            % Traditional Metric, user provided W & A separately
            Gk=W(:,:,k)*A(:,:,k);
            Gabs=abs(Gk);
            if gen_perm_inv_flag
               % normalization by row
               max_G=max(Gabs,[],2);
               Gabs=repmat(1./max_G,1,size(Gabs,2)).*Gabs;
            end
         else %if nargin==3
            % Equalize energy associated with each estimated source and true
            % source.
            %
            % y=W*A*s;
            % snorm=D*s; where snorm has unit variance: D=diag(1./std(s,0,2))
            % Thus: y=W*A*inv(D)*snorm
            % ynorm=U*y; where ynorm has unit variance: U=diag(1./std(y,0,2))
            % Thus: ynorm=U*W*A*inv(D)*snorm=G*snorm and G=U*W*A*inv(D)
            yk=W(:,:,k)*A(:,:,k)*s(:,:,k);
            Dk=diag(1./std(s(:,:,k),0,2));
            Uk=diag(1./std(yk,0,2));
            Gk=Uk*W(:,:,k)*A(:,:,k)/Dk;
            
            Gabs=abs(Gk);
         end
         G(:,:,k)=Gk;
         
         if nargin>=4
            Np=Nuse;
            Mp=Nuse;
            Lp=Nuse;
         else
            Np=N;
            Mp=M;
            Lp=L;
         end
         
         % determine if G is success by making sure that the location of maximum magnitude in
         % each row is unique.
         if k==1
            [zois3,colMaxG]=max(Gabs,[],2);
            if length(unique(colMaxG))~=Np
               % solution is failure in strictest sense
               success=false;
            end
         else
            [zois1,colMaxG_k]=max(Gabs,[],2);
            if ~all(colMaxG_k==colMaxG)
               % solution is failure in strictest sense
               success=false;
            end
         end
         
         GabsTotal=GabsTotal+Gabs;
         
         for n=1:Np
            isi=isi+sum(Gabs(n,:))/max(Gabs(n,:))-1;
         end
         for m=1:Mp
            isi=isi+sum(Gabs(:,m))/max(Gabs(:,m))-1;
         end
      end
      isi=isi/(2*Np*(Np-1)*K);
      
      Gabs=GabsTotal;
      if gen_perm_inv_flag
         % normalization by row
         max_G=max(Gabs,[],2);
         Gabs=repmat(1./max_G,1,size(Gabs,2)).*Gabs;
      end
      %       figure; imagesc(Gabs); colormap('bone'); colorbar
      isiGrp=0;
      for n=1:Np
         isiGrp=isiGrp+sum(Gabs(n,:))/max(Gabs(n,:))-1;
      end
      for m=1:Mp
         isiGrp=isiGrp+sum(Gabs(:,m))/max(Gabs(:,m))-1;
      end
      isiGrp=isiGrp/(2*Lp*(Lp-1));
   else
      error('Need inputs to all be of either dimension 2 or 3')
   end
elseif Wcell && Acell
   % IVA/GroupICA/MCCA Metrics
   % For this we want to average over the K groups as well as provide the additional
   % measure of solution to local permutation ambiguity (achieved by averaging the K
   % demixing-mixing matrices and then computing the ISI of this matrix).
   
   K=length(W);
   N=0; M=0;
   Nlist=zeros(K,1);
   for k=1:K
      Nlist(k)=size(W{k},1);
      N=max(size(W{k},1),N);
      M=max(size(A{k},2),M);
   end
   commonSources=false; % limits the ISI to first min(Nlist) sources
   if M~=N
      error('This more general case has not been considered here.')
   end
   L=M;
   
   % To make life easier below lets sort the datasets to have largest
   % dataset be in k=1 and smallest at k=K;
   [Nlist,isort]=sort(Nlist,'descend');
   W=W(isort);
   A=A(isort);
   if nargin > 2
      s=s(isort);
   end
   G=cell(K,1);
   isi=0;
   if commonSources
      minN=min(Nlist);
      GabsTotal=zeros(minN);
      Gcount=zeros(minN);
   else
      GabsTotal=zeros(N,M);
      Gcount=zeros(N,M);
   end
   for k=1:K
      if nargin==2
         % Traditional Metric, user provided W & A separately
         G{k}=W{k}*A{k};
         Gabs=abs(G{k});
         if gen_perm_inv_flag
            % normalization by row
            max_G=max(Gabs,[],2);
            Gabs=repmat(1./max_G,1,size(Gabs,2)).*Gabs;
         end
      elseif nargin>=3
         % Equalize energy associated with each estimated source and true
         % source.
         %
         % y=W*A*s;
         % snorm=D*s; where snorm has unit variance: D=diag(1./std(s,0,2))
         % Thus: y=W*A*inv(D)*snorm
         % ynorm=U*y; where ynorm has unit variance: U=diag(1./std(y,0,2))
         % Thus: ynorm=U*W*A*inv(D)*snorm=G*snorm and G=U*W*A*inv(D)
         yk=W{k}*A{k}*s{k};
         Dk=diag(1./std(s{k},0,2));
         Uk=diag(1./std(yk,0,2));
         G{k}=Uk*W{k}*A{k}/Dk;
         
         Gabs=abs(G{k});
      else
         error('Not acceptable.')
      end
      
      if commonSources
         Nk=minN;
         Gabs=Gabs(1:Nk,1:Nk);
      elseif nargin>=4
         commonSources=true;
         Nk=Nuse;
         minN=Nk;
      else
         Nk=Nlist(k);
      end
      
      if k==1
         [zois,colMaxG]=max(Gabs(1:Nk,1:Nk),[],2);
         if length(unique(colMaxG))~=Nk
            % solution is a failure in a strict sense
            success=false;
         end
      elseif success
         if nargin>=4
            [zois,colMaxG_k]=max(Gabs(1:Nk,1:Nk),[],2);
         else
            [zois,colMaxG_k]=max(Gabs,[],2);
         end
         if ~all(colMaxG_k==colMaxG(1:Nk))
            % solution is a failure in a strict sense
            success=false;
         end
      end
      
      if nargin>=4
         GabsTotal(1:Nk,1:Nk)=GabsTotal(1:Nk,1:Nk)+Gabs(1:Nk,1:Nk);
      else
         GabsTotal(1:Nk,1:Nk)=GabsTotal(1:Nk,1:Nk)+Gabs;
      end
      Gcount(1:Nk,1:Nk)=Gcount(1:Nk,1:Nk)+1;
      for n=1:Nk
         isi=isi+sum(Gabs(n,:))/max(Gabs(n,:))-1;
      end
      for m=1:Nk
         isi=isi+sum(Gabs(:,m))/max(Gabs(:,m))-1;
      end
      isi=isi/(2*Nk*(Nk-1));
   end
   
   if commonSources
      Gabs=GabsTotal;
   else
      Gabs=GabsTotal./Gcount;
   end
   % normalize entries into Gabs by the number of datasets
   % contribute to each entry
   
   if gen_perm_inv_flag
      % normalization by row
      max_G=max(Gabs,[],2);
      Gabs=repmat(1./max_G,1,size(Gabs,2)).*Gabs;
   end
   isiGrp=0;
   
   if commonSources
      for n=1:minN
         isiGrp=isiGrp+sum(Gabs(n,1:minN))/max(Gabs(n,1:minN))-1;
      end
      for m=1:minN
         isiGrp=isiGrp+sum(Gabs(1:minN,m))/max(Gabs(1:minN,m))-1;
      end
      isiGrp=isiGrp/(2*minN*(minN-1));
   else
      for n=1:Nk
         isiGrp=isiGrp+sum(Gabs(n,:))/max(Gabs(n,:))-1;
      end
      for m=1:Nk
         isiGrp=isiGrp+sum(Gabs(:,m))/max(Gabs(:,m))-1;
      end
      isiGrp=isiGrp/(2*L*(L-1));
   end
   
else
   % Have not handled when W is cell and A is single matrix or vice-versa.  Former makes
   % sense when you want performance of multiple algorithms for one mixing matrix, while
   % purpose of latter is unclear.
end

return

function [h,invQ,R]=decouple_trick(W,n,invQ,R)
% h=decouple_trick(W,n)
% h=decouple_trick(W,n,1)
% [h,invQ]=decouple_trick(W,n,invQ)
% [h,Q,R]=decouple_trick(W,n,Q,R)
%
% Computes the h vector for the decoupling trick [1] of the nth row of W. W
% can be K 'stacked' square matrices, i.e., W has dimensions N x N x K.
% The output vector h will be formatted as an N x K matrix.  There are many
% possible methods for computing h.  This routine provides four different
% (but of course related) methods depending on the arguments used.
%
% Method 1:
% h=decouple_trick(W,n)
% h=decouple_trick(W,n,0)
% -Both calls above will result in the same algorithm, namely the QR
% algorithm is used to compute h.
%
% Method 2:
% h=decouple_trick(W,n,~), where ~ is anything
% -Calls the projection method.
%
% Method 3:
% [h,invQ]=decouple_trick(W,n,invQ)
% -If two output arguments are specified then the recursive algorithm
% described in [2].  It is assumed that the decoupling will be performed in
% sequence see the demo subfunction for details.
% An example call sequence:
%  [h1,invQ]=decouple_trick(W,1);
%  [h2,invQ]=decouple_trick(W,2,invQ);
%
% Method 4:
% [h,Q,R]=decouple_trick(W,n,Q,R)
% -If three output arguments are specified then a recursive QR algorithm is
% used to compute h.
% An example call sequence:
%  [h1,Q,R]=decouple_trick(W,1);
%  [h2,Q,R]=decouple_trick(W,2,Q,R);
%
% See the subfunction demo_decoupling_trick for more examples.  The demo
% can be executed by calling decouple_trick with no arguments, provides a
% way to compare the speed and determine the accuracy of all four
% approaches.
%
% Note that methods 2 & 3 do not normalize h to be a unit vector.  For
% optimization this is usually not of interest.  If it is then set the
% variable boolNormalize to true.
%
% Main References:
% [1] X.-L. Li & X.-D. Zhang, "Nonorthogonal Joint Diagonalization Free of Degenerate Solution," IEEE Trans. Signal Process., 2007, 55, 1803-1814
% [2] X.-L. Li & T. Adali, "Independent component analysis by entropy bound minimization," IEEE Trans. Signal Process., 2010, 58, 5151-5164
%
% Coded by Matthew Anderson (matt dot anderson at umbc dot edu)

% Version 01 - 20120919 - Initial publication


if nargin==0
   help decouple_trick
   demo_decouple_trick
   return
end
if nargin==1
   help decouple_trick
   error('Not enough inputs -- see displayed help above.')
end
[M,N,K]=size(W);
if M~=N
   error('Assuming W is square matrix.')
end
h=zeros(N,K);

% enables an additional computation that is usually not necessary if the
% derivative is of  interest, it is only necessary so that sqrt(det(W*W'))
% = sqrt(det(Wtilde*Wtilde'))*abs(w'*h) holds.  Furthermore, it is only
% necessary when using the recursive or projection methods.
%
% a user might wish to enable the calculation by setting the quantity below
% to true
boolNormalize=false;

if nargout==3
   % use QR recursive method
   % [h,Qnew,Rnew]=decouple_trick(W,n,Qold,Rold)
   if n==1
      invQ=zeros(N,N,K);
      R=zeros(N,N-1,K);
   end
   for k=1:K
      if n==1
         Wtilde=W(2:N,:,k);
         [invQ(:,:,k),R(:,:,k)]=qr(Wtilde');
      else
         n_last=n-1;
         e_last = zeros(N-1,1);
         e_last(n_last) = 1;
         [invQ(:,:,k),R(:,:,k)]=qrupdate(invQ(:,:,k),R(:,:,k),-W(n,:,k)',e_last);
         [invQ(:,:,k),R(:,:,k)]=qrupdate(invQ(:,:,k),R(:,:,k),W(n_last,:,k)',e_last);
      end
      h(:,k)=invQ(:,end,k); % h should be orthogonal to W(nout,:,k)'
   end
elseif nargout==2
   % use recursive method
   % [h,invQ]=decouple_trick(W,n,invQ), for any value of n=1, ..., N
   % [h,invQ]=decouple_trick(W,1), when n=1
   
   if n==1
      invQ=zeros(N-1,N-1,K);
   end
   % Implement a faster approach to calculating h.
   for k=1:K
      if n==1
         Wtilde=W(2:N,:,k);
         invQ(:,:,k)=inv(Wtilde*Wtilde');
      else
         if nargin<3
            help decouple_trick
            error('Need to supply invQ for recursive approach.')
         end
         [Mq,Nq,Kq]=size(invQ);
         if Mq~=(N-1) || Nq~=(N-1) || Kq~=K
            help decouple_trick
            error('Input invQ does not have the expected dimensions.')
         end
         n_last=n-1;
         Wtilde_last=W([(1:n_last-1) (n_last+1:N)],:,k);
         w_last=W(n_last,:,k)';
         w_current=W(n,:,k)';
         c = Wtilde_last*(w_last - w_current);
         c(n_last) = 0.5*( w_last'*w_last - w_current'*w_current );
         %e_last = zeros(N-1,1);
         %e_last(n_last) = 1;
         temp1 = invQ(:,:,k)*c;
         temp2 = invQ(:,n_last,k);
         inv_Q_plus = invQ(:,:,k) - temp1*temp2'/(1+temp1(n_last));
         
         temp1 = inv_Q_plus'*c;
         temp2 = inv_Q_plus(:,n_last);
         invQ(:,:,k) = inv_Q_plus - temp2*temp1'/(1+c'*temp2);
         % inv_Q is Hermitian
         invQ(:,:,k) = (invQ(:,:,k)+invQ(:,:,k)')/2;
      end
      
      temp1 = randn(N, 1);
      Wtilde = W([(1:n-1) (n+1:N)],:,k);
      h(:,k) = temp1 - Wtilde'*invQ(:,:,k)*Wtilde*temp1;
   end
   if boolNormalize
      h=vecnorm(h);
   end
elseif nargin==2 || invQ==0
   % use (default) QR approach
   % h=decouple_trick(W,n)
   % h=decouple_trick(W,n,0)
   for k=1:K
      [Q,zois]=qr(W([(1:n-1) (n+1:N)],:,k)');
      h(:,k)=Q(:,end); % h should be orthogonal to W(nout,:,k)'
   end
else % use projection method
   % h=decouple_trick(W,n,~), ~ is anything
   for k=1:K
      temp1 = randn(N, 1);
      Wtilde = W([(1:n-1) (n+1:N)],:,k);
      h(:,k) = temp1 - Wtilde'*((Wtilde*Wtilde')\Wtilde)*temp1;
   end
   if boolNormalize
      h=vecnorm(h);
   end
end

return

function [z,V,U]=whiten(x)
% [z,V,U]=whiten(x)
%
% Whitens the data vector so that E{zz'}=I, where z=V*x.

if ~iscell(x)
   [N,T,K]=size(x);
   if K==1
      % Step 1. Center the data.
      x=bsxfun(@minus,x,mean(x,2));
      
      % Step 2. Form MLE of data covariance.
      covar=x*x'/T;
      
      % Step 3. Eigen decomposition of covariance.
      [eigvec, eigval] = eig (covar);
      
      % Step 4. Forming whitening transformation.
      V=sqrt(eigval) \ eigvec';
      U=eigvec * sqrt(eigval);
      
      % Step 5. Form whitened data
      z=V*x;
   else
      K=size(x,3);
      z=zeros(N,T,K);
      V=zeros(N,N,K);
      U=zeros(N,N,K);
      for k=1:K
         % Step 1. Center the data.
         xk=bsxfun(@minus,x(:,:,k),mean(x(:,:,k),2));
         
         % Step 2. Form MLE of data covariance.
         covar=xk*xk'/T;
         
         % Step 3. Eigen decomposition of covariance.
         [eigvec, eigval] = eig (covar);
         
         % Step 4. Forming whitening transformation.
         V(:,:,k)=sqrt(eigval) \ eigvec';
         U(:,:,k)=eigvec * sqrt(eigval);
         
         % Step 5. Form whitened data
         z(:,:,k)=V(:,:,k)*xk;
      end % k
   end % K>1
else % x is cell
   K=numel(x);
   sizex=size(x);
   V=cell(sizex);
   U=cell(sizex);
   z=cell(sizex);
   for k=1:K
      T=size(x{k},2);
      % Step 1. Center the data.
      xk=bsxfun(@minus,x{k},mean(x{k},2));
      
      % Step 2. Form MLE of data covariance.
      covar=xk*xk'/T;
      
      % Step 3. Eigen decomposition of covariance.
      [eigvec, eigval] = eig (covar);
      
      % Step 4. Forming whitening transformation.
      V{k}=sqrt(eigval) \ eigvec';
      U{k}=eigvec * sqrt(eigval);
      
      % Step 5. Form whitened data
      z{k}=V{k}*xk;
   end % k
end

%%
return