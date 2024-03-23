function [isi,isiGrp,success,G,Gabs]=bss_isi(W,A,s,Nuse)
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
            [~,colMaxG]=max(Gabs,[],2);
            if length(unique(colMaxG))~=Np
               % solution is failure in strictest sense
               success=false;
            end
         else
            [~,colMaxG_k]=max(Gabs,[],2);
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