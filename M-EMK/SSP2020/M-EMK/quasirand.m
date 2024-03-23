function z = quasirand(n1,n2,d);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function z = quasirand(n1,n2,d);
% This function generates quasirandom numbers n1 through n2 
% from the Van der Corput sequence in R^d.
% The value of d must be at most 90 (since only the first
% 90 prime numbers are stored).
% On return, the vector z has dimension (n2-n1+1) x d.
% quasirand.m Dianne P. O'Leary 09/2004
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check that we have enough prime numbers and initialize them.
% (Note that this is more efficient than calling "primes"
% each time.)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if d > 90
  disp('Dimension too high for quasirand')
  z = 0;
  return
end

prime = [
      2      3      5      7     11     13     17     19     23     29  ...
     31     37     41     43     47     53     59     61     67     71  ...
     73     79     83     89     97    101    103    107    109    113  ...
    127    131    137    139    149    151    157    163    167    173  ...
    179    181    191    193    197    199    211    223    227    229  ...
    233    239    241    251    257    263    269    271    277    281  ...
    283    293    307    311    313    317    331    337    347    349  ...
    353    359    367    373    379    383    389    397    401    409  ...
    419    421    431    433    439    443    449    457    461    463 ];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For efficiency, initialize the vector z so that storage 
% is allocated all at once.
% Also, note that the algorithm is column oriented.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

z = zeros(n2-n1+1,d);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Work with each dimension i in turn.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:d,
   ind = 0;
   primen = prime(i);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The variable nfactor stores the representation of nn
%   in the base  prime(i).
% (The Matlab function dec2base could also be used.)
% We initialize nn = n1 and determine nfactor for it.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   nn = n1;
   k = 0;
   logn2 = ceil(log(n2)/log(primen))+1;
   nfactor = zeros(1,logn2);
   while (nn >= primen)
      k = k + 1;
      nfactor(k) = rem(nn,primen);
      nn = floor(nn/primen);
   end
   k = k + 1;
   nfactor(k) = nn;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we set the quasirandom number to the sum of
%   nfactor(j) times prime(i)^{-j}.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   primepow(1) = 1/primen;
   for ii = 2:logn2
      primepow(ii,1) = primepow(ii-1,1)/primen;
   end
   ind = ind + 1;
   z(ind,i) = nfactor(1:logn2)*primepow(1:logn2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The variable nn is now varied between  n1+1  and  n2.
% To pass from one value of nn to the next, we just add
%   1 to the first entry of nfactor, handling any carries
%   that occur when the number exceeds prime(i) in the
%   while loop "while (nfactor(kk) == primen)".
% The quasirandom number is determined in the last line
%   of the "for nn" loop.  This last line is not
%   as efficient as it might be; the relation between
%   successive quasirandom numbers could be exploited
%   here, too, but round-off errors could build up
%   if that is done.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   for nn=n1+1:n2,
      kk = 1;
      nfactor(kk) = nfactor(kk) + 1;
      while (nfactor(kk) == primen)
         nfactor(kk) = 0;
         kk = kk + 1;
         nfactor(kk) = nfactor(kk) + 1;
      end
      ind = ind + 1;
      z(ind,i) = nfactor(1:logn2)*primepow(1:logn2);
   end
end
