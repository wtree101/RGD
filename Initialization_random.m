function [X0,U0] = Initialization_random(y,A,d1,d2,r,m)
%RGD Summary of this function goes here
%   input
%  Xl = USV'
%  Zl 
% Output
% Hr(Xl+Ptl*Zl)
U0 = randn(d1, r);
U0 = U0 / norm(U0, 'fro'); % Normalize to have unit Frobenius norm
U0 = U0 * 0.1;
X0 = U0 * U0';
% for i = 1:m
%     X0=X0+y(i)*reshape(A(i,:),[d1,d2]);
% end

% X0 = X0 + reshape(A' * y, [d1, d2])/sqrt(m);

% [U0,S0,V0] = svd(X0);
% Ul = U0(:,1:r);
% Sl = S0(1:r,1:r);
% Vl = V0(:,1:r);

% X0 = Ul*Sl*Vl';   % initialization one step hard threhold

end

