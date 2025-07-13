function [X0,U0,S0,V0] = Initialization_random(y,A,d1,d2,r,m)
%RGD Summary of this function goes here
%   input
%  Xl = USV'
%  Zl 
% Output
% Hr(Xl+Ptl*Zl)
X0 = randn(d1, d2)/n;
X0 = X0 / norm(X0, 'fro'); % Normalize to have unit Frobenius norm
X0 = X0 * 0.01;
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

