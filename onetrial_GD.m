function  err = onetrial_GD(m,d1,d2,r,Xstar,init_flag,verbose,mu)
    %todo: add lambda
if nargin < 7
    verbose = 0; % Set default value for 'verbose' if not provided
end
% groundtruth
if nargin < 8
    mu = 0.9;
end

% random sensing matrix
A = normrnd(0,1,m,d1*d2);
y = A*Xstar(:)/sqrt(m);
% 
if init_flag == 0
[X0,U0,S0,V0] = Initialization(y,A,d1,d2,r,m);
Xl = X0;
Ul = U0(:,1:r)*sqrt(S0(1:r,1:r));
else
    [~,U0] = Initialization_random(y,A,d1,d2,r,m);
end

T = 200+1;



% Error Tracking
Error_Stand = zeros(T,1);
Error_Stand(1) = norm(X0-Xstar,'fro');
%disp(norm(X0,'fro'));
s = zeros(m,1);

% standard RGD
for l = 2:T
    % compute Gl
    
   
    s = y - (A * Xl(:)) / sqrt(m);
  
    Ul = Ul + mu* (1*(1/sqrt(m)) * reshape(A' * s, [d1, d2]))*Ul;
    
    % Track Errors
    Error_Stand(l) = norm(Xl_new-Xstar,'fro');
    % Swap 
    Xl = Ul*Ul';
end


%test
    if (verbose==1)
        semilogy(Error_Stand)
    end
    %Is_success = (norm(Xl - Xstar) < 1e-2);
    
    err = norm(Xl - Xstar, 'fro') / norm(Xstar, 'fro');
    

end

