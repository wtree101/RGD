function err = onetrial(m,d1,d2,r,Xstar,init_f,verbose)
if nargin < 7
    verbose = 0; % Set default value for 'verbose' if not provided
end
% groundtruth

% random sensing matrix
A = normrnd(0,1,m,d1*d2);
y = A*Xstar(:)/sqrt(m);
% 
[X0,U0,S0,V0] = init_f(y,A,d1,d2,r,m);

T = 100+1; %100 iter should be enough (and so quicker)

Xl = X0;
Ul = U0(:,1:r);
Sl = S0(1:r,1:r);
Vl = V0(:,1:r);
% Error Tracking
Error_Stand = zeros(T,1);
Error_Stand(1) = norm(X0-Xstar,'fro');
%disp(norm(X0,'fro'));

% standard RGD
for l = 2:T
    % compute Gl

    % for i = 1:m
    %     s(i)= y(i)-A(i,:)*Xl(:)/sqrt(m);
    % end
    s = y - (A * Xl(:)) / sqrt(m);
    Gl = zeros(d1,d2);
    % for i = 1:m
    %     Gl = Gl+(1/sqrt(m))*s(i)*reshape(A(i,:),[d1,d2]);
    % end
    Gl = Gl + 1*(1/sqrt(m)) * reshape(A' * s, [d1, d2]);
    
    % RGD
    [Xl_new,Ul_new,Sl_new,Vl_new] = RGD(Ul,Sl,Vl,Gl,r);
    
    % Track Errors
    Error_Stand(l) = norm(Xl_new-Xstar,'fro');
    if Error_Stand(l) > 1e3 %break
        Is_success = 0;
        return;
    end
    % Swap 
    Xl = Xl_new;
    Ul = Ul_new;
    Sl = Sl_new;
    Vl = Vl_new;
end


%test
    if (verbose==1)
        semilogy(Error_Stand)
    end
    err = norm(Xl - Xstar, 'fro') / norm(Xstar, 'fro');


end

