function Error_Stand = onetrial_SGD(m, r, kappa, lambda, params)
    % SGD version of onetrial_GD
    d1 = params.d1;
    d2 = params.d2;
    batch_size = floor(m/4); % You can set this as a parameter introduce randomness
    if isfield(params, 'batch_size')
        batch_size = params.batch_size;
    end
    if isfield(params, 'verbose')
        verbose = params.verbose;
    else
        verbose = 0;
    end
    if isfield(params, 'Xstar')
        Xstar = params.Xstar;
    else
        Xstar = groundtruth(d1, d2, r, kappa);
    end
    if isfield(params, 'init_flag')
        init_flag = params.init_flag;
    else
        init_flag = 1;
    end
    if isfield(params, 'T')
        T = params.T;
    else
        T = 200;
    end
    if isfield(params, 'mu')
        mu = params.mu;
    else
        mu = 0.2;
    end

    if isfield(params, 'problem_flag')
        problem_flag = params.problem_flag;
    else
        problem_flag = 0; % default to sensing
    end

    if problem_flag == 0
        % random sensing matrix %sym
        A = normrnd(0,1,m,d1*d2);
    elseif problem_flag == 1
        % phase retrieval: d1 == d2, each row of A is vec(aa^T), a is d1 Gaussian vector
        if d1 ~= d2
            error('For phase retrieval, d1 must equal d2.');
        end
        A = zeros(m, d1*d2);
        for i = 1:m
            a = normrnd(0,1,d1,1);
            A(i,:) = reshape(a*a', 1, []);
        end
    elseif problem_flag == 2
        if d1 ~= d2
            error('For symmetric sensing, d1 must equal d2.');
        end
        % sensing with symmetric pointwise Gaussian
        A = normrnd(0,1,m,d1*d2);
        % symmetrize each row to correspond to symmetric matrices
        for i = 1:m
            Ai = reshape(A(i,:), d1, d2);
            Ai = (Ai + Ai')/2;
            A(i,:) = Ai(:)';
        end
    elseif problem_flag == 3
        if d1 ~= d2
            error('For Richards, d1 must equal d2.');
        end
        A = params.A;

    else
        error('Unknown problem_flag value. Use 0 for sensing, 1 for phase retrieval.');
    end

    y = A*Xstar(:)/sqrt(m);

    % Initialization
    if init_flag == 0
        [X0,U0,S0,V0] = Initialization(y,A,d1,d2,r,m);
        Xl = X0;
        Ul = U0(:,1:r)*sqrt(S0(1:r,1:r));
    else
        [X0,U0] = Initialization_random(y,A,d1,d2,r,m);
        Ul = U0;
        Xl = X0;
    end

    Error_Stand = zeros(T,1);
    Error_Stand(1) = norm(X0-Xstar,'fro');

    % SGD loop
    for l = 2:T
        idx = randperm(m, batch_size); % Random mini-batch
        A_batch = A(idx, :);
        y_batch = y(idx);

        s = y_batch - (A_batch * Xl(:)) / sqrt(batch_size);

        Ul = Ul + mu * ( (1/sqrt(batch_size)) * reshape(A_batch' * s, [d1, d2]) * Ul - lambda * Ul );
        Xl = Ul * Ul';
        Error_Stand(l) = norm(Xl-Xstar,'fro')/norm(Xstar,'fro');
    end

    if verbose == 1
        semilogy(Error_Stand)
    end
end