function [ A, MSE_A ,NRE_A, TIME_A] = IiSPALM_SAGA(X,ops)
% ============== input =====================================
% X  : the data tensor
% ops: algorithm parameters
%   o 'constraint'          - Latent factor constraints
%   o 'b0'                  - Initial stepsize
%   o 'n_mb'                - Number of fibers
%   o 'max_it'              - Maximum number of iterations
%   o 'A_ini'               - Latent factor initializations
%   o 'A_gt'                - Ground truth latent factors (for MSE computation only)
%   o 'tol'                 - stopping criterion
% ============= output ========================================
% A: the estimated factors
% MSE_A : the MSE of A at different iterations
% NRE_A : the cost function at different iterations
% TIME_A: the walltime at different iterations
%% Code
% Get the algorithm parameters
A       = ops.A_ini; 
b0      = ops.b0_f;
max_it  = ops.max_it;
A_gt    = ops.A_gt;
tol     = ops.tol;
out_iter = ops.out_iter;
pp1 = ops.pp1;
pp2 = ops.pp2;
pp3=ops.pp3;
dd=ops.dd;
% Get initial parametrs
dim = length(size(X));
n_mb    = ops.n_mb;%/dim;
dim_vec = size(X);
PP = tensor(ktensor(A)); 
XX = tensor(X);
err_e = 0.5*norm(XX(:) - PP(:),2)^2;
NRE_A(1) = (err_e);
MSE_A(1)=0;
for i=1:dim
    [row,col] = size(A{i});
    grad_book{i} = zeros(row,col);
    avg{i} = grad_book{i}/n_mb;
end
for dim_i=1:dim
    MSE_A(1)=MSE_A(1)+MSE_measure(A{dim_i},A_gt{dim_i});
end
MSE_A(1) = (1/dim)*MSE_A(1);
mmm = 1;
a=tic;
TIME_A(1)=toc(a);
A_old = A;
A_new = A;
A_t = A;
A_h = A;
A_b = A;
%tA = A;
% Run the algorithm until the stopping criterion
for it = 1:max_it 
    % inertial 
    inertial_1 = pp1*(it-1)/(it+2);
    inertial_2 = pp2*(it-1)/(it+2);
    inertial_3 = pp3*(it-1)/(it+2);
    %randomly permute the dimensions
    %block_vec = randperm(dim); 
    % select the block variable to update.
    for d_update=1:dim
        % sampling fibers and forming the X_{d}=H_{d} A_{d}^t least squares
        [tensor_idx, factor_idx] = sample_fibers(n_mb, dim_vec, d_update);
        % reshape the tensor from the selected samples
        X_sample = reshape(X(tensor_idx), dim_vec(d_update), [])';
        % perform a sampled khatrirao product 
        A_t{d_update} = A{d_update}+inertial_1*(A{d_update}-A_old{d_update});
        A_h{d_update} = A{d_update}+inertial_2*(A{d_update}-A_old{d_update});
        ii=1;
        for i=[1:d_update-1,d_update+1:dim]
            A_unsel{ii}= A_b{i};
             ii=ii+1;
        end
        H{d_update} = sampled_kr(A_unsel, factor_idx);
        grad{d_update} = ((A_h{d_update}*H{d_update}'-X_sample')*H{d_update})/n_mb/dd;
        grad_diff{d_update} = grad{d_update}- grad_book{d_update};
        % update the selected block
        if it>1
            A_new{d_update} = proxr(A_t{d_update}- alpha*(grad_diff{d_update}+avg{d_update}), ops, alpha, d_update);
        else 
            A_new{d_update} = proxr(A_t{d_update} - alpha*grad{d_update}, ops, alpha, d_update);
        end
        A_b{d_update}= A_new{d_update}+inertial_3*(A_new{d_update}-A{d_update});
        %A_b{d_update}(A_b{d_update}<0)=0;
        avg{d_update} = avg{d_update}+grad_diff{d_update};
        grad_book{d_update} = grad{d_update};
    end
     A_old = A;
     A = A_new;
    % compute MSE after each MTTKRP
    if mod(it,out_iter)==0
        TIME_A(mmm+1)= TIME_A(mmm)+toc(a);
        MSE_A(mmm+1)=0;
        for dim_i=1:dim
            MSE_A(mmm+1)=MSE_A(mmm+1)+MSE_measure(A{dim_i},A_gt{dim_i});
        end
        MSE_A(mmm+1)=(1/dim)*MSE_A(mmm+1);
        P = ktensor(A);
        PP = tensor(P);
        NRE_A(mmm+1) = 0.5*norm(XX(:) - PP(:))^2;
        %NRE_A(mmm+1) = NRE_obj(XX,PP,A,ops);
    
        if abs(NRE_A(mmm+1))<=tol
            break;
        end
        
        disp(['IiSPALM-SAGA at iteration ',num2str(mmm+1),' and the MSE is ',num2str(MSE_A(mmm+1))])
        disp(['IiSPALM-SAGA at iteration ',num2str(mmm+1),' and the NRE is ',num2str(NRE_A(mmm+1))])
        disp('====')
        mmm = mmm + 1;
        a=tic;
    end  
end   
end

