function [wflat fw] = minimize(wflat, X, Y_gt, arch, tied_w, nepochs, ...
    nbatches, iter_callback, momentum, learning_rate, shuffle)
% wflat: column vector of flattened weights ordered by layers.
%   Within each layer W{i} flattened by columns is followed by the bias b{i}.
%   In case of tied weights (tied_w is true), the W{i} parts are empty 
%   for the second half of layers (i.e. for the layers from ceil(nlayers/2))
% X: feature matrix of size nobjects x nfeatures.
%   Its rows contain descriptions of objects.
% Y_gt: ground truth labels matrix of size nobjects x nlabels.
%   In case of the autoencoder, Y_gt == X.
% arch: a struct array that describes the architecture of the ANN of size (nlayers+1).
%   arch(i).numw: number of weights on the i-th layer
%   arch(i).actfun: activation function on the i-th layer for i >= 2
%   arch(i).dactfun: gradient of the activation function on the i-th layer. 
%     It takes the VALUE of the activation function, not the argument. i >= 2
% tied_w: flags if the weights are tied in the architecture (default=false)
% nepochs: number of training loops over the training set (default=100)
% nbatches: number of batches training set is divided to on each epoch.
%   Pass -1 if you want one instance per batch (default=-1)
% iter_callback: the function called several times per epoch to monitor
%   current training error. Takes the iteration number and inferred labels
% momentum: training momentum (damping factor) (default=0.0)
% learning_rate: the factor stap on each iteration is multiplied by (default=1.0)
% shuffle: flags if the training set is shuffled in the beginning of each
%   epoch. Useful for on-line and minibatch training (default=true)
%
%  OUTPUT:
% wflat: the updated parameters
% fw: loss function value on the returned wflat

  TEST_PERIOD = 250;
  nobjects = size(X,1);
  
  if nargin < 4
    error('Too few input aruments');
  end
  if nargin < 5
    tied_w = false; 
  end
  if nargin < 6
    nepochs = 100;  
  end
  if nargin < 7
    nbatches = -1;  
  end
  if nbatches == -1
    nbatches = nobjects;
  end
  if nargin < 8
    iter_callback = @(varargin) 0;  % by default, empty callback
  end
  if nargin < 9
    momentum = 0.0; 
  end
  if nargin < 10
    learning_rate = 1.0; 
  end
  if nargin < 11
    shuffle = true;
  end

  % the following lines are needed for manual stopping
  stopdlg = msgbox('Press OK to stop training after this batch');
  cleanupObj = onCleanup(@() cleandlg(stopdlg));  % destructor
  
  for epoch = 1:nepochs
    fprintf('Epoch %d\n', epoch);
    
    batch = 1; % temp
    % IGNORE THE FOLLOWING LOOP UNTIL ASKED TO IMPLEMENT MINI-BATCH
    % for batch = 1:nbatches
      % if mod(batch,TEST_PERIOD) == 1
        % Y = nnfunc(wflat, X, Y_gt, arch, tied_w);  
        % iter_callback(nbatches*(epoch-1) + batch, Y);  
      % end
      
      % % PUT YOUR CODE HERE

      % drawnow
      % if ~ishandle(stopdlg)
        % break
      % end
    % end
    
    [~, fw, dfX] = nnfunc(wflat, X, Y_gt, arch, tied_w);

    step_size = 0;
    % PUT YOUR CODE HERE
    % compute the decreasing step size; use the learning_rate constant
    
    % update weights
    wflat = wflat - step_size .* dfX;

    % estimate accuracy
    Y = nnfunc(wflat, X, Y_gt, arch, tied_w); 
    iter_callback(nbatches*(epoch-1) + batch, Y); 
    
    drawnow
    if ~ishandle(stopdlg)
      break
    end
  end
end

function cleandlg(stopdlg)
  if ishandle(stopdlg)
    close(stopdlg);
  end
end
  