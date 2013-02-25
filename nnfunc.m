function [Y f df] = nnfunc( wflat, X, Y_gt, arch, tied_w )
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
%
%  OUTPUT:
% Y: predicted labels matrix of size nobjects x nlabels.
% f: loss function (MSE) value of Y and Y_gt. 
% df: gradient of f as function of wflat. Column vector. Computed on demand.

  if nargin < 4
    error('Too few input aruments');
  end
  if nargin < 5
    tied_w = false;
  end

  nlayers = length(arch);
  nobjects = size(X,1);

  [w, b] = unflatten_weights(wflat, arch, tied_w);
  
  % compute intermediate function values
  val = cell(1,nlayers);
  val{1} = X;
  for layer = 2 : nlayers
    val{layer} = zeros(nobjects, arch(layer).numw);
    % PUT YOUR CODE HERE
    % compute val{layer}
  end
  
  Y = val{nlayers};  
  
  if nargout < 2  % user is not interested in loss
    return
  end
  
  f = 0;  % MSE
  % PUT YOUR CODE HERE
  % compute loss value
  
  % compute the gradient
  df = [];
  if nargout < 3  % user is not interested in the gradient
    return
  end

  dfdy = zeros(nobjects, arch(nlayers).numw);
  dfdo = zeros(nobjects, arch(nlayers).numw);
  % PUT YOUR CODE HERE
  % compute the gradients of loss on the top layer

  dfdw = cell(1,nlayers-1);
  dfdb = cell(1,nlayers-1);
  for layer = nlayers-1:-1:1
    dfdw{layer} = zeros(arch(layer).numw, arch(layer+1).numw);
    dfdb{layer} = zeros(1, arch(layer+1).numw);
    dfdo = zeros(nobjects, arch(layer).numw);
    % PUT YOUR CODE HERE
    % compute the gradients of loss by W, b, o
  end
  
  % if weights are tied, retain only weiths on the lower layers, empty the rest
  if tied_w
    % PUT YOUR CODE HERE
    % handle tied weights
  end
  
  df = flatten_weights(dfdw, dfdb);
  df = df(1:length(wflat)); % TEMP: REMOVE THIS LINE WHEN IMPLEMENT TIED_W
  assert(length(df) == length(wflat));
end

