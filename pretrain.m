function [w b] = pretrain(arch, data)
  nlayers = length(arch)-1;
  w = cell(1, nlayers);
  b = cell(1, nlayers);

  % this is for logging all error computations. See compute_error_and_plot function.
  mse_hist = [];
  save pretrain_err mse_hist

  for layer = 1 : nlayers/2
    arch1 = [];
	% PUT YOUR CODE HERE
	% compose arch1 of the three relevant layers
	
    numw = [arch1.numw];
    nweights = sum((numw(1:end-1)+1) .* numw(2:end)) - ...
      sum((numw(1:end-1)) .* numw(2:end)) / 2;
    
    wflat = zeros(nweights, 1);
	% PUT YOUR CODE HERE
	% initialize wflat with random gaussian noise

    nepochs = 3;
    nbatches = 1500;
    momentum = 0.97;
    iter_callback = @(iter, Y) compute_error_and_plot(iter, Y, data, layer==1);

    wflat = minimize(wflat, data, data, arch1, true, nepochs, nbatches, ...
      iter_callback, momentum);
    [w1, b1] = unflatten_weights(wflat, arch1, true);
    w{layer} = w1{1};
    w{nlayers-layer+1} = w1{2}; 
    b{layer} = b1{1};
    b{nlayers-layer+1} = b1{2}; 

    data = zeros(size(data,1), arch1(2).numw);
	% PUT YOUR CODE HERE
	% compute input data for the next layer
  end
end
