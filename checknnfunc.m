function [] = checknnfunc()

	%% Setup random data / small model
	inputSize = 4;
	firstHiddenSize = 5;
	hiddenSize = 6;
	numCasses = 7;
	data   = randn(numCasses, inputSize);


	arch1.actfun = @(X) 1 ./ (1 + exp(-X));   % activation function f(X)
	arch1.dactfun = @(Y) Y .* (1 - Y); % its derivative AS A FUNCTION OF Y, Y=f(X)

	clear arch
	arch1.numw = 0;
	arch(1:5) = arch1;
	[arch.numw] = deal(inputSize, firstHiddenSize, hiddenSize, firstHiddenSize, inputSize);


	w = cell(1, 4);
	b = cell(1, 4);
	w{1} = 0.1 * randn(inputSize, firstHiddenSize);
	b{1} = zeros(1, firstHiddenSize);
	w{2} = 0.1 * randn(firstHiddenSize, hiddenSize);
	b{2} = zeros(1, hiddenSize);
	w{3} = 0.1 * randn(hiddenSize, firstHiddenSize);
	b{3} = zeros(1, firstHiddenSize);
	w{4} = 0.1 * randn(firstHiddenSize, inputSize);
	b{4} = zeros(1, inputSize);
	wflat = flatten_weights(w, b);

	[~, cost, grad] = nnfunc(wflat, data, data, arch, false);

	% Check that the numerical and analytic gradients are the same
	function value = J(x, tied_wights)
		[~, value] = nnfunc(x, data, data, arch, tied_wights);
	end
	numgrad = computeNumericalGradient(@(x) J(x, false), ...
	                                        wflat)';

	% Use this to visually compare the gradients side by side
	disp([numgrad grad]); 

	% Compare numerically computed gradients with the ones obtained from backpropagation
	disp('Norm between numerical and analytical gradient');
	diff = norm(numgrad-grad)/norm(numgrad+grad);
	disp(diff);


	numw = [arch.numw];
	% weights are tied, so W matrices are stored for only the first half of layers
	nweights = sum((numw(1:end-1)+1) .* numw(2:end)) - ...
		  sum((numw(1:end-1)) .* numw(2:end)) / 2;
	wflat = 0.1 * randn(nweights, 1);
	[~, cost, grad] = nnfunc(wflat, data, data, arch, true);

	% Check that the numerical and analytic gradients are the same
	numgrad = computeNumericalGradient(@(x) J(x, true), ...
	                                        wflat)';

	% Use this to visually compare the gradients side by side
	disp([numgrad grad]); 

	% Compare numerically computed gradients with the ones obtained from backpropagation
	disp('Norm between numerical and analytical gradient in tied weights case');
	diff = norm(numgrad-grad)/norm(numgrad+grad);
	disp(diff);
end
            
            
