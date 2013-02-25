function [w, b] = unflatten_weights(wflat, arch, tied_w)
  if ~exist('tied_w', 'var') || isempty(tied_w)
    tied_w = false;
  end

  nlayers = length(arch)-1;

  % transform weights
  top = 0;
  w = cell(1,nlayers);
  b = cell(1,nlayers);
  for layer = 1 : nlayers
    height = arch(layer).numw; 
    width = arch(layer+1).numw;
    if ~tied_w || layer <= (nlayers) / 2
      w{layer} = reshape(wflat(top+1 : top+height*width), height, width);
      top = top + height*width;
      b{layer} =  wflat(top+1 : top+width)';
      top = top + width;
    else
      w{layer} = w{nlayers-layer+1}';
      b{layer} =  wflat(top+1 : top+width)';
      top = top + width;
    end
  end
  assert(top == length(wflat));
end
