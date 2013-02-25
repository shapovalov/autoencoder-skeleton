function wflat = flatten_weights(w, b)
  if ~exist('b', 'var') || isempty(b)
    b = cell(1,length(w));
  end

  nlayers = length(w);
  assert(nlayers == length(b));
  
  wflat = [];
  for i = 1:nlayers, 
    wflat = [wflat; w{i}(:); b{i}(:)];  %#ok<AGROW> this is not called often
  end
end
