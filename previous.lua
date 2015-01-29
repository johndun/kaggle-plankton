test = function(model, criterion, files, labels)
  local confusion = optim.ConfusionMatrix(CLASSES)
  local loss = 0.0
  local N = #files
  local batch_size = config.batch_size or 12
  local num_batches = 0
  local inputs = torch.Tensor(batch_size, NUM_COLORS, INPUT_SZ, INPUT_SZ):cuda()
  local targets = torch.Tensor(batch_size, #CLASSES):cuda()
  local preprocess_params = torch.load(string.format(
                            'model/%s_preproc_params.t7', config.id))
  for t = 1, N, batch_size do
    if t + batch_size - 1 > N then
      batch_size = N - t + 1
    end
    if opt.progress then
      xlua.progress(t, N)
    end
    num_batches = num_batches + 1
    targets:zero()
    for i = 1, batch_size do
      local fname = files[t + i - 1]
      local label = labels[t + i - 1]
      targets[i][label] = 1
      local img = test_jitter(base_img_dir .. '/train/' .. fname)
      inputs[i]:copy(img)
    end
    inputs = preprocess(inputs, preprocess_params)
    local output = model:forward(inputs)
    local f = criterion:forward(output:narrow(1, 1, batch_size), 
                                targets:narrow(1, 1, batch_size))
    confusion:batchAdd(output:narrow(1, 1, batch_size), 
                       targets:narrow(1, 1, batch_size))
    loss = loss + f * batch_size
    collectgarbage()
  end
  if opt.progress then
    xlua.progress(N, N)
  end
  confusion:updateValids()
  loss = loss / num_batches
  return loss, confusion.totalValid
end