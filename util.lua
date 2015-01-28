require 'torch'
require 'image'
require 'optim'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

local train_decode_fname = 'data/train_decode.csv'
local base_img_dir = 'data'
local n_labels = 121
CLASSES = {}
for i = 1, n_labels do
  CLASSES[i] = i
end
NUM_COLORS = 1
INPUT_SZ = 48

string.split_it = function(str, sep)
  if str == nil then 
    return nil 
  end
  return string.gmatch(str, '[^\\' .. sep .. ']+')
end

string.split = function(str, sep)
  local ret = {}
  for seg in string.split_it(str, sep) do
    ret[#ret+1] = seg
  end
  return ret
end

load_meta_data = function()
  local file = io.open(train_decode_fname, 'r')
  local skip_head = true
  local n_samples = 1
  for line in file:lines() do
    if not skip_head then
      n_samples = n_samples + 1
    end
    skip_head = false
  end

  local file = io.open(train_decode_fname, 'r')
  local skip_head = true
  local line
  local j = 0
  local files = {}
  local labels = {}
  for line in file:lines() do
    if not skip_head then
      j = j + 1
      local row_str = string.split(line, ',')
      labels[j] = row_str[1]
      files[j] = row_str[2]
    end
    skip_head = false
  end
  return files, labels
end

prepare_val_meta = function(val_prop)
  local files, labels = load_meta_data()
  
  local train_files = {}
  local test_files = {}
  local train_labels = {}
  local test_labels = {}
  local cutoff = math.floor((1 - val_prop) * #files)
  
  for i = 1, #files do
    if i <= cutoff then
      table.insert(train_files, files[i])
      table.insert(train_labels, labels[i])
    else
      table.insert(test_files, files[i])
      table.insert(test_labels, labels[i])
    end
  end
  
  return train_files, test_files, train_labels, test_labels
end

calculate_preproc_params = function(train_files, id)
  print('### Calculating parameters for global contrast normalization')
  local id = id or config.id
  local params = torch.Tensor(#train_files, 2)
  for i = 1, #train_files do
    local img = image.loadJPG(base_img_dir .. '/train/' .. train_files[i])
    img = image.scale(img, INPUT_SZ, INPUT_SZ, 'bilinear')
    img:add(-1):mul(-1):abs()
    params[i][1] = img:mean()
    params[i][2] = img:std()^2
  end
  params = params:mean(1)[1]
  params = {mn = params[1], sd = math.sqrt(params[2])}
  torch.save(string.format('model/%s_preproc_params.t7', id), params)
end

preprocess = function(x, params)
  x:add(-1):mul(-1):abs()
  x:add(-params['mn'])
  x:mul(1/params['sd'])
  return x
end

sgd = function(model, criterion, files, labels)
  local parameters, gradParameters = model:getParameters()
  local confusion = optim.ConfusionMatrix(CLASSES)
  local loss = 0.0
  local N = #files
  local shuffle = torch.randperm(N)
  local batch_size = config.batch_size or 12
  local num_batches = 0
  local inputs = torch.Tensor(batch_size, NUM_COLORS, INPUT_SZ, INPUT_SZ):cuda()
  local targets = torch.Tensor(batch_size, #CLASSES):cuda()
  local preprocess_params = torch.load(string.format(
                            'model/%s_preproc_params.t7', config.id))
  for t = 1, N, batch_size do
    if t + batch_size - 1 > N then
      break
    end
    if opt.progress then
      xlua.progress(t, N)
    end
    num_batches = num_batches + 1
    targets:zero()
    for i = 1, batch_size do
      local fname = files[shuffle[t + i - 1]]
      local label = labels[shuffle[t + i - 1]]
      targets[i][label] = 1
      local img = image.loadJPG(base_img_dir .. '/train/' .. fname)
      img = image.scale(img, INPUT_SZ, INPUT_SZ, 'bilinear')
      inputs[i]:copy(img)
    end
    inputs = preprocess(inputs, preprocess_params)
    
    local feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      gradParameters:zero()
      local output = model:forward(inputs)
      local f = criterion:forward(output, targets)
      local df_do = criterion:backward(output, targets)
      confusion:batchAdd(output, targets)
      model:backward(inputs, df_do)
      loss = loss + f * batch_size
      return f, gradParameters
    end
    
    optim.sgd(feval, parameters, config)
    collectgarbage()
  end
  if opt.progress then
    xlua.progress(N, N)
  end
  confusion:updateValids()
  loss = loss / num_batches
  return loss, confusion.totalValid
end

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
      local img = image.loadJPG(base_img_dir .. '/train/' .. fname)
      img = image.scale(img, INPUT_SZ, INPUT_SZ, 'bilinear')
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

training_loop = function(model, criterion, 
                         train_files, train_labels, 
                         test_files,  test_labels)
  torch.manualSeed(config.train_seed)
  local parameters = model:getParameters()
  print('### Number of model parameters: ' .. parameters:size(1))
  local best_test_loss = config.starting_loss or 1000.
  local evaluate_every = config.evaluate_every or 1
  local best_epoch = 0
  local epochs_since_best = 0
  for epoch = 1, config.epochs do
    model:training()
    print('\nTraining epoch ' .. epoch)
    local loss, acc = sgd(model, criterion, train_files, train_labels)
    print('Final learning rate: ' .. 
          (config.learningRate / 
          (1 + config.learningRateDecay * config.evalCounter)))
    print('Accuracy on training set: ' .. acc)
    print('Loss on training set:     ' .. loss)
    if config.eval and epoch % evaluate_every == 0 then
      model:evaluate()
      local loss, acc = test(model, criterion, test_files, test_labels)
      print('Accuracy on test set:     ' .. acc)
      print('Loss on test set:         ' .. loss)
      if loss < best_test_loss then
        best_epoch = epoch
        best_test_loss = loss
        torch.save(string.format('model/%s.model', config.id), model)
        print('Saving new model')
        epochs_since_best = 0
      else
        epochs_since_best = epochs_since_best + evaluate_every
        if config.early_stop and 
           epochs_since_best >= config.early_stop then
          print(string.format('Stopping, no improvement in %s epochs', 
                              config.early_stop))
          break
        end
      end
    end
  end
  if config.eval then
    return best_epoch, best_test_loss
  else
    torch.save(string.format('model/%s.model', config.id), model)
    return model
  end
end

validate = function(model, criterion, learning_rates, seeds, epochs, val_prop)
  print('### Early stopping using validation set')
  local epochs = epochs
  config.eval = true
  local train_files, test_files, train_labels, test_labels = 
        prepare_val_meta(val_prop)
  calculate_preproc_params(train_files, config.id .. '_val') -- eventually will need to account for the effect of jittering
  for i = 1, #learning_rates do
    print(string.format('\n### Training at learning rate %s: %s', 
                        i, learning_rates[i]))
    config.learningRate = learning_rates[i]
    config.train_seed = seeds[i]
    config.epochs = epochs[i]
    config.evalCounter = nil
    epochs[i], config.starting_loss = training_loop(model, criterion,  
                                                    train_files, train_labels, 
                                                    test_files,  test_labels)
    model = torch.load(string.format('model/%s.model', config.id .. '_val'))
  end
  return epochs
end

train = function(model, criterion, learning_rates, seeds, epochs)
  print('\n### Train using full training set')
  local epochs = epochs
  config.eval = false
  local train_files, train_labels = load_meta_data()
  calculate_preproc_params(train_files)
  for i = 1, #learning_rates do
    print(string.format('\n### Training at learning rate %s: %s', 
                        i, learning_rates[i]))
    config.learningRate = learning_rates[i]
    config.train_seed = seeds[i]
    config.epochs = epochs[i]
    config.evalCounter = nil
    model = training_loop(model, criterion, train_files, train_labels)
  end
  return model
end