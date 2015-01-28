require 'torch'
require 'optim'
require 'xlua'
require 'cunn'

local function sgd(model, criterion, config, train_x, train_y)
  local parameters, gradParameters = model:getParameters()
  local confusion = optim.ConfusionMatrix(CLASSES)
  local loss = 0.0
  local batch_size = config.batch_size or 12
  local shuffle = torch.randperm(train_x:size(1))
  local inputs = torch.Tensor(batch_size, NUM_COLORS, INPUT_SZ, INPUT_SZ):cuda()
  local targets = torch.Tensor(batch_size, #CLASSES):cuda()
  local num_batches = 0
  for t = 1, train_x:size(1), batch_size do
    if t + batch_size - 1 > train_x:size(1) then
      break
    end
    if opt.progress then
      xlua.progress(t, train_x:size(1))
    end
    num_batches = num_batches + 1
    for i = 1, batch_size do
      local img = train_x[shuffle[t + i - 1]]
      if config.train_jitter then
        img = random_jitter(img)
      end
      inputs[i]:copy(img)
      targets[i]:copy(train_y[shuffle[t + i - 1]])
    end

    local feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      gradParameters:zero()
      local output = model:forward(inputs)
      local f = {}
      local df_do = {}
      
      if type(output) == 'table' then
        for j = 1, #criterion do
          f[j] = criterion[j]:forward(output[j], targets)
          df_do[j] = criterion[j]:backward(output[j], targets)
          if j > 1 then
            df_do[j]:mul(1 / #criterion)
          end
        end
        confusion:batchAdd(output[1], targets)
        loss = loss + batch_size * f[1]
      else
        f = criterion:forward(output, targets)
        df_do = criterion:backward(output, targets)
        confusion:batchAdd(output, targets)
        loss = loss + f
      end
      
      model:backward(inputs, df_do)
      return f, gradParameters
    end
    
    optim.sgd(feval, parameters, config)
    if num_batches % 10 == 0 then
      collectgarbage('collect')
    end
  end
  if opt.progress then
    xlua.progress(train_x:size(1), train_x:size(1))
  end
  confusion:updateValids()
  loss = batch_size * loss / num_batches
  return confusion.totalValid, loss
end

local function test(model, criterion, test_x, test_y, batch_size)
  if config.test_jitter then
    return test_augmented(model, criterion, test_x, test_y, batch_size)
  end
  
  criterion.sizeAverage = false
  
  local confusion = optim.ConfusionMatrix(CLASSES)
  local loss = 0.0
  local batch_size = batch_size or 12
  local inputs = torch.Tensor(batch_size, NUM_COLORS, INPUT_SZ, INPUT_SZ):cuda()
  local targets = torch.Tensor(batch_size, #CLASSES):cuda()
  local test_n = test_x:size(1)
  local num_batches = 0
  for t = 1, test_n, batch_size do
    if opt.progress then
      xlua.progress(t, test_n)
    end
    num_batches = num_batches + 1
    if t + batch_size - 1 > test_n then
      batch_size = test_n - t + 1
    end
    for i = 1, batch_size do
      inputs[i]:copy(test_x[t + i - 1])
      targets[i]:copy(test_y[t + i - 1])
    end

    local output = model:forward(inputs)
    if type(output) == 'table' then
      output = output[1]
    end
    f = criterion:forward(output:narrow(1, 1, batch_size), 
                         targets:narrow(1, 1, batch_size))
    confusion:batchAdd(output:narrow(1, 1, batch_size), 
                       targets:narrow(1, 1, batch_size))
    loss = loss + f
    if num_batches % 10 == 0 then
      collectgarbage('collect')
    end
  end
  if opt.progress then
    xlua.progress(test_n, test_n)
  end
  
  criterion.sizeAverage = true
  loss = loss / test_n
  
  confusion:updateValids()
  return confusion.totalValid, loss
end

local function training_loop(model, criterion, config, 
                             train_x, train_y, test_x, test_y)
  torch.manualSeed(config.train_seed)
  local parameters = model:getParameters()
  print('Number of model parameters: ' .. parameters:size(1))
  
  -- local best_test_acc = config.starting_acc or 0
  local best_test_err = config.starting_err or 1000.0
  local evaluate_every = config.evaluate_every or 1
  local best_epoch = 0
  local epochs_since_best = 0
  for epoch = 1, config.epochs do
    model:training()
    print('\nTraining epoch ' .. epoch)
    local acc, err = sgd(model, criterion, config, train_x, train_y)
    print('Final learning rate: ' .. 
          (config.learningRate / 
          (1 + config.learningRateDecay * config.evalCounter)))
    print('Accuracy on training set: ' .. acc)
    print('Loss on training set: ' .. err)
    if config.eval and epoch % evaluate_every == 0 then
      model:evaluate()
      local acc, err = test(model, criterion, test_x, test_y, config.batch_size)
      local log_str = 'Accuracy on evaluation set:                       ' .. 
        acc .. '\nLoss on evaluation set:                           ' .. err
      if err < best_test_err then
        best_epoch = epoch
        best_test_err = err
        torch.save(string.format('model/%s.model', config.id), model)
        log_str = log_str .. '*'
        epochs_since_best = 0
      else
        epochs_since_best = epochs_since_best + evaluate_every
        if config.early_stop and epochs_since_best >= config.early_stop then
          print(log_str)
          print(string.format('Stopping, no improvement in %s epochs', 
                              config.early_stop))
          break
        end
      end
      print(log_str)
    end
    collectgarbage()
  end
  
  if config.eval then
    return best_epoch, best_test_err
  else
    torch.save(string.format('model/%s.model', config.id), model)
    return model
  end
end

function validate(model, criterion, learning_rates, seeds, epochs, val_sz)
  print('### Early stopping using validation set')
  local epochs = epochs
  config.eval = true
  local train_x, train_y = unpack(torch.load(TRAIN_FNAME))
  local test_x, test_y, preprocess_params
  train_x, train_y, test_x, test_y = validation_split(train_x, train_y, val_sz)
  if not config.train_jitter then
    train_x = crop_images(train_x)
    preprocess_params = preprocess(train_x)
  else
    local tmp = crop_images(train_x)
    preprocess_params = preprocess(tmp)
    preprocess(train_x, preprocess_params)
  end
  if not config.test_jitter then
    test_x = crop_images(test_x)
  end
  preprocess(test_x, preprocess_params)
  
  for i = 1, #learning_rates do
    print(string.format('\n### Training at learning rate %s: %s', 
                        i, learning_rates[i]))
    config.learningRate = learning_rates[i]
    config.train_seed = seeds[i]
    config.epochs = epochs[i]
    config.evalCounter = nil
    epochs[i], config.starting_loss = training_loop(model, criterion, config, 
                                                   train_x, train_y, 
                                                   test_x, test_y)
    model = torch.load(string.format('model/%s.model', config.id))
  end
  return epochs
end

function train(model, criterion, learning_rates, seeds, epochs)
  print('\n### Train using full training set')
  config.eval = false
  local train_x, train_y = unpack(torch.load(TRAIN_FNAME))
  local preprocess_params
  if not config.train_jitter then
    train_x = crop_images(train_x)
    preprocess_params = preprocess(train_x)
  else
    local tmp = crop_images(train_x)
    preprocess_params = preprocess(tmp)
    preprocess(train_x, preprocess_params)
  end
  for i = 1, #learning_rates do
    print(string.format('\n### Training at learning %s: %s', 
                        i, learning_rates[i]))
    config.learningRate = learning_rates[i]
    config.train_seed = seeds[i]
    config.epochs = epochs[i]
    config.evalCounter = nil
    if config.epochs > 0 then
      model = training_loop(model, criterion, config, train_x, train_y)
    end
  end
  return model
end