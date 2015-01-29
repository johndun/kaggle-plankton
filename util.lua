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
AWS_SYNC_DIR = 's3://johndun.aws.bucket/kaggle-plankton'
TEST_DECODE_FNAME = 'data/test_images.csv'

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

get_test_image_list = function(decode_fname)
  local file = io.open(TEST_DECODE_FNAME, 'r')
  local images = {}
  for line in file:lines() do
    table.insert(images, line)
  end
  return images
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

save_and_sync = function(str, x, sync)
  local sync = sync or false
  torch.save(str, x)
  if sync then
    os.execute('aws s3 sync model ' .. AWS_SYNC_DIR)
  end
end

calculate_preproc_params = function(train_files, id)
  print('### Calculating parameters for global contrast normalization')
  local id = id or config.id
  local params = torch.Tensor(#train_files, 2)
  for i = 1, #train_files do
    local img = sample_image{fname    = base_img_dir .. '/train/' .. train_files[i], 
                             resize_x = INPUT_SZ, 
                             resize_y = INPUT_SZ}
    params[i][1] = img:mean()
    params[i][2] = img:std()^2
  end
  params = params:mean(1)[1]
  params = {mn = params[1], sd = math.sqrt(params[2])}
  save_and_sync(string.format('model/%s_preproc_params.t7', id), 
                params, config.s3_sync)
end

preprocess = function(x, params)
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
      local img = random_jitter(base_img_dir .. '/train/' .. fname)
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
      loss = loss + f * #CLASSES
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
        save_and_sync(string.format('model/%s.model', config.id), 
                      model, config.s3_sync)
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
    save_and_sync(string.format('model/%s.model', config.id), 
                  model, config.s3_sync)
    return model
  end
end

validate = function(model, criterion, learning_rates, seeds, epochs, val_prop)
  print('### Early stopping using validation set')
  local epochs = epochs
  config.eval = true
  config.id = config.id .. '_val'
  local train_files, test_files, train_labels, test_labels = 
        prepare_val_meta(val_prop)
  calculate_preproc_params(train_files, config.id) -- eventually will need to account for the effect of jittering
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
    model = torch.load(string.format('model/%s.model', config.id))
  end
  return epochs
end

train = function(model, criterion, learning_rates, seeds, epochs)
  print('\n### Train using full training set')
  local epochs = epochs
  config.eval = false
  config.id = string.gsub(config.id, '_val$', '')
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

test = function(model, criterion, files, labels)
  local confusion = optim.ConfusionMatrix(CLASSES)
  model:evaluate()
  local loss = 0.0
  local N = #files
  local batch_size = config.batch_size or 12
  local inputs = torch.Tensor(batch_size, NUM_COLORS, INPUT_SZ, INPUT_SZ):cuda()
  
  batch_size = math.floor(batch_size / TEST_JITTER_SZ)
  local targets = torch.Tensor(batch_size, #CLASSES):cuda()

  local num_batches = 0
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
      local imgs = test_jitter(base_img_dir .. '/train/' .. fname)
      inputs:narrow(1, 1 + TEST_JITTER_SZ*(i-1), TEST_JITTER_SZ):copy(imgs)
      local label = labels[t + i - 1]
      targets[i][label] = 1
    end
    inputs = preprocess(inputs, preprocess_params)
    local output = model:forward(inputs)
    local current_loss = 0
    for i = 1, batch_size do
      local preds = output:narrow(1, 
                    1 + TEST_JITTER_SZ*(i-1), 
                    TEST_JITTER_SZ):mean(1):reshape(#CLASSES)
      confusion:add(preds, targets[i])
      local f = criterion:forward(preds, targets[i])
      current_loss = current_loss + f * #CLASSES
    end
    current_loss = current_loss / batch_size
    loss = loss + current_loss
    collectgarbage()
  end
  if opt.progress then
    xlua.progress(N, N)
  end
  confusion:updateValids()
  loss = loss / num_batches
  return loss, confusion.totalValid
end

gen_predictions = function(model, images)
  print('\n### Generating test predictions')
  model:evaluate()
  local batch_size = config.batch_size or 12
  local N = #images
  local inputs = torch.Tensor(batch_size, NUM_COLORS, INPUT_SZ, INPUT_SZ):cuda()
  local preds = torch.Tensor(N, #CLASSES)
  batch_size = math.floor(batch_size / TEST_JITTER_SZ)
  local num_batches = 0
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
    for i = 1, batch_size do
      local fname = images[t + i - 1]
      local imgs = test_jitter(base_img_dir .. '/test/' .. fname)
      inputs:narrow(1, 1 + TEST_JITTER_SZ*(i-1), TEST_JITTER_SZ):copy(imgs)
    end
    inputs = preprocess(inputs, preprocess_params)
    local output = model:forward(inputs)
    for i = 1, batch_size do
      local current_preds = output:narrow(1, 
                            1 + TEST_JITTER_SZ*(i-1), 
                            TEST_JITTER_SZ):mean(1):reshape(#CLASSES)
      preds[t + i - 1]:copy(current_preds)
    end
    collectgarbage()
  end
  if opt.progress then
    xlua.progress(N, N)
  end
  preds:exp()
  return preds
end

write_predictions = function(preds, images)
  print('\n### Writing test predictions to file')
  local N = #images
  local file = io.open('data/submission_header.csv')
  local submission_header = file:read()
  local fname = 'result/' .. config.id .. '.csv'
  local file = io.open(fname, 'w')
  file:write(submission_header .. '\n')
  for i = 1, N do
    if opt.progress then
      xlua.progress(i, N)
    end
    local str = images[i]
    local line = preds[i]
    for j = 1, #CLASSES do
      str = str .. ',' .. line[j]
    end
    file:write(str .. '\n')
  end
  if opt.progress then
    xlua.progress(N, N)
  end
  file:close()
  os.execute('zip ' .. fname .. '.zip ' .. fname)
  os.execute('rm ' .. fname )
  os.execute('aws s3 sync result ' .. AWS_SYNC_DIR)
end

sample_image = function(arg)
  local src       = image.loadJPG(arg.fname):add(-1):mul(-1):abs()
  local hflip     = arg.hflip or false
  local rotate    = arg.rotate or false
  local resize_x  = arg.resize_x or src:size(3)
  local resize_y  = arg.resize_y or src:size(2)
  local pad       = arg.pad or false
  local crp_off_x = 1
  if arg.crp_off_x then
    crp_off_x = math.floor(arg.crp_off_x * resize_x) + 1
  end
  local crp_off_y = 1
  if arg.crp_off_y then
    crp_off_y = math.floor(arg.crp_off_y * resize_y) + 1
  end
  local crp_sz_x  = resize_x
  if arg.crp_sz_x then
    crp_sz_x = math.floor(arg.crp_sz_x * resize_x)
  end
  local crp_sz_y  = resize_y
  if arg.crp_sz_y then
    crp_sz_y = math.floor(arg.crp_sz_y * resize_y)
  end
  local n_colors  = src:size(1)
  local out_w     = arg.out_w or crp_sz_x
  local out_h     = arg.out_h or crp_sz_y
  
  src = image.scale(src, resize_x, resize_y, 'bilinear')
  if pad then
    local new_img = torch.Tensor(n_colors, 
                                 resize_y + 2 * pad, 
                                 resize_x + 2 * pad):zero()
    new_img:narrow(2, pad + 1, resize_y):narrow(3, pad + 1, resize_x):copy(src)
    src = new_img
  end
  if hflip then
    src = image.hflip(src)
  end
  if rotate then
    src = image.rotate(src, rotate)
  end
  src = image.crop(src, crp_off_x, crp_off_y, 
                   crp_off_x + crp_sz_x - 1, crp_off_y + crp_sz_y - 1)
  if src:size(3) ~= out_w or src:size(2) ~= out_h then
    src = image.scale(src, out_w, out_h, 'bilinear')
  end
  return src
end