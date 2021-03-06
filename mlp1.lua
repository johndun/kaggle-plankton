-- Loss on test set:         2.1458830573441
require 'torch'
require 'nn'
require './util'
require './jitter_rot_flip'
require 'cunn'

local function create_model()
  torch.manualSeed(config.model_seed)
  local model = nn.Sequential()
  model:add(nn.Reshape(INPUT_SZ * INPUT_SZ))
  model:add(nn.Linear(INPUT_SZ * INPUT_SZ, 1024))
  model:add(nn.Tanh())
  model:add(nn.Linear(1024, #CLASSES))
  model:add(nn.LogSoftMax())
  local criterion = nn.DistKLDivCriterion()
  
  model:cuda()
  criterion:cuda()
  return model, criterion
end


local cmd = torch.CmdLine()
cmd:option('-progress', false, 'show progress bars')
cmd:option('-id', 'mlp1', 'model id')
cmd:option('-s0', 1, 'model seed')
cmd:option('-s1', 1, 'seed for learning rate 1')
cmd:option('-s2', 2, 'seed for learning rate 2')
cmd:option('-s3', 3, 'seed for learning rate 3')
opt = cmd:parse(arg)

config = {
  id = opt.id or 'dummy', 
  learningRateDecay = 0.0, 
  momentum = 0.9, 
  batch_size = 32, 
  model_seed = opt.s0, 
  early_stop = 1, 
  evaluate_every = 1, 
  s3_sync = true, 
  test_jitter = false, 
  train_jitter = false
}

if not config.test_jitter then
  TEST_JITTER_SZ = 1
end

local learning_rates = {1.0, 0.1}
local seeds = {opt.s1, opt.s2}
local epochs = {1, 1}
local val_prop = 0.1
local model, criterion = create_model()
epochs = validate(model, criterion, learning_rates, seeds, epochs, val_prop)

local model, criterion = create_model()
model = train(model, criterion, learning_rates, seeds, epochs)

local test_images = get_test_image_list(TEST_DECODE_FNAME)
local model = torch.load('model/' .. config.id .. '.model')
local preds = gen_predictions(model, test_images)
write_predictions(preds, test_images)