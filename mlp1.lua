require 'torch'
require 'nn'
require './lib/util'
require './lib/train'
require 'cunn'
-- require 'fbcunn'
-- FB = true

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
cmd:option('-id', 'mlp1_val', 'model id')
cmd:option('-s0', 1, 'model seed')
cmd:option('-s1', 1, 'seed for learning rate 1')
cmd:option('-s2', 2, 'seed for learning rate 2')
cmd:option('-s3', 3, 'seed for learning rate 3')
cmd:option('-s4', 4, 'seed for learning rate 4')
opt = cmd:parse(arg)

config = {
  id = opt.id or 'mlp1_val', 
  learningRateDecay = 0.0, 
  momentum = 0.9, 
  batch_size = 128, 
  model_seed = opt.s0, 
  train_jitter = true, 
  test_jitter = false, 
  early_stop = 6, 
  evaluate_every = 1
}

local learning_rates = {1.5, 0.5, 0.1}
local seeds = {opt.s1, opt.s2, opt.s3}
local epochs = {100, 50, 25}
local val_sz = 3072
local model, criterion = create_model()
epochs = validate(model, criterion, learning_rates, seeds, epochs, val_sz)

config.id = string.gsub(config.id, '_val$', '')
local model, criterion = create_model()
model = train(model, criterion, learning_rates, seeds, epochs)

-- config.id = string.gsub(config.id, '_val$', '')
-- local model = torch.load(string.format('model/%s.model', config.id))
-- gen_predictions(model)