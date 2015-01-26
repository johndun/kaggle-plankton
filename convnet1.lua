require 'torch'
require 'nn'
require './lib/util'
require './lib/train'
require 'cunn'
-- require 'fbcunn'
-- FB = true
require 'ccn2'

local function create_model()
  torch.manualSeed(config.model_seed)
  
  local model = nn.Sequential()
  model:add(nn.Transpose({1,4},{1,3},{1,2}))
  
  model:add(ccn2.SpatialConvolution(1, 32, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialConvolution(32, 32, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialMaxPooling(2, 2, 2, 2)) -- 64
  model:add(nn.Dropout(0.25))
  
  model:add(ccn2.SpatialConvolution(32, 64, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialConvolution(64, 64, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialMaxPooling(2, 2, 2, 2)) -- 32
  model:add(nn.Dropout(0.25))

  model:add(ccn2.SpatialConvolution(64, 128, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialConvolution(128, 128, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialMaxPooling(2, 2, 2, 2)) -- 16
  model:add(nn.Dropout(0.25))

  model:add(ccn2.SpatialConvolution(128, 256, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialMaxPooling(2, 2, 2, 2)) -- 8
  
  model:add(ccn2.SpatialConvolution(256, 256, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialMaxPooling(2, 2, 2, 2)) -- 4
  model:add(nn.Dropout(0.25))
  
  model:add(ccn2.SpatialConvolution(256, 512, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialConvolution(512, 512, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialMaxPooling(2, 2, 2, 2)) -- 2
  model:add(nn.Dropout(0.25))

  model:add(ccn2.SpatialConvolution(512, 1024, 2, 1, 0))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(ccn2.SpatialConvolution(1024, 1024, 1, 1, 0))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.Transpose({4,1},{4,2},{4,3}))
  
  model:add(nn.SpatialConvolutionMM(1024, #CLASSES, 1, 1))
  model:add(nn.Reshape(#CLASSES))
  model:add(nn.LogSoftMax())
  
  local criterion = nn.DistKLDivCriterion()
  return model:cuda(), criterion:cuda()
end


local cmd = torch.CmdLine()
cmd:option('-progress', false, 'show progress bars')
cmd:option('-id', 'convnet1_val', 'model id')
cmd:option('-s0', 1, 'model seed')
cmd:option('-s1', 1, 'seed for learning rate 1')
cmd:option('-s2', 2, 'seed for learning rate 2')
cmd:option('-s3', 3, 'seed for learning rate 3')
cmd:option('-s4', 4, 'seed for learning rate 4')
opt = cmd:parse(arg)

config = {
  id = opt.id or 'convnet1_val', 
  learningRateDecay = 0.0, 
  momentum = 0.9, 
  batch_size = 32, 
  model_seed = opt.s0, 
  train_jitter = true, 
  test_jitter = false, 
  early_stop = 6, 
  evaluate_every = 1
}

local learning_rates = {1.0, 0.25, 0.05}
local seeds = {opt.s1, opt.s2, opt.s3}
local epochs = {40, 10, 5}
-- local epochs = {1, 1, 1}
local val_sz = 3072
local model, criterion = create_model()
epochs = validate(model, criterion, learning_rates, seeds, epochs, val_sz)

config.id = string.gsub(config.id, '_val$', '')
local model, criterion
collectgarbage('collect')
model, criterion = create_model()
model = train(model, criterion, learning_rates, seeds, epochs)

-- config.id = string.gsub(config.id, '_val$', '')
-- local model = torch.load(string.format('model/%s.model', config.id))
-- gen_predictions(model)