require 'torch'
require 'nn'
require './util'
require './jitter2'
require 'cunn'

local function create_model()
  torch.manualSeed(config.model_seed)
  local model = nn.Sequential()
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolutionMM(1, 64, 3, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2))
  model:add(nn.Dropout(0.25))
  
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolutionMM(64, 128, 3, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolutionMM(128, 128, 3, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2))
  model:add(nn.Dropout(0.25))
  
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolutionMM(128, 256, 3, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolutionMM(256, 256, 3, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2))
  model:add(nn.Dropout(0.25))
  
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolutionMM(256, 512, 3, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolutionMM(512, 512, 3, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2))
  model:add(nn.Dropout(0.25))
  
  model:add(nn.SpatialConvolutionMM(512, 1024, 3, 3, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  
  model:add(nn.SpatialConvolutionMM(1024, 1024, 1, 1, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  
  model:add(nn.SpatialConvolutionMM(1024, #CLASSES, 1, 1, 1, 1))
  model:add(nn.Reshape(#CLASSES))
  model:add(nn.LogSoftMax())
  local criterion = nn.DistKLDivCriterion()
  
  model:cuda()
  criterion:cuda()
  return model, criterion
end


local cmd = torch.CmdLine()
cmd:option('-progress', false, 'show progress bars')
cmd:option('-id', 'convnet5', 'model id')
cmd:option('-s0', 6744, 'model seed')
cmd:option('-s1', 9482, 'seed for learning rate 1')
cmd:option('-s2', 1984, 'seed for learning rate 2')
cmd:option('-s3', 4535, 'seed for learning rate 3')
opt = cmd:parse(arg)

config = {
  id = opt.id or 'dummy', 
  learningRateDecay = 1e-6, 
  momentum = 0.9, 
  batch_size = 64, 
  model_seed = opt.s0, 
  early_stop = 6, 
  evaluate_every = 1,
  evaluate_start = 50,  
  s3_sync = true, 
  test_jitter = true, 
  train_jitter = true
}

if not config.test_jitter then
  TEST_JITTER_SZ = 1
end

local val_prop = 0.1
local model, criterion = create_model()
-- local parameters, gradParameters = model:getParameters()
-- print(parameters:size())
-- local input = torch.Tensor(config.batch_size,
                           -- NUM_COLORS, 
                           -- INPUT_SZ, 
                           -- INPUT_SZ):cuda()
-- local output = model:forward(input)
-- print(output:size())

local learning_rates = {1.0, 0.1}
local seeds = {opt.s1, opt.s2}
local epochs = {100, 50}
epochs = validate(model, criterion, learning_rates, seeds, epochs, val_prop)
print(epochs)

local model, criterion = create_model()
model = train(model, criterion, learning_rates, seeds, epochs)

local model = torch.load('model/' .. config.id .. '.model')
local preds, images = gen_predictions(model)
write_predictions(preds, images)
