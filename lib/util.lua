require 'torch'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

FB = false
TRAIN_FNAME = 'data/train.t7'
TEST_FNAME = 'data/test.t7'
CLASSES = {}
for i = 1, 121 do
  CLASSES[i] = i
end
NUM_COLORS = 1
INPUT_SZ = 128
CROP_OFFSET = 16
ZOOM_AMT = 8

function crop_images(train_x)
  return batch_sample{src       = train_x, 
                      crp_off_x = CROP_OFFSET+1, 
                      crp_off_y = CROP_OFFSET+1, 
                      crp_sz_x  = INPUT_SZ,
                      crp_sz_y  = INPUT_SZ}
end

function add_conv_layer(model, a, b, c, d, e, f)
  if FB then
    model:add(nn.SpatialConvolutionCuFFT(a, b, c, d, e, f))
    return
  end
  model:add(nn.SpatialConvolutionMM(a, b, c, d, e, f))
end

-- Global contrast normalization
function preprocess(x, params)
  local params = params or {}
  if #params == 0 then
    params['g_mn'] = x:mean()
    params['g_sd'] = x:std()
    torch.save(string.format('model/%s_preproc_params.t7', config.id), params)
  end
  x:add(-params['g_mn'])
  x:mul(1/params['g_sd'])
  return params
end

-- Samples a single image with augmentation: 
-- horizontal flipping, rotation, cropping and rescaling
function sample_image(arg)
  local src       = arg.src
  local hflip     = arg.hflip or false
  local rotate    = arg.rotate or 0
  local crp_off_x = arg.crp_off_x or 1
  local crp_off_y = arg.crp_off_y or 1
  local crp_sz_x  = arg.crp_sz_x or src:size(3)
  local crp_sz_y  = arg.crp_sz_y or src:size(2)
  local n_colors  = src:size(1) 
  local out_w     = arg.out_w or crp_sz_x
  local out_h     = arg.out_h or crp_sz_y
  
  if hflip then
    src = image.hflip(src)
  end
  if rotate ~= 0 then
    src = image.rotate(src, rotate)
  end
  src = image.crop(src, crp_off_x, crp_off_y, 
                   crp_off_x + crp_sz_x - 1, crp_off_y + crp_sz_y - 1)
  if src:size(3) ~= out_w or src:size(2) ~= out_h then
    src = image.scale(src, out_w, out_h, 'bilinear')
  end
  return src
end

-- Builds a dataset for processing by applying one sampling to  
-- each image in a dataset
function batch_sample(arg)
  local src       = arg.src
  local hflip     = arg.hflip or false
  local rotate    = arg.rotate or 0
  local crp_off_x = arg.crp_off_x or 1
  local crp_off_y = arg.crp_off_y or 1
  local crp_sz_x  = arg.crp_sz_x or src:size(3)
  local crp_sz_y  = arg.crp_sz_y or src:size(2)
  local n_colors  = src:size(2)
  local out_w     = arg.out_w or crp_sz_x
  local out_h     = arg.out_h or crp_sz_y
  
  local n_samples = src:size(1)
  local out = torch.Tensor(n_samples, n_colors, out_h, out_w)
  for i = 1, n_samples do
    local new_img = sample_image{src       = src[i], 
                                 hflip     = hflip, 
                                 rotate    = rotate, 
                                 crp_off_x = crp_off_x, 
                                 crp_off_y = crp_off_y,
                                 crp_sz_x  = crp_sz_x, 
                                 crp_sz_y  = crp_sz_y,
                                 out_w     = out_w,
                                 out_h     = out_h}
    out[i]:copy(new_img)
  end
  return out
end

-- Random scaling and cropping on a single image
function random_jitter(src)
  local out_sz = INPUT_SZ
  local start_sz = INPUT_SZ + 2 * CROP_OFFSET
  local crp_sz = math.random(out_sz - ZOOM_AMT, start_sz)
  local crp_off_x = math.random(1, start_sz - crp_sz + 1)
  local crp_off_y = math.random(1, start_sz - crp_sz + 1)
  local new_x = sample_image{src       = src, 
                             crp_off_x = crp_off_x, 
                             crp_off_y = crp_off_y, 
                             crp_sz_x  = crp_sz, 
                             crp_sz_y  = crp_sz, 
                             out_w     = out_sz, 
                             out_h     = out_sz}
  return new_x
end

function validation_split(dat_x, dat_y, val_size)
  local n = dat_x:size(1)
  local train_x = torch.Tensor(n - val_size, 
                               dat_x:size(2), 
                               dat_x:size(3), 
                               dat_x:size(4))
  local train_y = torch.Tensor(n - val_size, 
                               dat_y:size(2))
  local test_x = torch.Tensor(val_size, 
                              dat_x:size(2), 
                              dat_x:size(3), 
                              dat_x:size(4))
  local test_y = torch.Tensor(val_size, 
                              dat_y:size(2))
  train_x:copy(dat_x:narrow(1, 1, n - val_size))
  train_y:copy(dat_y:narrow(1, 1, n - val_size))
  test_x:copy(dat_x:narrow(1, n - val_size + 1, val_size))
  test_y:copy(dat_y:narrow(1, n - val_size + 1, val_size))
  return train_x, train_y, test_x, test_y
end

-- local train_x, train_y = unpack(torch.load(TRAIN_FNAME))
-- print(train_x:size())
-- train_x = train_x:narrow(1, 1, 100)
-- train_x = crop_images(train_x)
-- local preprocess_params = preprocess(train_x)
-- local image_tile = image.toDisplayTensor{input=train_x, padding=4, nrow=10}
-- image.saveJPG('img/batch-sample.jpg', image_tile)




-- local train_x = torch.load(TEST_FNAME)
-- train_x = batch_sample{src       = train_x, 
                       -- crp_off_x = 5, 
                       -- crp_off_y = 5, 
                       -- crp_sz_x  = 24,
                       -- crp_sz_y  = 24}
-- local ids = {84, 539, 641, 845, 1151, 
             -- 1960, 2252, 2585, 3530, 3734, 
             -- 3871, 4276, 4342, 4928, 6980}
-- local x = torch.Tensor(#ids, INPUT_SZ[1], INPUT_SZ[2], INPUT_SZ[3])
-- for i = 1, #ids do
  -- x[i] = train_x[ids[i]]
-- end
-- local image_tile = image.toDisplayTensor{input=x, padding=4, nrow=5}
-- image.saveJPG('img/deleteme.jpg', image_tile)

-- local train_x, train_y = unpack(torch.load(TRAIN_FNAME))
-- local image_tile = torch.Tensor(100, NUM_COLORS, INPUT_SZ, INPUT_SZ)
-- for i = 1, 100 do
  -- local x = train_x[i]
  -- local new_x = random_jitter(x)
  -- image_tile[i]:copy(new_x)
-- end
-- image_tile = image.toDisplayTensor{input=image_tile, padding=4, nrow=10}
-- image.saveJPG('img/random-jitter.jpg', image_tile)

-- local train_x = torch.load(TEST_FNAME)
-- local x = train_x[276]
-- local image_tile = test_jitter(x)
-- image_tile = image.toDisplayTensor{input=image_tile, padding=4, nrow=6}
-- image.saveJPG('img/test-jitter.jpg', image_tile)