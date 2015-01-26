require 'torch'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

local input_sz = 128
local input_pad = 16
local padded_sz = 160

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

local train_decode_fname = 'data/train_decode.csv'
local base_img_dir = '/data1/deeplearning/plankton/train'
local train_fname = 'data/train.t7'
local n_labels = 121

local file = io.open(train_decode_fname, 'r')
local skip_head = true
local n_samples = 1
for line in file:lines() do
  if not skip_head then
    n_samples = n_samples + 1
  end
  skip_head = false
end
-- local n_samples = 24

local file = io.open(train_decode_fname, 'r')
local skip_head = true
local line
local j = 0
local dat_x = torch.Tensor(n_samples, 1, input_sz, input_sz)
local dat_y = torch.Tensor(n_samples, n_labels):zero()
for line in file:lines() do
  if not skip_head then
    j = j + 1
    local row_str = string.split(line, ',')
    local y = row_str[1]
    local img = image.loadJPG(string.format('%s/%s', base_img_dir, row_str[2]))
    local img_sz = img:size()
    local img_sz_max = math.max(img_sz[2], img_sz[3])
    local new_img = torch.Tensor(1, img_sz_max, img_sz_max):fill(1)
    local offset_w = math.floor((img_sz_max - img_sz[3]) / 2)
    local offset_h = math.floor((img_sz_max - img_sz[2]) / 2)
    new_img:narrow(2, offset_h+1, img_sz[2]):narrow(3, offset_w+1, img_sz[3]):copy(img)
    new_img = image.scale(new_img, input_sz, input_sz, 'bilinear')
    dat_x[j]:copy(new_img)
    dat_y[j][y] = 1.0
  end
  skip_head = false
  if j == n_samples then
    break
  end
end

dat_x:add(-1):mul(-1):abs()

local dat_x2 = torch.Tensor(n_samples, 1, padded_sz, padded_sz):zero()
dat_x2:narrow(3, input_pad+1, input_sz):narrow(4, input_pad+1, input_sz):copy(dat_x)
torch.save(train_fname, {dat_x2, dat_y})

-- local image_tile = image.toDisplayTensor{input=dat_x2, padding=4, nrow=6}
-- image.saveJPG('deleteme2.jpg', image_tile)