require 'torch'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)
INPUT_SZ = {1, 48, 48}

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

local train_decode_fname = 'train_decode.csv'
local base_img_dir = '/home/ubuntu/data/raw-data/kaggle-plankton/train'
local file = io.open(train_decode_fname, 'r')
local skip_head = true
local n_samples = 1
for line in file:lines() do
  if not skip_head then
    n_samples = n_samples + 1
  end
  skip_head = false
end
print(n_samples)

local file = io.open(train_decode_fname, 'r')
local skip_head = true
local line
local j = 0
local max_j = 200
local image_tile = torch.Tensor(max_j, INPUT_SZ[1], INPUT_SZ[2], INPUT_SZ[3])
for line in file:lines() do
  if not skip_head then
    j = j + 1
    local row_str = string.split(line, ',')
    local lab = row_str[1]
    local img = image.loadJPG(string.format('%s/%s', base_img_dir, row_str[2]))
    -- img:add(-1):mul(-1)
    local img_sz = img:size()
    local img_sz_max = math.max(img_sz[2], img_sz[3])
    -- local img_sz_max = math.max(img_sz[2], img_sz[3], 256)
    local new_img = torch.Tensor(INPUT_SZ[1], img_sz_max, img_sz_max):fill(1)
    local offset_w = math.floor((img_sz_max - img_sz[3]) / 2)
    local offset_h = math.floor((img_sz_max - img_sz[2]) / 2)
    new_img:narrow(2, offset_h+1, img_sz[2]):narrow(3, offset_w+1, img_sz[3]):copy(img)
    new_img = image.scale(new_img, 48, 48, 'bilinear')
    image_tile[j]:copy(new_img)
  end
  if j == max_j then
    break
  end
  skip_head = false
end

image_tile = image.toDisplayTensor{input=image_tile, padding=4, nrow=10}
image.saveJPG('img/test.jpg', image_tile)


-- local i = 1
-- for j = 1, 10 do
  -- i = i + 1
-- end
-- print(i)



  
-- local col = string.split(line, ",")
-- local img = image.load(string.format("%s/train/%d.png", DATA_DIR, tonumber(col[1])))
-- x[i]:copy(img)
-- y[i]:copy(label_vector(col[2]))
-- if i % 100 == 0 then
  -- xlua.progress(i, TRAIN_N)
-- end
-- i = i + 1
  -- end
-- end