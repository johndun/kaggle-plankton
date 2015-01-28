require 'torch'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

local train_decode_fname = 'train_decode.csv'
local base_img_dir = '/home/ubuntu/data/raw-data/kaggle-plankton/train'
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

prepare_val_meta = function(pct)
  local files, labels = load_meta_data()
  
  local train_files = {}
  local test_files = {}
  local train_labels = {}
  local test_labels = {}
  local cutoff = math.floor(pct * #files)
  
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

training_loop = function(config, files, labels)
  local N = #files
  -- local N = 128
  local shuffle = torch.randperm(N)
  local batch_size = config.batch_size or 12
  local c = 1
  local inputs = torch.Tensor(batch_size, NUM_COLORS, INPUT_SZ, INPUT_SZ)
  local targets = torch.Tensor(batch_size, #CLASSES)
  for t = 1, N, batch_size do
    if t + batch_size - 1 > N then
      break
    end
    if opt.progress then
      xlua.progress(t, N)
    end
    targets:zero()
    for i = 1, batch_size do
      local fname = files[shuffle[t + i - 1]]
      local label = labels[shuffle[t + i - 1]]
      targets[i][label] = 1
      local img = image.loadJPG(base_img_dir .. '/' .. fname)
      img = image.scale(img, INPUT_SZ, INPUT_SZ, 'bilinear')
      inputs[i]:copy(img)
    end
    inputs:add(-1):mul(-1):abs()
    -- local image_tile = image.toDisplayTensor{input=inputs, padding=4, nrow=8}
    -- image.saveJPG('img/test.jpg', image_tile)
  end
  if opt.progress then
    xlua.progress(N, N)
  end
end

local cmd = torch.CmdLine()
cmd:option('-progress', false, 'show progress bars')
opt = cmd:parse(arg)

config = {
  batch_size = 128
}

local train_files, train_labels = load_meta_data()
training_loop(config, train_files, train_labels)

-- local train_files, test_files, train_labels, test_labels = 
   -- prepare_val_meta(0.1)
-- print(#train_files)
-- print(#test_files)


-- image_tile = image.toDisplayTensor{input=image_tile, padding=4, nrow=10}
-- image.saveJPG('img/test.jpg', image_tile)


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