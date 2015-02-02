TEST_JITTER_SZ = 32
VAL_JITTER_SZ = 32

random_jitter = function(fname, jitter)
  local jitter = config.train_jitter or jitter or false
  local src = image.loadJPG(fname):add(-1):mul(-1):abs()
  if not jitter then
    return sample_image{
      src      = src, 
      resize_x = INPUT_SZ, 
      resize_y = INPUT_SZ
    }
  end
  local rotation = math.random(4) - 1
  local hflip = math.random() < 0.5
  local offsets = {0.0, 0.14}
  local x_offset = offsets[math.random(2)]
  local y_offset = offsets[math.random(2)]
  return sample_image{
    src       = src, 
    rotate    = rotation * 2 * math.pi / 4, 
    crp_sz_x  = 0.86, 
    crp_off_x = x_offset, 
    crp_sz_y  = 0.86, 
    crp_off_y = y_offset, 
    hflip     = hflip, 
    out_w     = INPUT_SZ, 
    out_h     = INPUT_SZ,
    resize_x  = 56, 
    resize_y  = 56
  }
end

--[[
val_jitter = function(fname, jitter)
  local jitter = config.train_jitter or jitter or false
  local src = image.loadJPG(fname):add(-1):mul(-1):abs()
  if not jitter then
    return sample_image{
      src      = src,
      resize_x = INPUT_SZ,
      resize_y = INPUT_SZ
    }
  end
  local imgs = torch.Tensor(VAL_JITTER_SZ, 
                            NUM_COLORS, 
                            INPUT_SZ, INPUT_SZ)
  for i = 1, VAL_JITTER_SZ do
    local rotation = math.random(4) - 1
    local hflip = math.random() < 0.5
    local offsets = {0.0, 0.14}
    local x_offset = offsets[math.random(2)]
    local y_offset = offsets[math.random(2)]
    local img = sample_image{
      src       = src,
      rotate    = rotation * 2 * math.pi / 4,
      crp_sz_x  = 0.86,
      crp_off_x = x_offset,
      crp_sz_y  = 0.86,
      crp_off_y = y_offset,
      hflip     = hflip,
      out_w     = INPUT_SZ,
      out_h     = INPUT_SZ,
      resize_x  = 56,
      resize_y  = 56
    }
    imgs:narrow(1, i, 1):copy(img)
  end
  return imgs
end
]]--

test_jitter = function(fname, jitter)
  local jitter = config.test_jitter or jitter or false
  local src = image.loadJPG(fname):add(-1):mul(-1):abs()
  if not jitter then
    return sample_image{
      src      = src, 
      resize_x = INPUT_SZ, 
      resize_y = INPUT_SZ
    }
  end
  local samples = {}
  for rotation = 0, 3 do
    for k, hflip in pairs({false, true}) do
      for k, x_off in pairs({0.0, 0.14}) do
        for k, y_off in pairs({0.0, 0.14}) do
          local img = sample_image{
            src       = src, 
            rotate    = rotation * 2 * math.pi / 4, 
            crp_sz_x  = 0.86, 
            crp_off_x = x_off, 
            crp_sz_y  = 0.86, 
            crp_off_y = y_off, 
            hflip     = hflip, 
            resize_x  = 56,  
            resize_y  = 56, 
            out_w     = INPUT_SZ, 
            out_h     = INPUT_SZ
          }
          table.insert(samples, img)
        end
      end
    end
  end
  local result = torch.Tensor(#samples, NUM_COLORS, INPUT_SZ, INPUT_SZ)
  for i = 1, #samples do
    result[i]:copy(samples[i])
  end
  return result
end

val_jitter = test_jitter

--[[
require './util.lua'
local x = test_jitter('data/test/100.jpg', true)
print(x:size())
local image_tile = image.toDisplayTensor{input=x, padding=4, nrow=TEST_JITTER_SZ / 4}
image.saveJPG('img/test-jitter.jpg', image_tile)

local x = val_jitter('data/test/100.jpg', true)
local image_tile = image.toDisplayTensor{input=x, padding=4, nrow=VAL_JITTER_SZ / 2}
print(x:size())
image.saveJPG('img/val-jitter.jpg', image_tile)

local x = random_jitter('data/test/100.jpg', true)
local image_tile = image.toDisplayTensor{input=x, padding=4, nrow=VAL_JITTER_SZ / 2}
image.saveJPG('img/random-jitter.jpg', image_tile)
]]--
