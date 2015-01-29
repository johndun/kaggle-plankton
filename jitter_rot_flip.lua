TEST_JITTER_SZ = 8

random_jitter = function(fname)
  local jitter = config.train_jitter or false
  if not jitter then
    return sample_image{
      fname = fname, 
      resize_x = INPUT_SZ, 
      resize_y = INPUT_SZ
    }
  end
  local rotation = 2 * math.pi * (math.random(4) - 1) / 4
  local hflip = math.random() < 0.5
  return sample_image{
    fname = fname, 
    resize_x = INPUT_SZ, 
    resize_y = INPUT_SZ, 
    rotate = rotation, 
    hflip = hflip
  }
end

test_jitter = function(fname, jitter)
  local jitter = config.test_jitter or jitter or false
  if not jitter then
    return sample_image{
      fname = fname, 
      resize_x = INPUT_SZ, 
      resize_y = INPUT_SZ
    }
  end
  local samples = {}
  for k, hflip in pairs({false, true}) do
    for rotation = 0, 3 do
      local img = sample_image{
        fname = fname, 
        rotate = rotation * 2 * math.pi / 4, 
        resize_x = INPUT_SZ, 
        resize_y = INPUT_SZ, 
        hflip    = hflip
      }
      table.insert(samples, img)
    end
  end
  local result = torch.Tensor(#samples, NUM_COLORS, INPUT_SZ, INPUT_SZ)
  for i = 1, #samples do
    result[i]:copy(samples[i])
  end
  return result
end

-- require './util.lua'
-- local x = test_jitter('data/test/100.jpg', true)
-- print(x:size())
-- local image_tile = image.toDisplayTensor{input=x, padding=4, nrow=4}
-- image.saveJPG('img/test-jitter.jpg', image_tile)
