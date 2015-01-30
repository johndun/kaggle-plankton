TEST_JITTER_SZ = 32

random_jitter = function(fname)
  local jitter = config.train_jitter or false
  if not jitter then
    return sample_image{
      fname = fname, 
      resize_x = INPUT_SZ, 
      resize_y = INPUT_SZ
    }
  end
  local rotation = math.random(4) - 1
  local hflip = math.random() < 0.5
  local scales = {0.9, 1.0}
  local x_scale = scales[math.random(2)]
  local y_scale = scales[math.random(2)]
  return sample_image{
    fname = fname, 
    rotate = rotation * 2 * math.pi / 4, 
    crp_sz_x = x_scale, 
    crp_off_x = (1 - x_scale) / 2, 
    crp_sz_y = y_scale, 
    crp_off_y = (1 - y_scale) / 2, 
    hflip    = hflip, 
    out_w    = INPUT_SZ, 
    out_h    = INPUT_SZ
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
  for rotation = 0, 3 do
    for k, hflip in pairs({false, true}) do
      for k, x_scale in pairs({0.9, 1.0}) do
        for k, y_scale in pairs({0.9, 1.0}) do
          local img = sample_image{
            fname = fname, 
            rotate = rotation * 2 * math.pi / 4, 
            crp_sz_x = x_scale, 
            crp_off_x = (1 - x_scale) / 2, 
            crp_sz_y = y_scale, 
            crp_off_y = (1 - y_scale) / 2, 
            hflip    = hflip, 
            out_w    = INPUT_SZ, 
            out_h    = INPUT_SZ
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

require './util.lua'
local x = test_jitter('data/test/100.jpg', true)
print(x:size())
local image_tile = image.toDisplayTensor{input=x, padding=4, nrow=8}
image.saveJPG('img/test-jitter.jpg', image_tile)
