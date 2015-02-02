require './util'
require 'image'

local LOG2 = math.log(2)

local files, labels = load_train_meta()
local areas = {}
local N = #files
local f = io.open(TRAIN_DECODE_FNAME, 'w')
f:write('label,fname,area\n')
for i = 1, N do
  local fname = 'data/train/' .. files[i]
  local src = image.loadJPG(fname)
  areas[i] = math.log(src:size(2) * src:size(3)) / LOG2
  f:write(labels[i] .. ',' ..
          files[i] .. ',' .. 
          areas[i] .. '\n')
end
f:close()

local files = load_test_meta()
local areas = {}
local N = #files
local f = io.open(TEST_DECODE_FNAME, 'w')
f:write('fname,area\n')
for i = 1, N do
  local fname = 'data/test/' .. files[i]
  local src = image.loadJPG(fname)
  areas[i] = math.log(src:size(2) * src:size(3)) / LOG2
  f:write(files[i] .. ',' ..
          areas[i] .. '\n')
end
f:close()
