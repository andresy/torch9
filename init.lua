if not jit then
   error('FATAL: torch9 is luajit *only*')
end

local ffi = require 'ffi'

ffi.cdef[[
void free(void *ptr);
void *malloc(size_t size);
void *realloc(void *ptr, size_t size);
typedef unsigned char byte;
]]

require 'torch.timer'

require 'torch.storage'
require 'torch.tensor'

require 'torch.apply'
require 'torch.dimapply'
require 'torch.maths'
require 'torch.lapack'
require 'torch.conv'
require 'torch.tensorop'
require 'torch.random'

require 'torch.file'
require 'torch.diskfile'
require 'torch.memoryfile'

require 'torch.serialization'

local torch = require 'torch.env'

torch.Tensor = torch.DoubleTensor

return torch
