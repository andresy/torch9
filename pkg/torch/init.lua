local ffi = require 'ffi'

ffi.cdef[[
void free(void *ptr);
void *malloc(size_t size);
void *realloc(void *ptr, size_t size);
typedef unsigned char byte;
]]

torch = {}
package.loaded.torch = torch

local argcheck = require 'torch.argcheck'
require 'torch.argtypes'
local argcheckenv = getfenv(argcheck)

require 'torch.type'
argcheckenv.type = torch.type
argcheckenv.istype = torch.istype

require 'torch.class'

require 'torch.timer'

require 'torch.storage'
require 'torch.tensor'

require 'torch.apply'
require 'torch.dimapply'
require 'torch.maths'
require 'torch.random'

require 'torch.file'
require 'torch.diskfile'
require 'torch.memoryfile'

require 'torch.serialization'

torch.Tensor = torch.DoubleTensor

return torch
