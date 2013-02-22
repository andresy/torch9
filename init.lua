print('Torch 9.0 -- Copyright (C) 2001-2013 Idiap, NEC Labs, NYU. http://www.torch.ch/')

if not jit then
   error('FATAL: torch9 is luajit *only*')
end

if jit.os == 'OSX' then
   local cpath = package.cpath
   for path in package.cpath:gmatch('[^;]+') do
      if path:match('%.so$') then
         cpath = cpath .. ';' .. path:gsub('%.so$', '.dylib')
      end
   end
   package.cpath = cpath
end

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
