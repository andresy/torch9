local ffi = require 'ffi'
require "paths"

ffi.cdef[[
void free(void *ptr);
void *malloc(size_t size);
void *realloc(void *ptr, size_t size);
typedef unsigned char byte;
]]

torch = {}
package.loaded.torch = torch

require 'torch.type'

function torch.class(tname, parenttname)

   local function constructor(...)
      local self = {}
      torch.setmetatable(self, tname)
      if self.__init then
         self:__init(...)
      end
      return self
   end
   
   local function factory()
      local self = {}
      torch.setmetatable(self, tname)
      return self
   end

   local mt = torch.newmetatable(tname, parenttname, constructor, nil, factory)
   local mpt
   if parenttname then
      mpt = torch.getmetatable(parenttname)
   end
   return mt, mpt
end

local argcheck = require 'torch.argcheck'
require 'torch.argtypes'
local argcheckenv = getfenv(argcheck)
argcheckenv.type = torch.type
argcheckenv.istype = torch.istype

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

torch.Tensor = torch.DoubleTensor

return torch
