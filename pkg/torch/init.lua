local ffi = require 'ffi'
require "paths"

ffi.cdef[[
void free(void *ptr);
void *malloc(size_t size);
void *realloc(void *ptr, size_t size);
typedef unsigned char byte;
]]


-- torch is global for now [due to getmetatable()]
torch = {}
package.loaded.torch = torch

-- adapt usual global functions to torch7 objects
local luatostring = tostring
function tostring(arg)
   local flag, func = pcall(function(arg) return arg.__tostring end, arg)
   if flag and func then
      return func(arg)
   end
   return luatostring(arg)
end

function torch.type(obj)
   local tname = type(obj)
   if tname == 'table' then
      return obj.__typename or tname
   end
   return tname
end
torch.typename = torch.type -- backward compatibility... keep it or not?

function torch.istype(obj, typename)
   local tname = type(obj)
   if tname == 'table' then
      if obj.__typename then
         obj = getmetatable(obj)
         while type(obj) == 'table' do
            if obj.__typename == typename then
               return true
            else
               obj = getmetatable(obj)
            end
         end
         return false
      else
         return tname == typename
      end
   else
      return typename == tname
   end
end

function torch.getmetatable(str)
   local module, name = str:match('([^%.]+)%.(.+)')   
   local rtbl = _G[module][name]
   if rtbl then 
      return getmetatable(rtbl)
   end
end

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
