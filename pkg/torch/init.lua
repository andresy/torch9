local ffi = require 'ffi'
require "paths"

ffi.cdef[[
void free(void *ptr);
void *malloc(size_t size);
void *realloc(void *ptr, size_t size);
]]


-- torch is global for now [due to getmetatable()]
torch = {}


-- adapt usual global functions to torch7 objects
local luatostring = tostring
function tostring(arg)
   local flag, func = pcall(function(arg) return arg.__tostring end, arg)
   if flag and func then
      return func(arg)
   end
   return luatostring(arg)
end

local luatype = type
function type(arg)
   local tname = luatype(arg)
   if tname == 'table' then
      return arg.__typename or tname
   end
   return tname
end

torch.typename = type -- backward compatibility... keep it or not?

function torch.getmetatable(str)
   local module, name = str:match('([^%.]+)%.(.+)')   
   local rtbl = _G[module][name]
   if rtbl then 
      return getmetatable(rtbl)
   end
end

function include(file, env)
   if env then
      local filename = paths.thisfile(file, 3)
      local f = io.open(filename)
      local txt = f:read('*all')
      f:close()
      local code, err = loadstring(txt, filename)
      if not code then
         error(err)
      end
      setfenv(code, env)
      code()      
   else
      paths.dofile(file, 3)
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

function torch.setdefaulttensortype(typename)
   assert(type(typename) == 'string', 'string expected')
   if torch.getconstructortable(typename) then
      torch.Tensor = torch.getconstructortable(typename)
      torch.Storage = torch.getconstructortable(torch.typename(torch.Tensor(1):storage()))
   else
      error(string.format("<%s> is not a string describing a torch object", typename))
   end
end

local function includetemplate(file, env)
   env = env or _G
   local filename = paths.thisfile(file, 3)
   local f = io.open(filename)
   local txt = f:read('*all')
   f:close()
   local types = {char='Char', short='Short', int='Int', long='Long', float='Float', double='Double'}
   types['unsigned char'] = 'Byte'
   for real,Real in pairs(types) do
      local txt = txt:gsub('([%p%s])real([%p%s])', '%1' .. real .. '%2')
      txt = txt:gsub('([%p%s])Storage([%p%s])', '%1' .. Real .. 'Storage' .. '%2')
      txt = txt:gsub('([%p%s])Tensor([%p%s])', '%1' .. Real .. 'Tensor' .. '%2')
      local code, err = loadstring(txt, filename)
      if not code then
         error(err)
      end
      setfenv(code, env)
      code()
   end   
end

--torch.setdefaulttensortype('torch.DoubleTensor')

local env = {ffi=ffi, torch=torch}
setmetatable(env, {__index=_G})

include('Timer.lua', env)
includetemplate('Storage.lua', env)
includetemplate('Tensor.lua', env)

return torch
