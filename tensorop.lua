local display = require 'torch.display'
local torch = require 'torch.env'
local class = require 'class'
local ffi = require 'ffi'
local C = require 'torch.TH'

local RealTensor = class.metatable('torch.RealTensor')

local function index_table(self, k, v)
   assert(#k <= self.__nDimension, 'invalid table size')
   local cdim = 0
   local res
   self = C.THRealTensor_newWithTensor(self)[0]
   for dim=0,self.__nDimension-1 do
      local z = k[dim+1]
      if type(z) == 'number' then
         z = z - 1
         if z < 0 then
            z = self.__size[cdim] + z + 1
         end
         assert(z >= 0 and z < self.__size[cdim], 'out of range')
         if self.__nDimension == 1 then
            res = self.__storage.__data+self.__storageOffset+z*self.__stride[0]
         else
            C.THRealTensor_select(self, nil, cdim, z)
         end
      elseif type(z) == 'table' then
         local a = 0
         local b = self.__size[cdim]-1

         local zz = z[1]
         if type(zz) == 'number' then
            a = zz-1
            b = a
         end
         if a < 0 then
            a = self.__size[cdim] + a + 1
         end
         assert(a >= 0 and a < self.__size[cdim], 'out of range')

         local zz = z[2]
         if type(zz) == 'number' then
            b = zz-1
         end
         if b < 0 then
            b = self.__size[cdim] + b + 1
         end
         assert(b >= 0 and b < self.__size[cdim], 'out of range')

         assert(b >= a, 'end index must be greater or equal to start index')
         C.THRealTensor_narrow(self, nil, cdim, a, b-a+1)
         cdim = cdim + 1
      elseif type(z) ~= 'nil' then
         error('invalid table')
      end
   end
   if v then
      if res then
         res[0] = v
         C.THRealTensor_free(self)
      else
         self:copy(v) -- DEBUG: this could fail
         C.THRealTensor_free(self)
      end
   else
      if res then
         C.THRealTensor_free(self)
         return tonumber(res)
      else
         ffi.gc(self, C.THRealTensor_free)
         return self
      end
   end
end

function RealTensor:__index(k)
   local type_k = class.type(k)
   if type_k == 'number' then
      if self.__nDimension == 1 then
         assert(k > 0 and k <= self.__size[0], 'out of range')
         return tonumber( self.__storage.__data[(k-1)*self.__stride[0]+self.__storageOffset] )
      elseif self.__nDimension > 1 then
         assert(k > 0 and k <= self.__size[0], 'out of range')
         return self:select(1, k-1)
      else
         error('empty tensor')
      end
   elseif type_k == 'torch.LongStorage' then
      assert(k.__size == self.__nDimension, 'invalid storage size')
      local idx = self.__storageOffset
      for dim=0,tonumber(k.__size)-1 do
         local z = k.__data[dim]-1
         assert(z >= 0 and z < self.__size[dim], 'out of range')
         idx = idx + z*self.__stride[dim]
      end
      return tonumber(self.__storage.__data[idx])
   elseif type_k == 'torch.ByteTensor' then
      local vals = torch.RealTensor()
      C.THRealTensor_maskedSelect(vals, self, k)
      return vals
   elseif type_k == 'table' then
      return index_table(self, k)
   else
      return RealTensor[k]
   end
end

function RealTensor:__newindex(k, v)
   local type_k = class.type(k)
   local type_v = class.type(v)
   if type_k == 'number' then
      if type_v == 'number' then
         if self.__nDimension == 1 then
            assert(k > 0 and k <= self.__size[0], 'out of range')
            self.__storage[self.__storageOffset+(k-1)*self.__stride[0]] = v
         elseif self.__nDimension > 1 then
            local t = C.THRealTensor_newWithTensor(t)
            C.THRealTensor_narrow(t, nil, 0, k-1, 1)
            C.THRealTensor_fill(t, v)
            C.THRealTensor_free(t)
         else
            error('empty tensor')
         end
      elseif
         type_v == 'torch.ByteTensor'
         or type_v == 'torch.CharTensor'
         or type_v == 'torch.ShortTensor'
         or type_v == 'torch.IntTensor'
         or type_v == 'torch.LongTensor'
         or type_v == 'torch.FloatTensor'
         or type_v == 'torch.DoubleTensor' then
            local t = self:narrow(1, k, 1) -- use gc, as this can fail
            t:copy(v)
      end
   elseif type_k == 'torch.LongStorage' then
      assert(type_v == 'number', 'number expected as value for a LongStorage as key')
      assert(k.__size == self.__nDimension, 'invalid storage size')
      local idx = self.__storageOffset
      for dim=0,tonumber(k.__size)-1 do
         local z = k.__data[dim]-1
         assert(z >= 0 and z < self.__size[dim], 'out of range')
         idx = idx + z*self.__stride[dim]
      end
      self.__storage.__data[idx] = v
   elseif type_k == 'torch.ByteTensor' then
      if type_v == 'number' then
         C.THRealTensor_maskedFill(self, k, v)
      elseif type_v == 'torch.RealTensor' then
         C.THRealTensor_maskedCopy(self, k, v)
      else
         error('when using a mask as a key, number or tensor are expected as value')
      end
   elseif type_k == 'table' then
      index_table(self, k, v)
   else
      rawset(self, k, v)
   end
end

RealTensor.__tostring = display.tensor

function RealTensor.__add(t1, t2)
   local type_t1 = class.type(t1)
   local type_t2 = class.type(t2)

   local r = torch.RealTensor()
   if type_t1 == 'torch.RealTensor' and type_t2 == 'number' then
      r:resizeAs(t1)
      r:fill(t2)
      r:add(t1)
   elseif type_t1 == 'number' and type_t2 == 'torch.RealTensor' then
      r:resizeAs(t2)
      r:fill(t1)
      r:add(t2)
   elseif type_t1 == 'torch.RealTensor' and type_t2 == 'torch.RealTensor' then
      r:resizeAs(t1)
      r:copy(t1)
      r:add(t2)
   else
      error('two tensors or one tensor and one number expected')
   end
   
   return r
end

function RealTensor.__sub(t1, t2)
   local type_t1 = class.type(t1)
   local type_t2 = class.type(t2)

   local r = torch.RealTensor()
   if type_t1 == 'torch.RealTensor' and type_t2 == 'number' then
      r:resizeAs(t1)
      r:copy(t1)
      r:add(-t2)
   elseif type_t1 == 'number' and type_t2 == 'torch.RealTensor' then
      r:resizeAs(t2)
      r:fill(t1)
      r:add(-1, t1)
   elseif type_t1 == 'torch.RealTensor' and type_t2 == 'torch.RealTensor' then
      r:resizeAs(t1)
      r:copy(t1)
      r:add(-1, t2)
   else
      error('two tensors or one tensor and one number expected')
   end
   
   return r
end

function RealTensor.__unm(self)
   local r = torch.RealTensor()
   r:resizeAs(self)
   r:zero()
   r:add(-1, self)
   return r
end

function RealTensor.__mul(t1, t2)
   local type_t1 = class.type(t1)
   local type_t2 = class.type(t2)

   local r = torch.RealTensor()
   if type_t1 == 'torch.RealTensor' and type_t2 == 'number' then
      r:resizeAs(t1)
      r:zero()
      r:add(t2, t1)
   elseif type_t1 == 'number' and type_t2 == 'torch.RealTensor' then
      r:resizeAs(t2)
      r:zero()
      r:add(t1, t2)
   elseif type_t1 == 'torch.RealTensor' and type_t2 == 'torch.RealTensor' then
      if t1.__nDimension == 1 and t2.__nDimension == 1 then
         return t1:dot(t2)
      elseif t1.__nDimension == 2 and t2.__nDimension == 1 then
         return t1:mv(t2)
      elseif t1.__nDimension == 2 and t2.__nDimension == 2 then
         return t1:mm(t2)
      else
         error(string.format('multiplication between %dD and %dD tensorsnot yet supported',
                             t1.__nDimension, t2.__nDimension))
      end
   else
      error('two tensors or one tensor and one number expected')
   end
   
   return r
end

function RealTensor.__div(t1, t2)
   local type_t1 = class.type(t1)
   local type_t2 = class.type(t2)

   assert(type_t2 == 'number', 'number expected')
   
   local r = torch.RealTensor()
   r:resizeAs(t1)
   r:copy(t1)
   r:mul(1/t2)

   return r
end
