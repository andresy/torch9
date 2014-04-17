local display = require 'torch.display'
local argcheck = require 'argcheck'
local torch = require 'torch.env'
local class = require 'class'
local ffi = require 'ffi'
local C = require 'torch.TH'

local RealTensor = class('torch.RealTensor', nil, ffi.typeof('THRealTensor&'))
torch.RealTensor = RealTensor

local longvlact = ffi.typeof('long[?]')

local function carray2table(arr, size)
   local tbl = {}
   for i=1,size do
      tbl[i] = tonumber(arr[i-1])
   end
   return tbl
end

-- access methods
RealTensor.storage = argcheck{
   {name='self', type='torch.RealTensor'},
   call =
      function(self)
         local storage = self.__storage[0]
         C.THRealStorage_retain(storage)
         ffi.gc(storage, C.THRealStorage_free)
         return storage
      end
}

RealTensor.storageOffset= argcheck{
   {name='self', type='torch.RealTensor'},
   call =
      function(self)
         return tonumber(self.__storageOffset+1)
      end
}
RealTensor.offset = RealTensor.storageOffset

RealTensor.nDimension = argcheck{
   {name='self', type='torch.RealTensor'},
   call =
      function(self)
         return tonumber(self.__nDimension)
      end
}
RealTensor.dim = RealTensor.nDimension

RealTensor.size = argcheck{
   {name='self', type='torch.RealTensor'},
   {name='dim', type='number', opt=true},
   call =
   function(self, dim)
      if dim then
         assert(dim > 0 and dim <= self.__nDimension, 'out of range')
         return tonumber(self.__size[dim-1])
      else
         return carray2table(self.__size, self.__nDimension) -- DEBUG: 0-index inconsistency
      end
   end
}


RealTensor.stride = argcheck{
   {name='self', type='torch.RealTensor'},
   {name='dim', type='number', opt=true},
   call =
   function(self, dim)
      if dim then
         assert(dim > 0 and dim <= self.__nDimension, 'out of range')
         return tonumber(self.__stride[dim-1])
      else
         return carray2table(self.__stride, self.__nDimension) -- DEBUG: 0-index inconsistency
      end
   end
}

RealTensor.data = argcheck{
   {name='self', type='torch.RealTensor'},
   call =
   function(self)
      if self.__storage then
         return self.__storage.__data+self.__storageOffset
      end
   end
}

RealTensor.setFlag = argcheck{
   {name='self', type='torch.RealTensor'},
   {name='flag', type='number'},
   call =
   function(self, flag)
      self.__flag = bit.bor(self.__flag, flag)
      return self
   end
}

RealTensor.clearFlag = argcheck{
   {name='self', type='torch.RealTensor'},
    {name='flag', type='number'},
    call =
   function(self, flag)
      self.__flag = bit.band(self.__flag, bit.bnot(flag))
      return self
   end
}


RealTensor.new = argcheck{
   call =
      function()
         local self = C.THRealTensor_new()[0]
         ffi.gc(self, C.THRealTensor_free)
         return self
      end
}

argcheck{
   {name='storage', type='torch.Storage'},
   {name='storageOffset', type='number', default=1},
   {name='size', type='table', check=checknumbers, opt=true},
   {name='stride', type='table',  check=checknumbers, opt=true},
   chain = RealTensor.new,
   call =
      function(storage, storageOffset, size, stride)
         if size then
            size = ffi.new('long[?]', #size, size)
         end
         if stride then
            stride = ffi.new('long[?]', #stride, stride)
         end
         local self = C.THRealTensor_newWithStorage(storage, storageOffset-1, size, stride)[0]
         ffi.gc(self, C.THRealTensor_free)
         return self
      end
}

argcheck{
   {name='dim1', type='number'},
   {name='dim2', type='number', default=0},
   {name='dim3', type='number', default=0},
   {name='dim4', type='number', default=0},
   chain = RealTensor.new,
   call =
      function(dim1, dim2, dim3, dim4)
         local self = C.THRealTensor_newWithSize4d(dim1, dim2, dim3, dim4)[0]
         ffi.gc(self, C.THRealTensor_free)
         return self         
      end
}

argcheck{
   {name='size', type='table', check=checknumbers}, -- lower priority than the data init
   chain = RealTensor.new,
   call =
   function(size)
         size = ffi.new('long[?]', #size, size)
         local self = C.THRealTensor_newWithSize(size, nil)[0]
         ffi.gc(self, C.THRealTensor_free)
         return self
   end
}

argcheck{
   {name='tensor', type='torch.RealTensor'},
   chain = RealTensor.new,
   call =
      function(tensor)
         local self = C.THRealTensor_newWithTensor(tensor)[0]
         ffi.gc(self, C.THRealTensor_free)
         return self
      end
}

RealTensor.set = argcheck{
   {name='self', type='torch.RealTensor'},
    {name='src', type='torch.RealTensor'},
    call =
       function(self, src)
          C.THRealTensor_set(self, src)
          return self
       end
}

RealTensor.resize = argcheck{
   {name='self', type='torch.RealTensor'},
   {name='size', type='table', check=checknumbers},
   {name='stride', type='table', check=checknumbers, opt=true},
   call =
      function(self, size, stride)
         local dim = #size
         assert(not stride or (#stride == dim), 'inconsistent size/stride sizes')
         size = ffi.new('long[?]', dim, size)
         if stride then
            stride = ffi.new('long[?]', dim, stride)
         end
         C.THRealTensor_resize(self, size, stride)
         return self
      end
}

argcheck{
   {name='self', type='torch.RealTensor'},
   {name='dim1', type='number'},
   {name='dim2', type='number', default=0},
   {name='dim3', type='number', default=0},
   {name='dim4', type='number', default=0},
   chain = RealTensor.resize,
   call =
      function(self, dim1, dim2, dim3, dim4)
         C.THRealTensor_resize4d(self, dim1, dim2, dim3, dim4)
         return self
      end
}

RealTensor.resizeAs = argcheck{
   {name='self', type='torch.RealTensor'},
   {name='src', type='torch.RealTensor'},
   call =
      function(self, src)
         C.THRealTensor_resizeAs(self, src)
         return self
      end
}

RealTensor.narrow = argcheck{
   {name='self', type='torch.RealTensor'},
   {name='src', type='torch.RealTensor', opt=true},
   {name='dim', type='number'},
   {name='idx', type='number'},
   {name='size', type='number'},
   call =
   function(self, src, dim, idx, size)
         if src then
            C.THRealTensor_narrow(self, src, dim-1, idx-1, size)
            return self
         else
            local tensor = C.THRealTensor_newNarrow(self, dim-1, idx-1, size)[0]
            ffi.gc(tensor, C.THRealTensor_free)
            return tensor
         end
   end
}

RealTensor.select = argcheck{
   {name='self', type='torch.RealTensor'},
    {name='src', type='torch.RealTensor', opt=true},
    {name='dim', type='number'},
    {name='idx', type='number'},
    call =
       function(self, src, dim, idx)
         if src then
            C.THRealTensor_select(self, src, dim, idx)
            return self
         else
            local tensor = C.THRealTensor_newSelect(self, dim-1, idx-1)[0]
            ffi.gc(tensor, C.THRealTensor_free)
            return tensor
         end
       end
}

RealTensor.transpose = argcheck{
   {name='self', type='torch.RealTensor'},
    {name='src', type='torch.RealTensor', opt=true},
    {name='dim1', type='number'},
    {name='dim2', type='number'},
    call =
       function(self, src, dim1, dim2)
         if src then
            C.THRealTensor_transpose(self, src, dim1-1, dim2-1)
            return self
         else
            local tensor = C.THRealTensor_newTranspose(self, dim1-1, dim2-1)[0]
            ffi.gc(tensor, C.THRealTensor_free)
            return tensor
         end
       end
}

RealTensor.unfold = argcheck{
   {name='self', type='torch.RealTensor'},
    {name='src', type='torch.RealTensor', opt=true},
    {name='dim', type='number'},
    {name='size', type='number'},
    {name='step', type='number'},
    call =
       function(self, src, dim, size, step)
         if src then
            C.THRealTensor_unfold(self, src, dim-1, size, step)
            return self
         else
            local tensor = C.THRealTensor_newTranspose(self, src, dim-1, size, step)[0]
            ffi.gc(tensor, C.THRealTensor_free)
            return tensor
         end
       end
}

RealTensor.squeeze = argcheck{
   {name='self', type='torch.RealTensor'},
    {name='src', type='torch.RealTensor', opt=true},
    call =
       function(self, src)
          local dst = src and self or torch.RealTensor()
          src = src or self
          if src.__nDimension == 0 then
             return
          elseif src.__nDimension == 1 then
             return src:data()[0]
          else
             C.THRealTensor_squeeze(dst, src)
             return dst
          end
       end
}

RealTensor.squeeze = argcheck{
   {name='self', type='torch.RealTensor'},
    {name='src', type='torch.RealTensor', opt=true},
    {name='dim', type='number'},
    chain = RealTensor.squeeze,
    call =
       function(self, src, dim)
          local dst = src and self or torch.RealTensor()
          src = src or self
          C.THRealTensor_squeeze1d(dst, src, dim-1)
          return dst
       end
}

RealTensor.isContiguous = argcheck{
   {name='self', type='torch.RealTensor'},
   call =
      function(self)
         return C.THRealTensor_isContiguous(self) == 1
      end
}

RealTensor.nElement = argcheck{
   {name='self', type='torch.RealTensor'},
   call =
      function(self)
         return tonumber(C.THRealTensor_nElement(self))
      end
}

function RealTensor:__index(k)
   if type(k) == 'number' then
      assert(self.__nDimension >= 1 and k > 0 and k <= self.__size[0], 'out of bounds')
      if self.__nDimension == 1 then
         k = k - 1
         return tonumber( self.__storage.__data[k*self.__stride[0]+self.__storageOffset] )
      else
         return self:select(1, k)
      end
   else
      return RealTensor[k]
   end
end

RealTensor.__tostring = display.tensor

function RealTensor:write(file)
   file:writeLong(self.__nDimension)
   file:writeRaw('long', self.__size, self.__nDimension)
   file:writeRaw('long', self.__stride, self.__nDimension)
   file:writeLong(self.__storageOffset)
   file:writeObject(self.__storage)
end

function RealTensor:read(file, version)
   self.__nDimension = file:readLong()
   self.__size = longvlact(self.__nDimension)
   self.__stride = longvlact(self.__nDimension)
   file:readRaw('long', self.__size, self.__nDimension)
   file:readRaw('long', self.__stride, self.__nDimension)
   self.__storageOffset = file:readLong()
   if version == 1 then
      self.__storageOffset = self.__storageOffset - 1
   end
   self.__storage = file:readObject()
end

ffi.metatype('THRealTensor', getmetatable(RealTensor))
