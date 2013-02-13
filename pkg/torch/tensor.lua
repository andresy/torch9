local torch = require 'torch'
local argcheck = require 'torch.argcheck'
local display = require 'torch.display'
local ffi = require 'ffi'

local Tensor = torch.class('torch.Tensor')

local longvlact = ffi.typeof('long[?]')

local function carray2table(arr, size)
   local tbl = {}
   for i=1,size do
      tbl[i] = tonumber(arr[i-1])
   end
   return tbl
end

local function rawInit(self)
   self.__storageOffset = 0
   self.__nDimension = 0
   self.__flag = 0
   return self
end

local function rawResize(self, nDimension, size, stride)
   local hascorrectsize = true

   local nDimension_ = 0
   for d=0,nDimension-1 do
      if size[d] > 0 then
         nDimension_ = nDimension_ + 1
         if self.__nDimension > d and size[d] ~= self.__size[d] then
            hascorrectsize = false
         end
         if self.__nDimension > d and stride and stride[d] >= 0 and stride[d] ~= self.__stride[d] then
            hascorrectsize = false
         end
      else
         break
      end
   end
   nDimension = nDimension_

   if nDimension ~= self.__nDimension then
      hascorrectsize = false
   end

   if hascorrectsize then
      return
   end
   
   if nDimension > 0 then
      if nDimension ~= self.__nDimension then
         self.__size = longvlact(nDimension)
         self.__stride = longvlact(nDimension)
         self.__nDimension = nDimension
      end
      
      totalSize = 1
      for d=self.__nDimension-1,0,-1 do
         self.__size[d] = size[d]
         if stride and stride[d] >= 0 then
            self.__stride[d] = stride[d]
         else
            if d == self.__nDimension-1 then
               self.__stride[d] = 1
            else
               self.__stride[d] = self.__size[d+1]*self.__stride[d+1]
            end
         end
         totalSize = totalSize + (self.__size[d]-1)*self.__stride[d]
      end
      
      if totalSize+self.__storageOffset > 0 then
         if not self.__storage then
            self.__storage = torch.Storage()
         end
         if totalSize+self.__storageOffset > self.__storage.__size then
            self.__storage:resize(tonumber(totalSize+self.__storageOffset))
         end
      end
   else
      self.__nDimension = 0
   end
end

local function rawSet(self, storage, storageOffset, nDimension, size, stride)
   -- storage
   self.__storage = storage
   
   -- storageOffset
   assert(storageOffset >= 0, "Tensor: invalid storage offset")
   self.__storageOffset = storageOffset

   -- size and stride
   rawResize(self, nDimension, size, stride)
end

-- access methods
Tensor.storage = argcheck{
   {{name='self', type='torch.Tensor'}},
   function(self)
      return self.__storage
   end
}

Tensor.storageOffset= argcheck{
   {{name='self', type='torch.Tensor'}},
   function(self)
      return self.__storageOffset
   end
}
Tensor.offset = Tensor.storageOffset

Tensor.nDimension= argcheck{
   {{name='self', type='torch.Tensor'}},
   function(self)
      return self.__nDimension
   end
}
Tensor.dim = Tensor.nDimension

Tensor.size = argcheck{
   {{name='self', type='torch.Tensor'},
    {name='dim', type='number', opt=true}},
   function(self, dim)
      if dim then
         assert(dim > 0 and dim <= self.__nDimension, 'out of range')
         return tonumber(self.__size[dim-1])
      else
         return carray2table(self.__size, self.__nDimension) -- DEBUG: 0-index inconsistency
      end
   end
}


Tensor.stride = argcheck{
   {{name='self', type='torch.Tensor'},
    {name='dim', type='number', opt=true}},
   function(self, dim)
      if dim then
         assert(dim > 0 and dim <= self.__nDimension, 'out of range')
         return tonumber(self.__stride[dim-1])
      else
         return carray2table(self.__stride, self.__nDimension) -- DEBUG: 0-index inconsistency
      end
   end
}

Tensor.data = argcheck{
   {{name='self', type='torch.Tensor'}},
   function(self)
      if self.__storage then
         return self.__storage.__data+self.__storageOffset
      end
   end
}

Tensor.setFlag = argcheck{
   {{name='self', type='torch.Tensor'},
    {name='flag', type='number'}},
   function(self, flag)
      self.__flag = bit.bor(self.__flag, flag)
      return self
   end
}

Tensor.clearFlag = argcheck{
   {{name='self', type='torch.Tensor'},
    {name='flag', type='number'}},
   function(self, flag)
      self.__flag = bit.band(self.__flag, bit.bnot(flag))
      return self
   end
}

-- creation
local function new(storage, storageOffset, size, stride)
   local self = Tensor.__init()
   rawInit(self)

   local dim = size and #size or 0

   storageOffset = storageOffset or 0

   if size and stride then
      assert(dim == #stride, 'inconsistent size/stride sizes')
   end

   rawSet(self,
          storage,
          storageOffset,
          dim,
          size and longvlact(dim, size) or nil,
          stride and longvlact(dim, stride) or nil)

   return self
end

Tensor.new = argcheck{
   {},
   new,

   {{name='storage', type='torch.Storage'},
    {name='storageOffset', type='number', default=0},
    {name='size', type='numbers', vararg=false, opt=true},
    {name='stride', type='numbers', vararg=false, opt=true}},
   new,

   {{name='size', type='numbers'}}, -- lower priority than the data init
   function(size)
      return new(nil, nil, size, nil)
   end,

   {{name='tensor', type='torch.Tensor'}},
   function(tensor)
      return new(tensor.__storage,
                 tensor.__storageOffset,
                 carray2table(tensor.__size, tensor.__nDimension),
                 carray2table(tensor.__stride, tensor.__nDimension))
   end
}

Tensor.set = argcheck{
   {{name='self', type='torch.Tensor'},
    {name='src', type='torch.Tensor'}},
   function(self, src)
      if self ~= src then
         rawSet(self,
                src.__storage,
                src.__storageOffset,
                src.__nDimension,
                src.__size,
             src.__stride)
      end
      return self
   end
}

Tensor.resize = argcheck{
   {{name='self', type='torch.Tensor'},
    {name='size', type='numbers', vararg=false},
    {name='stride', type='numbers', vararg=false}},
   function(self, size, stride)
      local dim = #size
      assert(not stride or (#stride == dim), 'inconsistent size/stride sizes')
      rawResize(self, dim, longvlact(dim, size), stride and longvlact(dim, stride))
   end,

   {{name='self', type='torch.Tensor'},
    {name='size', type='numbers'}},
   function(self, size)
      local dim = #size
      rawResize(self, dim, longvlact(dim, size))
   end
}

Tensor.resizeAs = argcheck{
   {{name='self', type='torch.Tensor'},
    {name='src', type='torch.Tensor'}},
   function(self, src)
      rawResize(self, src.__nDimension, src.__size, nil)
      return self
   end
}

Tensor.narrow = argcheck{
   {{name='self', type='torch.Tensor'},
    {name='src', type='torch.Tensor', opt=true},
    {name='dim', type='number'},
    {name='idx', type='number'},
    {name='size', type='number'}},
   function(self, src, dim, idx, size)
      local dst = src and self or torch.Tensor()
      src = src or self
      dim = dim - 1
      idx = idx - 1

      dst:set(src)

      assert(dim >= 0 and dim < src.__nDimension, 'out of range')
      assert(idx >= 0 and idx < src.__size[dim], 'out of range')
      assert(size > 0 and idx+size <= src.__size[dim], 'out of range')

      if idx > 0 then
         dst.__storageOffset = dst.__storageOffset + idx*dst.__stride[dim];
      end
      dst.__size[dim] = size

      return dst
   end
}

Tensor.select = argcheck{
   {{name='self', type='torch.Tensor'},
    {name='src', type='torch.Tensor', opt=true},
    {name='dim', type='number'},
    {name='idx', type='number'}},
   function(self, src, dim, idx)
      local dst = src and self or torch.Tensor()
      src = src or self
      dim = dim - 1
      idx = idx - 1

      assert(dim >= 0 and dim < src.__nDimension, 'out of range')
      assert(idx >= 0 and idx < src.__size[dim], 'out of range')

      if src.__nDimension == 1 then
         return tonumber( (src.__storage.__data + src.__storageOffset)[idx*self.__stride[0]] )
      else
         dst:narrow(src, dim+1, idx+1, 1) -- DEBUG: 0-index confusing
         for d=dim,self.__nDimension-2 do
            dst.__size[d] = dst.__size[d+1]
            dst.__stride[d] = dst.__stride[d+1]
         end
         dst.__nDimension = dst.__nDimension -1
      end

      return dst
   end
}

Tensor.transpose = argcheck{
   {{name='self', type='torch.Tensor'},
    {name='src', type='torch.Tensor', opt=true},
    {name='dim1', type='number'},
    {name='dim2', type='number'}},
   function(self, src, dim1, dim2)
      local dst = src and self or torch.Tensor()
      src = src or self
      dst:set(src)
      dim1 = dim1 - 1
      dim2 = dim2 - 1

      assert(dim1 >= 0 and dim1 < src.__nDimension, 'out of range')
      assert(dim2 >= 0 and dim2 < src.__nDimension, 'out of range')

      if dim1 == dim2 then
         return dst
      end

      local z = dst.__stride[dim1]
      dst.__stride[dim1] = dst.__stride[dim2]
      dst.__stride[dim2] = z
      z = dst.__size[dim1]
      dst.__size[dim1] = dst.__size[dim2]
      dst.__size[dim2] = z

      return dst
   end
}

Tensor.unfold = argcheck{
   {{name='self', type='torch.Tensor'},
    {name='src', type='torch.Tensor', opt=true},
    {name='dim', type='number'},
    {name='size', type='number'},
    {name='step', type='number'}},
   function(self, src, dim, size, step)
      local dst = src and self or torch.Tensor()
      src = src or self
      dst:set(src)
      dim = dim -1

      assert(src.__nDimension > 0, "cannot unfold an empty tensor")
      assert(dim >= 0 and dim < src.__nDimension, "out of range")
      assert(size <= src.__size[dim], "out of range")
      assert(step > 0, "invalid step")

      local newSize = longvlact(dst.__nDimension+1)
      local newStride = longvlact(dst.__nDimension+1)

      newSize[dst.__nDimension] = size
      newStride[dst.__nDimension] = dst.__stride[dim]
      for d=0,dst.__nDimension-1 do
         if d == dim then
            newSize[d] = math.floor((dst.__size[d] - size) / step) + 1
            newStride[d] = step*dst.__stride[d]
         else
            newSize[d] = dst.__size[d]
            newStride[d] = dst.__stride[d]
         end
      end

      dst.__size = newSize
      dst.__stride = newStride
      dst.__nDimension = dst.__nDimension + 1

      return dst
   end
}

Tensor.squeeze = argcheck{
   {{name='self', type='torch.Tensor'},
    {name='src', type='torch.Tensor', opt=true}},
   function(self, src)
      local dst = src and self or torch.Tensor()
      src = src or self
      dst:set(src)

      -- return nothing if tensor is empty!
      if dst.__nDimension == 0 then
         return
      end

      local ndim = 0
      for d=0,src.__nDimension-1 do
         if src.__size[d] ~= 1 then
            if d ~= ndim then
               dst.__size[ndim] = src.__size[d]
               dst.__stride[ndim] = src.__stride[d]
            end
            ndim = ndim + 1
         end
      end

      --- handle 0-dimension tensors
      if ndim == 0 then
         return tonumber( (dst.__storage.__data + dst.__storageOffset)[0] )
      end
      dst.__nDimension = ndim

      return dst
   end
}

Tensor.squeeze1d = argcheck{
   {{name='self', type='torch.Tensor'},
    {name='src', type='torch.Tensor', opt=true},
    {name='dim', type='number'}},
   function(self, src, dim)
      local dst = src and self or torch.Tensor()
      src = src or self
      dst:set(src)
      dim = dim - 1

      assert(dim >= 0 and dim < src.__nDimension, "dimension out of range")

      dst:set(src)
      if src.__size[dim] == 1 and src.__nDimension > 1 then
         for d=dimension,dst.__nDimension-2 do
            dst.__size[d] = dst.__size[d+1]
            dst.__stride[d] = dst.__stride[d+1]
         end
         dst.__nDimension = dst.__nDimension - 1
      end
      return dst
   end
}

Tensor.isContiguous = argcheck{
   {{name='self', type='torch.Tensor'}},
   function(self)
      local z = 1
      for d=self.__nDimension-1,0,-1 do
         if self.__size[d] ~= 1 then
            if self.__stride[d] == z then
               z = z * self.__size[d]
            else
               return false
            end
         end
      end
      return true
   end
}

Tensor.nElement = argcheck{
   {{name='self', type='torch.Tensor'}},
   function(self)
      if self.__nDimension == 0 then
         return 0
      else
         local nElement = 1;
         for d=0,self.__nDimension-1 do
            nElement = nElement*self.__size[d]
         end
         return tonumber(nElement)
      end
   end
}

function Tensor:__index(k)
   if type(k) == 'number' then
      if self.__nDimension == 1 then
         return self.__storage[tonumber(k+self.__storageOffset)]
      elseif self.__nDimension > 1 then
         return self:select(1, k)
      else
         error('empty tensor')
      end
   else
      return Tensor[k]
   end
end

Tensor.__tostring = display.tensor

function Tensor:write(file)
   file:writeLong(self.__nDimension)
   file:writeRaw('long', self.__size, self.__nDimension)
   file:writeRaw('long', self.__stride, self.__nDimension)
   file:writeLong(self.__storageOffset)
   file:writeObject(self.__storage)
end

function Tensor:read(file)
   self.__nDimension = file:readLong()
   self.__size = longvlact(self.__nDimension)
   self.__stride = longvlact(self.__nDimension)
   file:readRaw('long', self.__size, self.__nDimension)
   file:readRaw('long', self.__stride, self.__nDimension)
   self.__storageOffset = file:readLong()
   self.__storage = file:readObject()
end

torch.Tensor = torch.constructor(Tensor)
