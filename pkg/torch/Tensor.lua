local Tensor = {__typename="torch.Tensor"}
local mt

local function rawInit()
   local self = {}
   self.__storageOffset = 0
   self.__nDimension = 0
   self.__flag = 0
   setmetatable(self, mt)
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
         self.__size = ffi.new("long[?]", nDimension)
         self.__stride = ffi.new("long[?]", nDimension)
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
function Tensor:storage()
  return self.__storage
end

function Tensor:storageOffset()
  return self.__storageOffset + 1
end

function Tensor:nDimension()
  return self.__nDimension
end

function Tensor:dim()
  return self.__nDimension
end

function Tensor:size(dim)
   if dim then
      assert(dim > 0 and dim <= self.__nDimension, 'out of range')
      return tonumber(self.__size[dim-1])
   else
      return torch.LongStorage(self.__nDimension):rawCopy(self.__size)
   end
end

function Tensor:stride(dim)
   if dim then
      assert(dim > 0 and dim <= self.__nDimension, 'out of range')
      return tonumber(self.__stride[dim-1])
   else
      return torch.LongStorage(self.__nDimension):rawCopy(self.__stride)
   end
end

function Tensor:data()
  if self.__storage then
     return self.__storage.data+self.__storageOffset
  else
    return nil
  end
end

function Tensor:setFlag(flag)
   self.__flag = bit.bor(self.__flag, flag)
   return self
end

function Tensor:clearFlag(flag)
   self.__flag = bit.band(self.__flag, bit.bnot(flag))
   return self
end

-- creation

-- checkout http://www.torch.ch/manual/torch/tensor
local function readtensorsizestride(arg)
   local storage
   local offset
   local size
   local stride
   local narg = #arg

   if narg == 0 then
      return nil, 0, nil, nil
   elseif narg == 1 and type(arg[1]) == 'number' then
      return nil, 0, torch.LongStorage{arg[1]}, nil
   elseif narg == 1 and type(arg[1]) == 'table' then
      error('not implemented yet')
      -- todo
   elseif narg == 1 and type(arg[1]) == 'torch.LongStorage' then
      return nil, 0, arg[1], nil
   elseif narg == 1 and type(arg[1]) == 'torch.Storage' then
      return arg[1], 0, nil, nil
   elseif narg == 1 and type(arg[1]) == 'torch.Tensor' then
      return arg[1]:storage(), arg[1]:storageOffset(), arg[1]:size(), arg[1]:stride()
   elseif narg == 2 and type(arg[1]) == 'number' and type(arg[2]) == 'number' then
      return nil, 0, torch.LongStorage{arg[1], arg[2]}, nil
   elseif narg == 2 and type(arg[1]) == 'torch.LongStorage' and type(arg[2]) == 'torch.LongStorage' then
      return nil, 0, arg[1], arg[2]
   elseif narg == 3 and type(arg[1]) == 'number' and type(arg[2]) == 'number' and type(arg[3]) == 'number' then
      return nil, 0, torch.LongStorage{arg[1], arg[2], arg[3]}
   elseif narg == 3 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number' and type(arg[3]) == 'torch.LongStorage' then
      return arg[1], arg[2], arg[3], nil
   elseif narg == 3 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number' and type(arg[3]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3]}, nil
   elseif narg == 4 and type(arg[1]) == 'number' and type(arg[2]) == 'number' and type(arg[3]) == 'number' and type(arg[4]) == 'number' then
      return nil, 0, torch.LongStorage{arg[1], arg[2], arg[3], arg[4]}
   elseif narg == 4 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number' and type(arg[3]) == 'torch.LongStorage' and type(arg[4]) == 'torch.LongStorage' then
      return arg[1], arg[2], arg[3], arg[4]
   elseif narg == 4 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3]}, torch.LongStorage{arg[4]}
   elseif narg == 5 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' and type(arg[5]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3], arg[5]}, torch.LongStorage{arg[4]}
   elseif narg == 6 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' and type(arg[5]) == 'number' and type(arg[6]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3], arg[5]}, torch.LongStorage{arg[4], arg[6]}
   elseif narg == 7 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' and type(arg[5]) == 'number' and type(arg[6]) == 'number' and type(arg[7]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3], arg[5], arg[7]}, torch.LongStorage{arg[4], arg[6]}
   elseif narg == 8 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' and type(arg[5]) == 'number' and type(arg[6]) == 'number' and type(arg[7]) == 'number' and type(arg[8]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3], arg[5], arg[7]}, torch.LongStorage{arg[4], arg[6], arg[8]}
   elseif narg == 9 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' and type(arg[5]) == 'number' and type(arg[6]) == 'number' and type(arg[7]) == 'number' and type(arg[8]) == 'number' and type(arg[9]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3], arg[5], arg[7], arg[9]}, torch.LongStorage{arg[4], arg[6], arg[8]}
   elseif narg == 10 and type(arg[1]) == 'torch.Storage' and type(arg[2]) == 'number'  and type(arg[3]) == 'number' and type(arg[4]) == 'number' and type(arg[5]) == 'number' and type(arg[6]) == 'number' and type(arg[7]) == 'number' and type(arg[8]) == 'number' and type(arg[9]) == 'number' and type(arg[10]) == 'number' then
      return arg[1], arg[2], torch.LongStorage{arg[3], arg[5], arg[7], arg[9]}, torch.LongStorage{arg[4], arg[6], arg[8], arg[10]}
   else
      error('invalid arguments')
   end
end

local function readsizestride(arg)
   local size
   local stride
   local narg = #arg

   if narg == 1 and type(arg[1]) == 'number' then
      return torch.LongStorage{arg[1]}, nil
   elseif narg == 1 and type(arg[1]) == 'table' then
      return torch.LongStorage(arg[1]), nil
   elseif narg == 1 and type(arg[1]) == 'torch.LongStorage' then
      return arg[1], nil
   elseif narg == 2 and type(arg[1]) == 'number' and type(arg[2]) == 'number' then
      return torch.LongStorage{arg[1], arg[2]}, nil
   elseif narg == 2 and type(arg[1]) == 'table' and type(arg[2]) == 'table' then
      return torch.LongStorage(arg[1]), torch.LongStorage(arg[2])
   elseif narg == 2 and type(arg[1]) == 'torch.LongStorage' and type(arg[2]) == 'torch.LongStorage' then
      return arg[1], arg[2]
   elseif narg == 3 and type(arg[1]) == 'number' and type(arg[2]) == 'number' and type(arg[3]) == 'number' then
      return torch.LongStorage{arg[1], arg[2], arg[3]}, nil
   elseif narg == 4 and type(arg[1]) == 'number' and type(arg[2]) == 'number' and type(arg[3]) == 'number' and type(arg[4]) == 'number' then
      return torch.LongStorage{arg[1], arg[2], arg[3], arg[4]}, nil
   else
      error('invalid arguments')
   end
end

function Tensor.new(...)
   local arg = {...}
   local storage, storageOffset, size, stride = readtensorsizestride(arg)

   local self = rawInit()
   
   if size and stride then
      assert(size.__size == stride.__size, 'inconsistent size')
   end

   rawSet(self,
          storage,
          storageOffset,
          size and size.__size or (stride and stride.__size or 0),
          size and size.__data or nil,
          stride and stride.__data or nil)

   return self
end

function Tensor:set(src)
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

function Tensor:narrow(...)
   local arg = {...}
   local narg = #arg
   local src, dimension, firstIndex, size
   if narg == 3 then
      src, dimension, firstIndex, size = self, arg[1]-1, arg[2]-1, arg[3]
   elseif narg == 4 then
      src, dimension, firstIndex, size = arg[1], arg[2]-1, arg[3]-1, arg[4]
   else
      error('invalid arguments')
   end

   assert(dimension >= 0 and dimension < src.__nDimension, 'out of range')
   assert(firstIndex >= 0 and firstIndex < src.__size[dimension], 'out of range')
   assert(size > 0 and firstIndex+size <= src.__size[dimension], 'out of range')
  
   self:set(src)
   
   if firstIndex > 0 then
      self.__storageOffset = self.__storageOffset + firstIndex*self.__stride[dimension];
   end
   self.__size[dimension] = size

   return self
end

function Tensor:select(...)
   local arg = {...}
   local narg = #arg
   local src, dimension, sliceIndex
   if narg == 2 then
      src, dimension, sliceIndex = self, arg[1]-1, arg[2]-1
   elseif narg == 3 then
      src, dimension, sliceIndex, size = arg[1], arg[2]-1, arg[3]-1
   else
      error('invalid arguments')
   end

   assert(dimension >= 0 and dimension < src.__nDimension, 'out of range')
   assert(sliceIndex >= 0 and sliceIndex < src.__size[dimension], 'out of range')

   if self.__nDimension == 1 then
      return tonumber( (self.__storage.__data + self.__storageOffset)[sliceIndex*self.__stride[0]] )
   else
      self:narrow(self, src, dimension, sliceIndex, 1)
      for d=dimension,self.__nDimension-2 do
         self.__size[d] = self.__size[d+1]
         self.__stride[d] = self.__stride[d+1]
      end
      self.__nDimension = self.__nDimension -1
   end

   return self
end

function Tensor:transpose(...)
   local arg = {...}
   local narg = #arg
   local src, dimension1, dimension2
   if narg == 2 then
      src, dimension1, dimension2 = self, arg[1]-1, arg[2]-1
   elseif narg == 3 then
      src, dimension1, dimension2 = arg[1], arg[2]-1, arg[3]-1
   else
      error('invalid arguments')
   end

   assert(dimension1 >= 0 and dimension1 < src.__nDimension, 'out of range')
   assert(dimension2 >= 0 and dimension2 < src.__nDimension, 'out of range')

   self:set(src)

   if dimension1 == dimension2 then
      return self
   end
 
   local z = self.__stride[dimension1]
   self.__stride[dimension1] = self.__stride[dimension2]
   self.__stride[dimension2] = z
   z = self.__size[dimension1]
   self.__size[dimension1] = self.__size[dimension2]
   self.__size[dimension2] = z

   return self
end

function Tensor:unfold(...)
   local arg = {...}
   local narg = #arg
   local src, dimension, size, step
   if narg == 3 then
      src, dimension, size, step = self, arg[1]-1, arg[2], arg[3]
   elseif narg == 4 then
      src, dimension, size, step = arg[1], arg[2]-1, arg[3], arg[4]
   else
      error('invalid arguments')
   end

   assert(src.__nDimension > 0, "cannot unfold an empty tensor")
   assert(dimension < src.__nDimension, "out of range")
   assert(size <= src.__size[dimension], "out of range")
   assert(step > 0, "invalid step")

   self:set(src)

   local newSize = ffi.new("long[?]", self.__nDimension+1)
   local newStride = ffi.new("long[?]", self.__nDimension+1)

   newSize[self.__nDimension] = size
   newStride[self.__nDimension] = self.__stride[dimension]
   for d=0,self.__nDimension-1 do
      if d == dimension then
         newSize[d] = math.floor((self.__size[d] - size) / step) + 1
         newStride[d] = step*self.__stride[d]
      else
         newSize[d] = self.__size[d]
         newStride[d] = self.__stride[d]
      end
   end

   self.__size = newSize
   self.__stride = newStride
   self.__nDimension = self.__nDimension + 1

   return self
end

function Tensor:squeeze(src)
   src = src or self
   self:set(src)

   -- return nothing if tensor is empty!
   if self.__nDimension == 0 then
      return
   end

   local ndim = 0
   for d=0,src.__nDimension-1 do
      if src.__size[d] ~= 1 then
         if d ~= ndim then
            self.__size[ndim] = src.__size[d]
            self.__stride[ndim] = src.__stride[d]
         end
         ndim = ndim + 1
      end
   end

   --- handle 0-dimension tensors
   if ndim == 0 then
      return tonumber( (self.__storage.__data + self.__storageOffset)[0] )
   end
   self.__nDimension = ndim

   return self
end

function Tensor:squeeze1d(...)
   local arg = {...}
   local narg = #arg
   local src, dimension
   if narg == 1 then
      src, dimension = self, arg[1]-1
   else
      src, dimension = arg[1], arg[2]-1
   end

  assert(dimension < src.__nDimension, "dimension out of range")

  self:set(src)

  if src.__size[dimension] == 1 and src.__nDimension > 1 then
     for d=dimension,self.__nDimension-2 do
        self.__size[d] = self.__size[d+1]
        self.__stride[d] = self.__stride[d+1]
     end
     self.__nDimension = self.__nDimension - 1
  end

  return self
end

function Tensor:isContiguous()
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

function Tensor:nElement()
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

mt = {__index=function(self, k)
                 if type(k) == 'number' then
                    if self.__nDimension == 1 then
                       return tonumber(TH.THTensor_get1d(self, k-1))
                    elseif self.__nDimension > 1 then
                       local t = TH.THTensor_newSelect(self, 0, k-1)
                       ffi.gc(t, function(self)
                                    print('freeing tensor -- []')
                                    TH.THTensor_free(self)
                                 end)
                       return t
                    else
                       error('empty tensor')
                    end
                 else
                    return Tensor[k]
                 end
              end}


torch.Tensor = {}
setmetatable(torch.Tensor, {__index=Tensor,
                            __metatable=Tensor,
                            __newindex=Tensor,
                            __call=function(self, ...)
                                      return Tensor.new(...)
                                   end})
