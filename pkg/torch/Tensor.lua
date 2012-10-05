local Tensor = {__typename="torch.Tensor"}
local mt

local function rawInit(self)
   self.__storage = nil
   self.__storageOffset = 0
   self.__size = nil
   self.__stride = nil
   self.__nDimension = 0
   self.__flag = 0
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
         if self.__storage == nil then
            self.__storage = torch.Storage() --:retain()
         end
         if totalSize+self.__storageOffset > self.__storage.__size then
            self.__storage:resize(totalSize+self.__storageOffset)
         end
      end
   else
      self.__nDimension = 0
   end
end

local function rawSet(self, storage, storageOffset, nDimension, size, stride)
   print('rawSet', self, storage, storageOffset, nDimension, size, stride)
  -- storage
  if self.__storage ~= storage then
     if self.__storage ~= nil then
--X        self.__storage:free()
     end

    if storage ~= nil then
       self.__storage = storage
       self.__storage:retain()
    else
--       self.__storage = nil
    end
 end

 -- storageOffset
 assert(storageOffset >= 0, "Tensor: invalid storage offset")
 self.__storageOffset = storageOffset

 -- size and stride
 rawResize(self, nDimension, size, stride)
end

-- access methods
function Tensor:storage()
   -- DEBUG: check this out
  return self.__storage
end

function Tensor:storageOffset()
  return self.__storageOffset
end

function Tensor:nDimension()
  return self.__nDimension;
end

function Tensor:dim()
  return self.__nDimension;
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

   print('READ', storage, storageOffset, size, stride)

   local self = {}--ffi.new("Tensor")
   setmetatable(self, mt)
   rawInit(self)
--    ffi.gc(self, function(self)
--                    Tensor.free(self)
--                 end)

   
   if size and stride then
      assert(size.__size == stride.__size, 'inconsistent size')
   end

   rawSet(self,
          storage,
          storageOffset,
          tonumber(size and size.__size or (stride and stride.__size or 0)),
          size and size.__data or nil,
          stride and stride.__data or nil)

   return self
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
                    return mt[k]
                 end
              end}


torch.Tensor = {}
setmetatable(torch.Tensor, {__index=Tensor,
                            __metatable=Tensor,
                            __newindex=Tensor,
                            __call=function(self, ...)
                                      return Tensor.new(...)
                                   end})
