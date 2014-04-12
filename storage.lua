-- todo:
-- RealRealStorage (plus pratique, ne serait-que pour THRealStorage, RealStorage...)
-- changer le script de template accordingly
-- make TH func return storage, tensors... this would avoid extra lua func redirections
-- prefixer les champs des structures (.data en .__data...) dans les declarations FFI

local display = require 'torch.display'
local argcheck = require 'argcheck'
local torch = require 'torch.env'
local class = require 'class'
local ffi = require 'ffi'
local C = require 'torch.TH'

local RealStorage = class('torch.RealStorage', nil, ffi.typeof('THRealStorage&'))
torch.RealStorage = RealStorage

RealStorage.__factory =
   function()
      local self =  C.THRealStorage_new()[0]
      ffi.gc(self, C.THRealStorage_free)
      return self
   end

RealStorage.new = argcheck{
   {name="size", type="number", default=0},
   call =
      function(size)
         local self = C.THRealStorage_newWithSize(size)[0]
         ffi.gc(self, C.THRealStorage_free)
         return self
      end
}

argcheck{
   {name="table", type="table"},
   chain = RealStorage.new,
   call =
      function(self, tbl)
         local size = #tbl
         self = C.THRealStorage_newWithSize(size)[0]
         ffi.gc(self, C.THRealStorage_free)
         for i=1,size do
            self.__data[i-1] = tbl[i]
         end
         return self
      end
}

RealStorage.fill = argcheck{
   {name="self", type="torch.RealStorage"},
   {name="value", type="number"},
   call = C.THRealStorage_fill
}

RealStorage.size = argcheck{
   {name="self", type="torch.RealStorage"},
   call =
      function(self)
         return tonumber(self.__size)
      end
}

RealStorage.resize = argcheck{
   {name="self", type="torch.RealStorage"},
   {name="size", type="number"},
   call = C.THRealStorage_resize
}

RealStorage.rawCopy = argcheck{
   {name="self", type="torch.RealStorage"},
   {name="data", type="cdata"},
   call =
      function(self, data)
         ffi.copy(self.__data, data, ffi.sizeof('real')*self.__size)
         return self
      end
}

RealStorage.totable = argcheck{
   {name="self", type="torch.RealStorage"},
   call =
   function(self)
      local tbl = {}
      for i=1,self.__size do
         tbl[i] = self.__data[i-1]
      end
      return tbl
   end
}

if "RealStorage" == "CharRealStorage" or "RealStorage" == "ByteRealStorage" then
   RealStorage.string = argcheck{
         {name="self", type="torch.RealStorage"},
         call =
            function(self)
               return ffi.string(self.__data, self.__size)
            end
      }

   argcheck{
      {name="self", type="torch.RealStorage"},
      {name="data", type="string"},
      chain = RealStorage.string,
      call =
         function(self, data)
            self:resize(#data)
            C.THRealStorage_rawCopy(self, ffi.cast('real*', data))
            return self
         end
   }
end

RealStorage.copy = argcheck{
   {name="self", type='torch.RealStorage'},
   {name="src", type='torch.RealStorage'},
   call = C.THRealStorage_copy
}

argcheck{
   {name="self", type='torch.RealStorage'},
   {name="src", type='torch.ByteStorage'},
   chain = RealStorage.copy,
   call = C.THRealStorage_copyByte
}

argcheck{
   {name="self", type='torch.RealStorage'},
    {name="src", type='torch.CharStorage'},
   chain = RealStorage.copy,
   call = C.THRealStorage_copyChar
}

argcheck{
   {name="self", type='torch.RealStorage'},
    {name="src", type='torch.ShortStorage'},
   chain = RealStorage.copy,
   call = C.THRealStorage_copyShort
}

argcheck{
   {name="self", type='torch.RealStorage'},
    {name="src", type='torch.IntStorage'},
   chain = RealStorage.copy,
   call = C.THRealStorage_copyInt
}

argcheck{
   {name="self", type='torch.RealStorage'},
    {name="src", type='torch.LongStorage'},
   chain = RealStorage.copy,
   call = C.THRealStorage_copyLong
}

argcheck{
   {name="self", type='torch.RealStorage'},
    {name="src", type='torch.FloatStorage'},
   chain = RealStorage.copy,
   call = C.THRealStorage_copyFloat
}

argcheck{
   {name="self", type='torch.RealStorage'},
    {name="src", type='torch.DoubleStorage'},
   chain = RealStorage.copy,
   call = C.THRealStorage_copyDouble
}

function RealStorage:__index(k)
--   print('REQ', k)
   if type(k) == 'number' then
      if k > 0 and k <= tonumber(self.__size) then
         return tonumber(self.__data[k-1])
      else
         error('index out of bounds')
      end
   else
      return RealStorage[k]
   end
end

function RealStorage:__newindex(k, v)
   if type(k) == 'number' then
      if k > 0 and k <= self.__size then
         self.__data[k-1] = v
      else
         error('index out of bounds')
      end
   else
      rawset(self, k, v)
   end
end

function RealStorage:__len()
   return self.__size
end

function RealStorage:write(file)
   file:writeLong(self.__size)
   file:writeRaw('real', self.__data, self.__size)
end

function RealStorage:read(file)
   local size = file:readLong()
   rawInitWithSize(self, size)
   file:readRaw('real', self.__data, self.__size)
end

RealStorage.__tostring = display.storage

ffi.metatype('THRealStorage', getmetatable(RealStorage))
