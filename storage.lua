local display = require 'torch.display'
local argcheck = require 'argcheck'
local torch = require 'torch.env'
local class = require 'class'
local ffi = require 'ffi'
local C = require 'torch.clib'

local Storage = class('torch.Storage')
torch.Storage = Storage

local realsz = ffi.sizeof('real')
local realptrct = ffi.typeof('real*')

local function rawInitWithSize(self, size)
   if size and size > 0 then
      self.__data = ffi.cast(realptrct, ffi.C.malloc(realsz*size))
      self.__size = size
      self.__refcount = 1
      self.__flag = 0
   else
      self.__data = nil
      self.__size = 0
      self.__refcount = 1
      self.__flag = 0
   end
   return self
end

Storage.__init = argcheck{
   {name="self", type="torch.Storage"},
   {name="size", type="number", default=0},
   call =
      rawInitWithSize
}

argcheck{
   {name="self", type="torch.Storage"},
   {name="table", type="table"},
   chain = Storage.__init,
   call =
      function(self, tbl)
         local size = #tbl
         self = rawInitWithSize(self, size)
         for i=1,size do
            self.__data[i-1] = tbl[i]
         end
      end
}

Storage.fill = argcheck{
   {name="self", type="torch.Storage"},
   {name="value", type="number"},
   call =
      function(self, value)
         for i=0,tonumber(self.__size)-1 do
            self.__data[i] = value
         end
         return self
      end
}

Storage.size = argcheck{
   {name="self", type="torch.Storage"},
   call =
      function(self)
         return tonumber(self.__size)
      end
}

Storage.resize = argcheck{
   {name="self", type="torch.Storage"},
   {name="size", type="number"},
   call =
   function(self, size)
      if size > 0 and size > self.__size then
         self.__data = ffi.cast(realptrct,
                                ffi.C.realloc(self.__data, realsz*size)
                             )
         self.__size = size
      end
      return self
   end
}

Storage.rawCopy = argcheck{
   {name="self", type="torch.Storage"},
   {name="data", type="cdata"},
   call =
   function(self, data)
      ffi.copy(self.__data, data, realsz*self.__size)
      return self
   end
}

Storage.totable = argcheck{
   {name="self", type="torch.Storage"},
   call =
   function(self)
      local tbl = {}
      for i=1,self.__size do
         tbl[i] = self.__data[i-1]
      end
      return tbl
   end
}

if "Storage" == "CharStorage" or "Storage" == "ByteStorage" then
   Storage.string = argcheck{
         {name="self", type="torch.Storage"},
         call =
            function(self)
               return ffi.string(self.__data, self.__size)
            end
      }

   argcheck{
      {name="self", type="torch.Storage"},
      {name="data", type="string"},
      chain = Storage.string,
      call =
         function(self, data)
            self:resize(#data)
            ffi.copy(self.__data, ffi.cast('char*', data), self.__size)
            return self
         end
   }
end

Storage.copy = argcheck{
   {name="self", type='torch.Storage'},
   {name="src", type='torch.Storage'},
   call =
      function(self, src)
         assert(self.__size == src.__size, 'size mismatch')
         C.th_copy_real_real(self.__size, src.__data, 1, self.__data, 1)
      end
}

argcheck{
   {name="self", type='torch.Storage'},
   {name="src", type='torch.ByteStorage'},
   chain = Storage.copy,
   call =
      function(self, src)
         assert(self.__size == src.__size, 'size mismatch')
         C.th_copy_real_byte(self.__size, src.__data, 1, self.__data, 1)
      end
}

argcheck{
   {name="self", type='torch.Storage'},
    {name="src", type='torch.CharStorage'},
   chain = Storage.copy,
   call =
   function(self, src)
      assert(self.__size == src.__size, 'size mismatch')
      C.th_copy_real_char(self.__size, src.__data, 1, self.__data, 1)
   end
}

argcheck{
   {name="self", type='torch.Storage'},
    {name="src", type='torch.ShortStorage'},
   chain = Storage.copy,
   call =
   function(self, src)
      assert(self.__size == src.__size, 'size mismatch')
      C.th_copy_real_short(self.__size, src.C.__data, 1, self.__data, 1)
   end
}

argcheck{
   {name="self", type='torch.Storage'},
    {name="src", type='torch.IntStorage'},
   chain = Storage.copy,
   call =
   function(self, src)
      assert(self.__size == src.C.__size, 'size mismatch')
      C.th_copy_real_int(self.__size, src.__data, 1, self.__data, 1)
   end
}

argcheck{
   {name="self", type='torch.Storage'},
    {name="src", type='torch.LongStorage'},
   chain = Storage.copy,
   call =
   function(self, src)
      assert(self.__size == src.__size, 'size mismatch')
      C.th_copy_real_long(self.__size, src.__data, 1, self.__data, 1)
   end
}

argcheck{
   {name="self", type='torch.Storage'},
    {name="src", type='torch.FloatStorage'},
   chain = Storage.copy,
   call =
   function(self, src)
      assert(self.__size == src.__size, 'size mismatch')
      C.th_copy_real_float(self.__size, src.__data, 1, self.__data, 1)
   end
}

argcheck{
   {name="self", type='torch.Storage'},
    {name="src", type='torch.DoubleStorage'},
   chain = Storage.copy,
   call =
   function(self, src)
      assert(self.__size == src.__size, 'size mismatch')
      C.th_copy_real_double(self.__size, src.__data, 1, self.__data, 1)
   end
}

function Storage:__index(k)
--   print('REQ', k)
   if type(k) == 'number' then
      if k > 0 and k <= self.__size then
         return tonumber(self.__data[k-1])
      else
         error('index out of bounds')
      end
   else
      return Storage[k]
   end
end

function Storage:__newindex(k, v)
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

function Storage:__len()
   return self.__size
end

function Storage:write(file)
   file:writeLong(self.__size)
   file:writeRaw('real', self.__data, self.__size)
end

function Storage:read(file)
   local size = file:readLong()
   rawInitWithSize(self, size)
   file:readRaw('real', self.__data, self.__size)
end

Storage.__tostring = display.storage
