local torch = require 'torch'
local argcheck = require 'torch.argcheck'
local display = require 'torch.display'
local ffi = require 'ffi'

local Storage = torch.class('torch.Storage')

local th = ffi.load(paths.concat(paths.install_lua_path,
                                 'torch',
                                 ((jit.os == 'Windows') and '' or 'lib') .. 'maths' ..
                                 ((jit.os == 'Windows') and '.dll' or ((jit.os == 'OSX') and '.dylib' or '.so'))))

local realsz = ffi.sizeof('real')
local realptrct = ffi.typeof('real*')

local function rawInitWithSize(self, size)
   size = size or 0
   if size > 0 then
      self.__data = ffi.cast(realptrct, ffi.C.malloc(realsz*size))
      ffi.gc(self.__data, ffi.C.free)
      self.__size = size
   else
      self.__size = 0
   end
   return self
end

Storage.__init =
   argcheck(
   {{name="self", type="torch.Storage"}},
   rawInitWithSize,

   {{name="self", type="torch.Storage"},
    {name="size", type="number"}},
   rawInitWithSize,

   {{name="self", type="torch.Storage"},
    {name="table", type="table"}},
   function(self, tbl)
      local size = #tbl
      self = rawInitWithSize(self, size)
      for i=1,size do
         self.__data[i-1] = tbl[i]
      end
   end,

   {{name="self", type="torch.Storage"},
    {name="filename", type="string"},
    {name="shared", type="boolean", default=false}},
   function(self, filename, shared)
      error('NYI')
   end
)

Storage.fill =
   argcheck(
   {{name="self", type="torch.Storage"},
    {name="value", type="number"}},
   function(self, value)
      for i=0,self.__size-1 do
         self.__data[i] = value
      end
      return self
   end
)

Storage.size =
   argcheck(
   {{name="self", type="torch.Storage"}},
   function(self)
    return self.__size
   end
)

Storage.resize =
   argcheck(
   {{name="self", type="torch.Storage"},
    {name="size", type="number"}},
   function(self, size)
      if size > 0 and size > self.__size then
         if self.__data then
            ffi.gc(self.__data, nil)
         end
         self.__data = ffi.cast(realptrct,
                                ffi.C.realloc(self.__data, realsz*size)
                             )
         ffi.gc(self.__data, ffi.C.free)
         self.__size = size
      end
      return self
   end
)

Storage.rawCopy =
   argcheck(
   {{name="self", type="torch.Storage"},
    {name="data", type="cdata"}},
   function(self, data)
      ffi.copy(self.__data, data, realsz*self.__size)
      return self
   end
)

if "Storage" == "CharStorage" or "Storage" == "ByteStorage" then
   Storage.string =
      argcheck(
      {{name="self", type="torch.Storage"}},
      function(self)
         return ffi.string(self.__data, self.__size)
      end,

      {{name="self", type="torch.Storage"},
       {name="data", type="string"}},
      function(self, data)
         self:resize(#data)
         ffi.copy(self.__data, ffi.cast('char*', data), self.__size)
         return self
      end
   )
end

Storage.copy =
   argcheck(
   {{name="self", type='torch.Storage'},
    {name="src", type='torch.Storage'}},
   function(self, src)
      assert(self.__size == src.__size, 'size mismatch')
      th.copy_real_real(self.__data, 1, src.__data, 1, self.__size)
   end,

   {{name="self", type='torch.Storage'},
    {name="src", type='torch.ByteStorage'}},
   function(self, src)
      assert(self.__size == src.__size, 'size mismatch')
      th.copy_real_byte(self.__data, 1, src.__data, 1, self.__size)
   end,

   {{name="self", type='torch.Storage'},
    {name="src", type='torch.CharStorage'}},
   function(self, src)
      assert(self.__size == src.__size, 'size mismatch')
      th.copy_real_char(self.__data, 1, src.__data, 1, self.__size)
   end,

   {{name="self", type='torch.Storage'},
    {name="src", type='torch.ShortStorage'}},
   function(self, src)
      assert(self.__size == src.__size, 'size mismatch')
      th.copy_real_short(self.__data, 1, src.__data, 1, self.__size)
   end,

   {{name="self", type='torch.Storage'},
    {name="src", type='torch.IntStorage'}},
   function(self, src)
      assert(self.__size == src.__size, 'size mismatch')
      th.copy_real_int(self.__data, 1, src.__data, 1, self.__size)
   end,

   {{name="self", type='torch.Storage'},
    {name="src", type='torch.LongStorage'}},
   function(self, src)
      assert(self.__size == src.__size, 'size mismatch')
      th.copy_real_long(self.__data, 1, src.__data, 1, self.__size)
   end,

   {{name="self", type='torch.Storage'},
    {name="src", type='torch.FloatStorage'}},
   function(self, src)
      assert(self.__size == src.__size, 'size mismatch')
      th.copy_real_float(self.__data, 1, src.__data, 1, self.__size)
   end,

   {{name="self", type='torch.Storage'},
    {name="src", type='torch.DoubleStorage'}},
   function(self, src)
      assert(self.__size == src.__size, 'size mismatch')
      th.copy_real_double(self.__data, 1, src.__data, 1, self.__size)
   end
)

function Storage:__index(k)
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

torch.Storage = torch.constructor(Storage)
