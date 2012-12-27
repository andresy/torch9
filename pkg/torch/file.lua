local argcheck = require 'torch.argcheck'
local File = {__typename="torch.File"}
local ffi = require 'ffi'

-- should initialize basic variables (__isBinary, __isAutoSpacing... here in a basic constructor)

File.isQuiet =
   argcheck(
   {{name="self", type="torch.File"}},
   function(self)
      return self.__isQuiet
   end
)

File.isReadable =
   argcheck(
   {{name="self", type="torch.File"}},
   function(self)
      return self.__isReadable
   end
)

File.isWritable =
   argcheck(
   {{name="self", type="torch.File"}},
   function(self)
      return self.__isWritable
   end
)

File.isAutoSpacing =
   argcheck(
   {{name="self", type="torch.File"}},
   function(self)
      return self.__isAutoSpacing
   end
)

File.hasError =
   argcheck(
   {{name="self", type="torch.File"}},
   function(self)
      return self.__hasError
   end
)

File.isBinary =
   argcheck(
   {{name="self", type="torch.File"}},
   function(self)
      return self.__isBinary
   end
)

local types = {
   {Type="Byte", ctype=ffi.typeof("unsigned char[1]"), ptype="%uc", Storage=torch.ByteStorage},
   {Type="Char", ctype=ffi.typeof("char[1]"), ptype="%c", Storage=torch.CharStorage},
   {Type="Short", ctype=ffi.typeof("short[1]"), ptype="%hd", Storage=torch.ShortStorage},
   {Type="Int", ctype=ffi.typeof("int[1]"), ptype="%d", Storage=torch.IntStorage},
   {Type="Long", ctype=ffi.typeof("long[1]"), ptype="%ld", Storage=torch.LongStorage},
   {Type="Float", ctype=ffi.typeof("float[1]"), ptype="%f", Storage=torch.FloatStorage},
   {Type="Double", ctype=ffi.typeof("double[1]"), ptype="%lf", Storage=torch.DoubleStorage},
}

for _, ttype in ipairs(types) do

   local function write(self, cdata, size)
      local n
      if self.__isBinary then
         n = self:__write(cdata, size)
      else
         n = self:__printf(ttype.ptype, cdata, size)
      end
      if n ~= size then
         if not self.__isQuiet then
            error(string.format('wrote %d values instead of %d', n, size))
         end
      end
      return n
   end

   File['write' .. ttype.Type] =
      argcheck(
      {{name="self", type="torch.File"},
       {name="storage", type="torch." .. ttype.Type .. "Storage"}},
      function(self, storage)
         return write(self, storage.__data, storage.__size)
      end,

      {{name="self", type="torch.File"},
       {name="value", type="number"}},
      function(self, value)
         local p = ttype.ctype(value)
         return write(self, p, 1)
      end
   )

   local function read(self, cdata, size)
      local n
      if self.__isBinary then
         n = self:__read(cdata, size)
      else
         n = self:__scanf(ttype.ptype, cdata, size)
      end
      if n ~= size then
         if not self.__isQuiet then
            error(string.format('read %d values instead of %d', n, size))
         end
      end
      return n
   end

   File['read' .. ttype.Type] =
      argcheck(
      {{name="self", type="torch.File"},
       {name="storage", type="torch." .. ttype.Type .. "Storage"}},
      function(self, storage)
         return read(self, storage.__data, storage.__size)
      end,

      {{name="self", type="torch.File"},
       {name="size", type="number"}},
      function(self, size)
         local storage = ttype.Storage(size)
         local n = read(self, storage.__data, storage.__size)
         if n ~= size then
            storage:resize(n)
         end
         return storage
      end,

      {{name="self", type="torch.File"}},
      function(self)
         local p = ttype.ctype()
         local n = read(self, p, 1)
         if n ~= 0 then
            return p[0]
         end
      end
   )
end

File.binary =
   argcheck(
   {{name="self", type="torch.File"}},
   function(self)
      self.__isBinary = true
      return self
   end
)

File.ascii =
   argcheck(
   {{name="self", type="torch.File"}},
   function(self)
      self.__isBinary = false
      return self
   end
)

File.autoSpacing =
   argcheck(
   {{name="self", type="torch.File"}},
   function(self)
      self.__isAutoSpacing = true
      return self
   end
)

File.noAutoSpacing =
   argcheck(
   {{name="self", type="torch.File"}},
   function(self)
      self.__isAutoSpacing = false
      return self
   end
)

File.quiet =
   argcheck(
   {{name="self", type="torch.File"}},
   function(self)
      self.__isQuiet = true
      return self
   end
)

File.pedantic =
   argcheck(
   {{name="self", type="torch.File"}},
   function(self)
      self.__isQuiet = false
      return self
   end
)

File.clearError =
   argcheck(
   {{name="self", type="torch.File"}},
   function(self)
      self.__hasError = false
      return self
   end
)
    
torch.File = {}
setmetatable(torch.File, {__index=File,
                          __metatable=File,
                          __newindex=File,
                          __call=function(self, ...)
                                    error('virtual class')
                                 end})
