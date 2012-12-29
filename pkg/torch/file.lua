local argcheck = require 'torch.argcheck'
local ffi = require 'ffi'

local File = torch.class('torch.File')

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
   {Type="Byte", ctype=ffi.typeof("unsigned char[1]"), ptype="%hhu", Storage=torch.ByteStorage, sizeof=ffi.sizeof('unsigned char')},
   {Type="Char", ctype=ffi.typeof("char[1]"), ptype="%hhd", Storage=torch.CharStorage, sizeof=ffi.sizeof('char')},
   {Type="Short", ctype=ffi.typeof("short[1]"), ptype="%hd", Storage=torch.ShortStorage, sizeof=ffi.sizeof('short')},
   {Type="Int", ctype=ffi.typeof("int[1]"), ptype="%d", Storage=torch.IntStorage, sizeof=ffi.sizeof('int')},
   {Type="Long", ctype=ffi.typeof("long[1]"), ptype="%ld", Storage=torch.LongStorage, sizeof=ffi.sizeof('long')},
   {Type="Float", ctype=ffi.typeof("float[1]"), ptype="%g", Storage=torch.FloatStorage, sizeof=ffi.sizeof('float')},
   {Type="Double", ctype=ffi.typeof("double[1]"), ptype="%lg", Storage=torch.DoubleStorage, sizeof=ffi.sizeof('double')},
}

for _, ttype in ipairs(types) do

   local function write(self, cdata, size)
      local n
      if self.__isBinary then
         n = self:__write(cdata, ttype.sizeof, size)
      else
         n = self:__printf(ttype.ptype, cdata, size)
         if self.__isAutoSpacing and n > 0 then
            local ret = '\n'
            self:__write(ffi.cast('char*', ret), 1, 1)
         end
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
         n = self:__read(cdata, ttype.sizeof, size)
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

local function readchars(self, n)
   local buffsize = 1024
   local buffer = ffi.cast('char*', ffi.C.malloc(buffsize))
   local size = 0
   while true do
      assert(buffer ~= nil, 'out of memory')
      local rlen = math.min(buffsize, n)
      local nr = self:__read(buffer+size, 1, rlen)
      if nr > 0 then
         size = size + nr
         n = n - nr
      end
      if n > 0 and nr == rlen then
         buffer = ffi.cast('char*', ffi.C.realloc(buffer, size+buffsize))
      else
         break
      end
   end
   if size > 0 then
      local str = ffi.string(buffer, size)
      ffi.C.free(buffer)
      return str
   else
      ffi.C.free(buffer)
   end
end

File.read =
   argcheck(
   {{name="self", type="torch.File"},
    {name="format", type="string"}},
   function(self, format)
      if format:match('^%*n') then
         local p = ffi.new('double[1]')
         local n = self:__scanf("%lf", p, 1)
         if n == 1 then
            return p[0]
         end
      elseif format:match('^%*a') then
         return readchars(self, math.huge)
      elseif format:match('^%*l') then
         return self:__gets()
      elseif tonumber(format) then
         return readchars(self, tonumber(format))
      else
         error('invalid format')
      end
   end
)

File.write =
   argcheck(
   {{name="self", type="torch.File"},
    {name="value", type="string"}},
   function(self, value)
      self:__write(ffi.cast('char*', value), 1, #value)
   end,

   {{name="self", type="torch.File"},
    {name="value", type="number"}},
   function(self, value)
      local p = ffi.new('double[1]')
      p[0] = value
      self:__printf("%lf", p, 1)
   end
)

File.new =
   function()
      error('virtual class')
   end

torch.File = torch.constructor(File)
