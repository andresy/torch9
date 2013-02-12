local argcheck = require 'torch.argcheck'
local ffi = require 'ffi'

local iotypes = {
   byte   = {ctype='unsigned char', format='%hhu'},
   char   = {ctype='char', format='%hhd'},
   short  = {ctype='short', format='%hd'},
   int    = {ctype='int', format='%d'},
   long   = {ctype='long', format='%ld'},
   float  = {ctype='float', format='%g'},
   double = {ctype='double', format='%lg'},
}

for ioluatype, iotype in pairs(iotypes) do
   iotype.type = ioluatype
   iotype.Type = string.upper(string.sub(ioluatype, 1, 1)) .. string.sub(ioluatype, 2, -1)
   iotype.Storage = torch[iotype.Type .. 'Storage']
   iotype.sizeof = ffi.sizeof(iotype.ctype)
   iotype.ffictype = ffi.typeof(iotype.ctype)
   iotype.ffipval = ffi.typeof(iotype.ctype .. '[1]')

   iotype.write =
      function(self, cdata, size)
         local n
         if self.__isBinary then
            n = self:__write(cdata, iotype.sizeof, size)
         else
            n = self:__printf(iotype.format, cdata, size)
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

   iotype.read =
      function(self, cdata, size)
         local n
         if self.__isBinary then
            n = self:__read(cdata, iotype.sizeof, size)
         else
            n = self:__scanf(iotype.format, cdata, size)
         end
         if n ~= size then
            if not self.__isQuiet then
               error(string.format('read %d values instead of %d', n, size))
            end
         end
         return n
      end
end


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

for _, iotype in pairs(iotypes) do
   File['write' .. iotype.Type] =
      argcheck(
      {{name="self", type="torch.File"},
       {name="storage", type="torch." .. iotype.Type .. "Storage"}},
      function(self, storage)
         return iotype.write(self, storage.__data, storage.__size)
      end,

      {{name="self", type="torch.File"},
       {name="value", type="number"}},
      function(self, value)
         local p = iotype.ffipval(value)
         return iotype.write(self, p, 1)
      end
   )

   File['read' .. iotype.Type] =
      argcheck(
      {{name="self", type="torch.File"},
       {name="storage", type="torch." .. iotype.Type .. "Storage"}},
      function(self, storage)
         return iotype.read(self, storage.__data, storage.__size)
      end,

      {{name="self", type="torch.File"},
       {name="size", type="number"}},
      function(self, size)
         local storage = iotype.Storage(size)
         local n = iotype.read(self, storage.__data, storage.__size)
         if n ~= size then
            storage:resize(n)
         end
         return storage
      end,

      {{name="self", type="torch.File"}},
      function(self)
         local p = iotype.ffipval()
         local n = iotype.read(self, p, 1)
         if n ~= 0 then
            return tonumber(p[0])
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
   {{{name="self", type="torch.File"},
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
})

File.write =
   argcheck(
   {{{name="self", type="torch.File"},
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
})

File.readRaw =
   argcheck(
   {{name="self", type="torch.File"},
    {name="typex", type="string"},
    {name="cdata", type="cdata"},
    {name="size", type="number"}},
   function(self, type, cdata, size)
      assert(iotypes[type], 'unknown type')
      iotypes[type].read(self, cdata, size)
   end
)

File.writeRaw =
   argcheck(
   {{name="self", type="torch.File"},
    {name="typex", type="string"},
    {name="cdata", type="cdata"},
    {name="size", type="number"}},
   function(self, type, cdata, size)
      assert(iotypes[type], 'unknown type')
      iotypes[type].write(self, cdata, size)
   end
)

File.new =
   function()
      error('virtual class')
   end

torch.File = torch.constructor(File)
