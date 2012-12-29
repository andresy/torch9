local argcheck = require 'torch.argcheck'

ffi = require 'ffi'

ffi.cdef[[
 int snprintf(char *restrict s, size_t n, const char *restrict format, ...);
 int sscanf(const char *restrict s, const char *restrict format, ...);
]]

local MemoryFile = torch.class('torch.MemoryFile', 'torch.File')

local function grow(self, size)
   if self.__position + size + 1 >= self.__buffersize then
      local gsz = math.max(self.__position + size + 1, -- count trailing '\0'
                           self.__growsize + size)

      if self.__buffer then
         ffi.gc(self.__buffer, nil)
         self.__buffer = ffi.cast('char*', ffi.C.realloc(self.__buffer, gsz))
         ffi.gc(self.__buffer, ffi.C.free)
      else
         self.__buffer = ffi.cast('char*', ffi.C.malloc(gsz))
         ffi.gc(self.__buffer, ffi.C.free)
      end

      assert(self.__buffer ~= nil, 'out of memory')
      self.__buffersize = gsz
      self.__buffer[gsz-1] = 0
   end
end

MemoryFile.isOpened =
   argcheck(
   {{name="self", type="torch.MemoryFile"}},
   function(self)
      assert(self.__buffer, 'attempt to use a closed file')
      return self.__buffer ~= nil
   end
)

MemoryFile.synchronize =
   argcheck(
   {{name="self", type="torch.MemoryFile"}},
   function(self)
   end
)

MemoryFile.seek =
   argcheck(
   {{name="self", type="torch.MemoryFile"},
    {name="position", type="number"}},
   function(self, position)
      assert(self.__buffer, 'attempt to use a closed file')
      if position < 0 or position >= self.__size then
         self.__hasError = 1
         if not self.__isQuiet then
            error('unable to seek in file')
         end
      end
      self.__position = position
      return self
   end
)

MemoryFile.seekEnd =
   argcheck(
   {{name="self", type="torch.MemoryFile"}},
   function(self)
      assert(self.__buffer, 'attempt to use a closed file')
      self.__position = self.__size
      return self
   end
)

MemoryFile.position =
   argcheck(
   {{name="self", type="torch.MemoryFile"}},
   function(self)
      assert(self.__buffer, 'attempt to use a closed file')
      return self.__position
   end
)

MemoryFile.close =
   argcheck(
   {{name="self", type="torch.MemoryFile"}},
   function(self)
      assert(self.__buffer, 'attempt to use a closed file')
      ffi.gc(self.__buffer, nil)
      ffi.C.free(self.__buffer)
      self.__buffer = nil
      return self
   end
)

MemoryFile.__write =
   argcheck(
   {{name="self", type="torch.MemoryFile"},
    {name="data", type="cdata"},
    {name="elemsize", type="number"},
    {name="size", type="number"}},
   function(self, data, elemsize, size)
      assert(self.__buffer, 'attempt to write in a closed file')
      assert(self.__isWritable, 'read-only file')
      grow(self, size*elemsize)
      ffi.copy(self.__buffer+self.__position, 
               data,
               size*elemsize)
      self.__position = self.__position + size*elemsize
      self.__size = math.max(self.__position, self.__size)
      self.__buffer[self.__size] = 0
      return size
   end
)

MemoryFile.__read =
   argcheck(
   {{name="self", type="torch.MemoryFile"},
    {name="data", type="cdata"},
    {name="elemsize", type="number"},
    {name="size", type="number"}},
   function(self, data, elemsize, size)
      assert(self.__buffer, 'attempt to write in a closed file')
      assert(self.__isReadable, 'write-only file')
      local n = math.min(math.floor((self.__size-self.__position)/elemsize), size)
      if n > 0 then
         ffi.copy(data, self.__buffer+self.__position, n*elemsize)
         self.__position = self.__position + n
      end
      return n
   end
)

local format2cast = {
   ['%hhu'] = ffi.typeof('unsigned char'),
   ['%hhd'] = ffi.typeof('char'),
   ['%hd'] = ffi.typeof('short'),
   ['%d'] = ffi.typeof('int'),
   ['%ld'] = ffi.typeof('long'),
   ['%g'] = ffi.typeof('float'),
   ['%lg'] = ffi.typeof('double'),
}

MemoryFile.__printf =
   argcheck(
   {{name="self", type="torch.MemoryFile"},
    {name="format", type="string"},
    {name="data", type="cdata"},
    {name="size", type="number"}},
   function(self, format, data, size)
      assert(self.__buffer, 'attempt to write in a closed file')
      assert(self.__isWritable, 'read-only file')
      local cast = format2cast[format]
      for i=0,size-1 do
         repeat
            local szm = self.__buffersize-self.__position
            local szw = ffi.C.snprintf(self.__buffer+self.__position,
                                       szm,
                                       format,
                                       cast(data[i]))
            
            if szm <= szw then
               grow(self, szw)
            else
               self.__position = self.__position + szw
               self.__size = math.max(self.__position, self.__size)
               self.__buffer[self.__size] = 0
            end
         until szm > szw
      end
      return size
   end
)

MemoryFile.__scanf =
   argcheck(
   {{name="self", type="torch.MemoryFile"},
    {name="format", type="string"},
    {name="data", type="cdata"},
    {name="size", type="number"}},
   function(self, format, data, size)
      assert(self.__buffer, 'attempt to write in a closed file')
      assert(self.__isReadable, 'write-only file')
      format = format .. "%n"
      local p = ffi.new('int[1]')
      local n = 0
      for i=0,size-1 do
         local ret = ffi.C.sscanf(self.__buffer+self.__position, format, data+i, p)
         if ret <= 0 then
            break
         else
            self.__position = self.__position + tonumber(p[0])
            n = n + 1
         end
      end
      return n
   end
)

MemoryFile.__gets =
   argcheck(
   {{name="self", type="torch.MemoryFile"}},
   function(self)
      assert(self.__buffer, 'attempt to write in a closed file')
      assert(self.__isReadable, 'write-only file')
      local size = self.__size-self.__position
      local buffer = self.__buffer + self.__position
      local ret = string.byte('\n')
      local eof = (size == 0)
      for i=0,size-1 do
         if buffer[i] == ret then
            size = i + 1
            break
         end
      end
      
      self.__position = self.__position + size
      if buffer[size-1] == ret then
         size = size - 1
      end

      if not eof then
         local str = ffi.string(buffer, size)
         return str
      end
   end
)

MemoryFile.new =
   argcheck(
   {{name="mode", type="string", default='rw'},
    {name="quiet", type="boolean", default=false}},
   function(mode, quiet)
      assert(mode == 'r' or mode == 'w' or mode == 'rw', 'invalid mode (r, w or rw expected)')

      self = MemoryFile.__init()
      self.__growsize = 1024
      self.__buffersize = 0
      self.__position = 0
      self.__size = 0
      grow(self, 0)

      self.__isQuiet = quiet
      self.__isReadable = (mode == 'r') or (mode == 'rw')
      self.__isWritable = (mode == 'w') or (mode == 'rw')
      self.__isBinary = false
      self.__isAutoSpacing = true
      self.__hasError = false

      return self
   end
)

torch.MemoryFile = torch.constructor(MemoryFile)
