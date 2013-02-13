local argcheck = require 'torch.argcheck'

ffi = require 'ffi'

ffi.cdef[[
      typedef struct FILE FILE;
      FILE* fopen(const char *restrict filename, const char *restrict mode);
      size_t fread(void *restrict ptr, size_t size, size_t nitems, FILE *restrict stream);
      size_t fwrite(const void *restrict ptr, size_t size, size_t nitems, FILE *restrict stream);
      int fclose(FILE *stream);
      int fseek(FILE *stream, long offset, int whence);
      long ftell(FILE *stream);
      int fflush(FILE *stream);
      int fprintf(FILE *restrict stream, const char *restrict format, ...);
      int fscanf(FILE *restrict stream, const char *restrict format, ...);
      char *fgets(char *restrict s, int n, FILE *restrict stream);
      size_t strlen(const char *s);
      int fgetc(FILE *stream);
      int ungetc(int c, FILE *stream);
]]

local DiskFile = torch.class('torch.DiskFile', 'torch.File')
DiskFile.SEEK_SET = 0
DiskFile.SEEK_END = 2

local function reversememory(dst, src, blocksize, n)
   local halfblocksize = blocksize/2
   local charsrc = ffi.cast('char*', src)
   local chardst = ffi.cast('char*', dst)
   for b=0,n-1 do
      for i=0,halfblocksize-1 do
         local z = charsrc[i]
         chardst[i] = charsrc[blocksize-1-i]
         chardst[blocksize-1-i] = z
      end
      charsrc = charsrc + blocksize
      chardst = chardst + blocksize
   end
end

DiskFile.name =
   argcheck(
   {{name="self", type="torch.DiskFile"}},
   function(self)
      assert(self.__handle, 'attempt to use a closed file')
      return self.__name
   end
)

DiskFile.isOpened =
   argcheck(
   {{name="self", type="torch.DiskFile"}},
   function(self)
      assert(self.__handle, 'attempt to use a closed file')
      return self.__handle ~= nil
   end
)

DiskFile.synchronize =
   argcheck(
   {{name="self", type="torch.DiskFile"}},
   function(self)
      assert(self.__handle, 'attempt to use a closed file')
      ffi.C.fflush(self.__handle)
      return self
   end
)

DiskFile.seek =
   argcheck(
   {{name="self", type="torch.DiskFile"},
    {name="position", type="number"}},
   function(self, position)
      assert(self.__handle, 'attempt to use a closed file')
      if ffi.C.fseek(self.__handle, position, self.SEEK_SET) < 0 then
         self.__hasError = 1
         if not self.__isQuiet then
            error('unable to seek in file')
         end
      end
      return self
   end
)

DiskFile.seekEnd =
   argcheck(
   {{name="self", type="torch.DiskFile"}},
   function(self)
      assert(self.__handle, 'attempt to use a closed file')
      if ffi.C.fseek(self.__handle, 0, self.SEEK_END) < 0 then
         self.__hasError = 1
         if not self.__isQuiet then
            error('unable to seek in file')
         end
      end
      return self
   end
)

DiskFile.position =
   argcheck(
   {{name="self", type="torch.DiskFile"}},
   function(self)
      assert(self.__handle, 'attempt to use a closed file')
      local position = tonumber(ffi.C.ftell(self.__handle))
      if position < 0 then
         self.__hasError = 1
         if not self.__isQuiet then
            error('unable to get position in file')
         end
      end
      return position
   end
)

DiskFile.close =
   argcheck(
   {{name="self", type="torch.DiskFile"}},
   function(self)
      assert(self.__handle, 'attempt to use a closed file')
      ffi.gc(self.__handle, nil)
      ffi.C.fclose(self.__handle)
      self.__handle = nil
      return self
   end
)

DiskFile.__write =
   argcheck(
   {{name="self", type="torch.DiskFile"},
    {name="data", type="cdata"},
    {name="elemsize", type="number"},
    {name="size", type="number"}},
   function(self, data, elemsize, size)
      assert(self.__handle, 'attempt to write in a closed file')
      assert(self.__isWritable, 'read-only file')
      if self.__isNativeEncoding or elemsize == 1 then
         return tonumber(ffi.C.fwrite(data, elemsize, size, self.__handle))
      else
         local buffer = ffi.C.malloc(elemsize*size)
         assert(buffer ~= nil, 'out of memory')
         reversememory(buffer, data, elemsize, size)
         local n = tonumber(ffi.C.fwrite(buffer, elemsize, size, self.__handle))
         ffi.C.free(buffer)
         return n
      end
   end
)

DiskFile.isLittleEndianCPU = 
   argcheck(
   {},
   function()
      local x = ffi.new('int[1]', 7)
      local ptr = ffi.cast('char*', x)
      if ptr[0] == 0 then
         return false
      else
         return true
      end
   end
)

DiskFile.isBigEndianCPU =
   argcheck(
   {},
   function()
      return not DiskFile.isLittleEndianCPU()
   end
)

DiskFile.nativeEndianEncoding =
   argcheck(
   {{name="self", type="torch.DiskFile"}},
   function(self)
      assert(self.__handle, 'attempt to write in a closed file')
      self.__isNativeEncoding = true
      return self
   end
)

DiskFile.littleEndianEncoding =
   argcheck(
   {{name="self", type="torch.DiskFile"}},
   function(self)
      assert(self.__handle, 'attempt to write in a closed file')
      self.__isNativeEncoding = DiskFile.isLittleEndianCPU()
      return self
   end
)

DiskFile.bigEndianEncoding =
   argcheck(
   {{name="self", type="torch.DiskFile"}},
   function(self)
      assert(self.__handle, 'attempt to write in a closed file')
      self.__isNativeEncoding = DiskFile.isBigEndianCPU()
      return self
   end
)

DiskFile.__read =
   argcheck(
   {{name="self", type="torch.DiskFile"},
    {name="data", type="cdata"},
    {name="elemsize", type="number"},
    {name="size", type="number"}},
   function(self, data, elemsize, size)
      assert(self.__handle, 'attempt to write in a closed file')
      assert(self.__isReadable, 'write-only file')
      local n = tonumber(ffi.C.fread(data, elemsize, size, self.__handle))
      if not self.__isNativeEncoding and elemsize > 1 then
         reversememory(data, data, elemsize, n)
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

DiskFile.__printf =
   argcheck(
   {{name="self", type="torch.DiskFile"},
    {name="format", type="string"},
    {name="data", type="cdata"},
    {name="size", type="number"}},
   function(self, format, data, size)
      assert(self.__handle, 'attempt to write in a closed file')
      assert(self.__isWritable, 'read-only file')
      local n = 0
      local cast = format2cast[format]
      for i=0,size-1 do
         local ret = ffi.C.fprintf(self.__handle, format, cast(data[i]))
         if ret <= 0 then
            break
         else
            n = n + 1
         end
         if i < size-1 then
            ffi.C.fprintf(self.__handle, ' ')
         end
      end
      return n
   end
)

DiskFile.__scanf =
   argcheck(
   {{name="self", type="torch.DiskFile"},
    {name="format", type="string"},
    {name="data", type="cdata"},
    {name="size", type="number"}},
   function(self, format, data, size)
      assert(self.__handle, 'attempt to write in a closed file')
      assert(self.__isReadable, 'write-only file')
      local n = 0
      for i=0,size-1 do
         local ret = ffi.C.fscanf(self.__handle, format, data+i)
         if ret <= 0 then
            break
         else
            n = n + 1
         end
      end
      if self.__isAutoSpacing and size > 0 then
         local c = ffi.C.fgetc(self.__handle)
         if string.char(c) ~= '\n' and c ~= -1 then
            ffi.C.ungetc(c, self.__handle)
         end
      end
      return n
   end
)

DiskFile.__gets =
   argcheck(
   {{name="self", type="torch.DiskFile"}},
   function(self)
      assert(self.__handle, 'attempt to write in a closed file')
      assert(self.__isReadable, 'write-only file')
      local size = 0
      local buffsize = 1024
      local buffer = ffi.cast('char*', ffi.C.malloc(buffsize))
      local eof
      while true do
         assert(buffer ~= nil, 'out of memory')
         if ffi.C.fgets(buffer+size, buffsize, self.__handle) == nil then
            eof = true
            break
         end
         local l = tonumber(ffi.C.strlen(buffer+size))
         if l == 0 or string.char(buffer[size+l-1]) ~= '\n' then
            size = size + l
            buffer = ffi.cast('char*', ffi.C.realloc(buffer, size+buffsize))
         else
            size = size + l - 1 -- do not add eol
            break
         end
      end
      if not eof then
         local str = ffi.string(buffer, size)
         ffi.C.free(buffer)
         return str
      else
         ffi.C.free(buffer)
      end
   end
)

DiskFile.__init =
   argcheck(
   {
    {name="self", type="torch.DiskFile"},
    {name="name", type="string"},
    {name="mode", type="string", default='r'},
    {name="quiet", type="boolean", default=false}},
   function(self, name, mode, quiet)
      assert(mode == 'r' or mode == 'w' or mode == 'rw', 'invalid mode (r, w or rw expected)')

      local handle
      if mode == 'rw' then
         handle = ffi.C.fopen(name, 'r+b')
         if handle == nil then
            handle = ffi.C.fopen(name, 'wb')
            if handle ~= nil then
               ffi.C.fclose(handle)
               handle = ffi.C.fopen(name, 'r+b')
            end
         end
      else
         handle = ffi.C.fopen(name, mode == 'r' and 'rb' or 'wb')
      end

      if handle == nil then
         if quiet then
            return
         else
            error(string.format('cannot open file <%s> in mode <%s>)', name, mode))
         end
      end

      self.__handle = handle
      self.__name = name
      self.__isQuiet = quiet
      self.__isReadable = (mode == 'r') or (mode == 'rw')
      self.__isWritable = (mode == 'w') or (mode == 'rw')
      self.__isBinary = false
      self.__isNativeEncoding = true
      self.__isAutoSpacing = true
      self.__hasError = false

      ffi.gc(self.__handle, ffi.C.fclose)
   end
)

torch.DiskFile = torch.constructor(DiskFile)
