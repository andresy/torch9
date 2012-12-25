local Storage = {__typename="torch.Storage"}
local mt

local realsz = ffi.sizeof('real')
local realptrct = ffi.typeof('real*')

local function rawInitWithSize(size)
   local self = {}
   setmetatable(self, mt)
   if size > 0 then
      self.__data = ffi.cast(realptrct, ffi.C.malloc(realsz*size))
      ffi.gc(self.__data, ffi.C.free)
      self.__size = size
   else
      self.__size = 0
   end
   self.__flag = 0
   return self
end

function Storage.new(...)
   local narg = select('#', ...)
   local self
   if narg == 0 then
      return rawInitWithSize(0)
   elseif narg == 1 and type(select(1, ...)) == 'number' then
      return rawInitWithSize(select(1, ...))
   elseif narg == 1 and type(select(1, ...)) == 'table' then
      local tbl = select(1, ...)
      local size = #tbl
      self = rawInitWithSize(size)
      for i=1,size do
         self.__data[i-1] = tbl[i]
      end
   elseif narg == 1 and type(select(1, ...)) == 'string' then
--      self = TH.THStorage_newWithMapping(select(1, ...), 0)[0]
   elseif narg == 2 and type(select(1, ...)) == 'string' and type(select(2, ...)) == 'boolean' then
--      self = TH.THStorage_newWithMapping(select(1, ...), select(2, ...))[0]
   else
      error('invalid arguments')
   end
   return self
end

function Storage:fill(value)
   for i=0,self.__size-1 do
      self.__data[i] = value
   end
   return self
end

function Storage:size()
   return self.__size
end

function Storage:resize(size)
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

function Storage:rawCopy(data)
   ffi.copy(self.__data, data, realsz*self.__size)
   return self
end

mt = {
   __index=function(self, k)
              if type(k) == 'number' then
                 if k > 0 and k <= self.__size then
                    return tonumber(self.__data[k-1])
                 else
                    error('index out of bounds')
                 end
              else
                 return Storage[k]
              end
           end,

   __newindex=function(self, k, v)
                 if type(k) == 'number' then
                    if k > 0 and k <= self.__size then
                       self.__data[k-1] = v
                    else
                       error('index out of bounds')
                    end
                 else
                    rawset(self, k, v)
                 end
              end,

   __metatable = Storage
}

torch.Storage = {}
setmetatable(torch.Storage, {__index=Storage,
                             __metatable=Storage,
                             __newindex=Storage,
                             __call=function(self, ...)
                                       return Storage.new(...)
                                    end})

