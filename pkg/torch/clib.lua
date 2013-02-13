local function findclib()
   for path in string.gmatch(package.cpath, '[^%;]+') do
      path = path:gsub('%?', 'libtorch')
      local f = io.open(path)
      if f then
         f:close()
         return path
      end
   end
end

local ffi = require 'ffi'
local clibpath = findclib()

assert(clibpath, 'torch C library not found')

return ffi.load(clibpath)
