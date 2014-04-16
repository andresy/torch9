local register_ = require 'torch.register'
local torch = require 'torch.env'

local function copy_args(args)
   local tbl = {}
   for k,v in pairs(args) do
      tbl[k] = v
   end
   return tbl
end

-- handle numbers type
local function register(args, namespace, metatable)
   local nidx
   for idx,arg in ipairs(args) do
      if arg.type == 'numbers' then
         if nidx then
            error('only one argument can be of <numbers> type')
         end
         nidx = idx
      end
   end
   if nidx then
      assert(args.call, '<numbers> is supposed to be used together with <call>')

      -- with table
      local new_args = copy_args(args)
      new_args[nidx] = copy_args(new_args[nidx]) -- avoid modification with no warning
      new_args[nidx].type = 'table'
      local funcargs = {}
      local callargs = {}
      for i=1,#new_args do
         table.insert(funcargs, string.format('arg%d', i))
         table.insert(callargs, string.format('arg%d', i))
      end
      callargs[nidx] = 'numbers'
      funcargs = table.concat(funcargs, ', ')
      callargs = table.concat(callargs, ', ')

      local numbers = torch.LongStorage()
      local code = [[
local call
local numbers
return function(%s)
  local sz = #arg%d
  numbers:resize(sz)
  for i=1,sz do
    numbers.__data[i-1] = arg%d[i]
  end
  return call(%s)
end
]]
      code = string.format(code, funcargs, nidx, nidx, callargs)
      code = loadstring(code)()
      debug.setupvalue(code, 1, numbers)
      debug.setupvalue(code, 2, args.call)
      new_args.call = code
      register_(new_args, namespace, metatable)

      -- with numbers (up to N)
      local N = 5
      local new_args = copy_args(args)
      table.remove(new_args, nidx)
      for i=1,N do
         local arg = copy_args(args[nidx])
         arg.name = arg.name .. i
         arg.type = "number"
         if i > 1 then
            arg.default = 0
         end
         table.insert(new_args, nidx+i-1, arg)
      end
      local funcargs = {}
      local callargs = {}
      for i=1,#new_args do
         table.insert(funcargs, string.format('arg%d', i))
         table.insert(callargs, string.format('arg%d', i))
      end
      callargs[nidx] = 'numbers'
      for i=2,N do
         table.remove(callargs, nidx+1)
      end
      funcargs = table.concat(funcargs, ', ')
      callargs = table.concat(callargs, ', ')

      local numbers = torch.LongStorage(5)
      local code = [[
local call
local numbers
return function(%s)
  numbers.__data[0] = arg%d
  numbers.__data[1] = arg%d
  numbers.__data[2] = arg%d
  numbers.__data[3] = arg%d
  numbers.__data[4] = arg%d
  return call(%s)
end
]]
      code = string.format(code, funcargs, nidx, nidx+1, nidx+2, nidx+3, nidx+4, callargs)
      code = loadstring(code)()
      debug.setupvalue(code, 1, numbers)
      debug.setupvalue(code, 2, args.call)
      new_args.call = code
      register_(new_args, namespace, metatable)

      -- with LongStorage
      local new_args = copy_args(args)
      args[nidx].type = 'torch.LongStorage'
      register_(new_args, namespace, metatable)
   else
      register_(args, namespace, metatable)
   end
end

return register
