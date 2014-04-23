local torch = require 'torch.env'

-- NOTE:
-- the c1, c2... c5 trick is due to VARG not being compiled in luaJIT

local function generate_apply(dim)
   local func = {}
   local funcarg = {}
   for n=1,#dim do
      table.insert(funcarg, string.format('t%d', n))
   end
   table.insert(func, string.format('return function(%s, func, c1, c2, c3, c4, c5)', table.concat(funcarg, ', ')))
   for n=1,#dim do
      for i=0,dim[n]-1 do
         table.insert(func, string.format('local t%dsz%d, t%dst%d = tonumber(t%d.__size[%d]), tonumber(t%d.__stride[%d])', n, i, n, i, n, i, n, i))
      end
      table.insert(func, string.format('local t%ddata = t%d.__storage.__data + t%d.__storageOffset', n, n, n))
   end
   for n=1,#dim do
      for i=0,dim[n]-1 do
         table.insert(func, string.format('local t%di%d = 0', n, i))
      end
   end
   local cond = {}
   for n=1,#dim do
      if dim[n] > 1 then
         table.insert(cond, string.format('t%di0 < t%dsz0', n, n))
      end
   end

   if #cond > 0 then
      table.insert(func, string.format('while %s do', table.concat(cond, ' and ')))
   end

   local maxarg = {}
   for n=1,#dim do
      if dim[n] > 0 then
         table.insert(maxarg, string.format('t%dsz%d-t%di%d', n, dim[n]-1, n, dim[n]-1))
      else
         table.insert(maxarg, '0')
      end
   end
   table.insert(func, string.format('local r = math.min(%s)', table.concat(maxarg, ', ')))

   -- do stuff
   local funcarg = {}
   for n=1,#dim do
      local data = {string.format('t%ddata', n)}
      for i=0,dim[n]-1 do
         table.insert(data, string.format(' + t%di%d*t%dst%d', n, i, n, i))
      end
      table.insert(funcarg, table.concat(data, ''))
      if dim[n] > 0 then
         table.insert(funcarg, string.format('t%dst%d', n, dim[n]-1))
      else
         table.insert(funcarg, '0')
      end
   end
   table.insert(func, string.format('func(r, %s, c1, c2, c3, c4, c5)', table.concat(funcarg, ', ')))


   for n=1,#dim do
      if dim[n] > 0 then
         table.insert(func, string.format('t%di%d = t%di%d + r', n, dim[n]-1, n, dim[n]-1))
      end
      if dim[n] > 1 then
         table.insert(func, string.format('if t%di%d == t%dsz%d then', n, dim[n]-1, n, dim[n]-1))
         table.insert(func, string.format('t%di%d = 0', n, dim[n]-1))
         for i=dim[n]-2,0,-1 do
            table.insert(func, string.format('t%di%d = t%di%d + 1', n, i, n, i))
            if i > 0 then
               table.insert(func, string.format('if t%di%d == t%dsz%d then', n, i, n, i))
               table.insert(func, string.format('t%di%d = 0', n, i))
            end
         end
         for i=dim[n]-2,1,-1 do
            table.insert(func, 'end')
         end
         table.insert(func, 'end')
      end
   end

   if #cond > 0 then
      table.insert(func, 'end')
   end
   table.insert(func, 'end')
   return table.concat(func, '\n')
end

local applyfuncs = {}
function torch.rawapply(t1, func, c1, c2, c3, c4, c5)
   local dim = tonumber(t1.__nDimension)
   local applyfunc = applyfuncs[dim]
   if not applyfunc then
      applyfunc = loadstring(generate_apply({dim}))()
      applyfuncs[dim] = applyfunc
   end
   applyfunc(t1, func, c1, c2, c3, c4, c5)
end

local apply2funcs = {}
function torch.rawapply2(t1, t2, func, c1, c2, c3, c4, c5)
   local dim1 = tonumber(t1.__nDimension)
   local dim2 = tonumber(t2.__nDimension)
   apply2funcs[dim1] = apply2funcs[dim1] or {}
   local applyfunc = apply2funcs[dim1][dim2]
   if not applyfunc then
      applyfunc = loadstring(generate_apply({dim1,dim2}))()
      apply2funcs[dim1][dim2] = applyfunc
   end
   applyfunc(t1, t2, func, c1, c2, c3, c4, c5)
end

local apply3funcs = {}
function torch.rawapply3(t1, t2, t3, func, c1, c2, c3, c4, c5)
   local dim1 = tonumber(t1.__nDimension)
   local dim2 = tonumber(t2.__nDimension)
   local dim3 = tonumber(t3.__nDimension)
   apply3funcs[dim1] = apply3funcs[dim1] or {}
   apply3funcs[dim1][dim2] = apply3funcs[dim1][dim2] or {}
   local applyfunc = apply3funcs[dim1][dim2][dim3]
   if not applyfunc then
      applyfunc = loadstring(generate_apply({dim1,dim2,dim3}))()
      apply3funcs[dim1][dim2][dim3] = applyfunc
   end
   applyfunc(t1, t2, t3, func, c1, c2, c3, c4, c5)
end
