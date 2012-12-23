-- narg: #of input tensors
-- dim: #of dim of the input tensors
-- dima: dimension over which we apply
local function generatedimapply_narg_dim_dima(narg, dim, dima)
   local func = {}
   local funcarg = {}
   for n=1,narg do
      table.insert(funcarg, string.format('t%d', n))
   end
   table.insert(func, string.format('return function(%s, func)', table.concat(funcarg, ', ')))
   for n=1,narg do
      for i=0,dim-1 do
         table.insert(func, string.format('local t%dsz%d, t%dst%d = tonumber(t%d.__size[%d]), tonumber(t%d.__stride[%d])', n, i, n, i, n, i, n, i))
      end
      table.insert(func, string.format('local t%ddata = t%d.__storage.__data+t%d.__storageOffset', n, n, n))
   end
   for i=0,dim-1 do
      if i ~= dima then
         table.insert(func, string.format('for i%d=0,t%dsz%d-1 do', i, 1, i))
      end
   end
   local funcarg = {}
   for n=1,narg do
      local ptr = {string.format('t%ddata', n)}
      for i=0,dim-1 do
         if i ~= dima then
            table.insert(ptr, string.format('+i%d*t%dst%d', i, n, i))
         end
      end
      table.insert(funcarg, table.concat(ptr, ''))
      table.insert(funcarg, string.format('t%dst%d', n, dima))
      table.insert(funcarg, string.format('t%dsz%d', n, dima))
   end
   table.insert(func, string.format('func(%s)', table.concat(funcarg, ', ')))
   for i=0,dim-1 do
      if i ~= dima then
         table.insert(func, 'end')
      end
   end
   table.insert(func, 'end')
   return table.concat(func, '\n')
end

local function generatedimapply_n(n)
   local func = {}
   local decl = {}
   for i=1,n do
      table.insert(decl, string.format('t%d', i))
   end
   table.insert(func, table.concat({string.format('function torch.dimapply%d(', n),
                                    table.concat(decl, ', '),
                                    ', dim, func)'}, ''))
   table.insert(func, 'local ndim = t1.__nDimension')
   table.insert(func, 'dim = dim - 1')
   for ndim=1,2 do
      table.insert(func, string.format('%sif ndim == %d then', ndim == 1 and '' or 'else', ndim))
      table.insert(func, generatedimapply_dim_n(ndim, n))
   end
   table.insert(func, 'else')
   table.insert(func, 'error("the provided tensor has too many dimensions")')
   table.insert(func, 'end') -- if/elseif
   table.insert(func, 'end')
   return table.concat(func, '\n')
end

local dimapply1funcs = {}
function torch.dimapply(t1, dim, func)
   local dim1 = t1.__nDimension
   dimapply1funcs[dim1] = dimapply1funcs[dim1] or {}
   local dimapplyfunc = dimapply1funcs[dim1][dim-1]
   if not dimapplyfunc then
      dimapplyfunc = loadstring(generatedimapply_narg_dim_dima(1, dim1, dim-1))()
      dimapply1funcs[dim1][dim-1] = dimapplyfunc
   end
   dimapplyfunc(t1, func)
end

local dimapply2funcs = {}
function torch.dimapply2(t1, t2, dim, func)
   local dim1 = t1.__nDimension
   dimapply2funcs[dim1] = dimapply2funcs[dim1] or {}
   local dimapplyfunc = dimapply2funcs[dim1][dim-1]
   if not dimapplyfunc then
      dimapplyfunc = loadstring(generatedimapply_narg_dim_dima(2, dim1, dim-1))()
      dimapply2funcs[dim1][dim-1] = dimapplyfunc
   end
   dimapplyfunc(t1, t2, func)
end

local dimapply3funcs = {}
function torch.dimapply3(t1, t2, t3, dim, func)
   local dim1 = t1.__nDimension
   dimapply3funcs[dim1] = dimapply3funcs[dim1] or {}
   local dimapplyfunc = dimapply3funcs[dim1][dim-1]
   if not dimapplyfunc then
      dimapplyfunc = loadstring(generatedimapply_narg_dim_dima(3, dim1, dim-1))()
      dimapply3funcs[dim1][dim-1] = dimapplyfunc
   end
   dimapplyfunc(t1, t2, t3, func)
end
