local function generate_apply(dim)
   local func = {}
   local funcarg = {}
   for n=1,#dim do
      table.insert(funcarg, string.format('t%d', n))
   end
   table.insert(func, string.format('return function(%s, func)', table.concat(funcarg, ', ')))
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
   table.insert(func, string.format('func(%s, r)', table.concat(funcarg, ', ')))


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

local applyfunc = {}
for i=0,10 do
   applyfunc[i] = loadstring(generate_apply({i}))()
end
function torch.apply(t1, func)
   local applyfunc = applyfunc[tonumber(t1.__nDimension)]
   applyfunc(t1, func)
end

local apply2func = {}
for i=0,10 do
   apply2func[i] = {}
   for j=1,10 do
      apply2func[i][j] = loadstring(generate_apply({i,j}))()
   end
end

function torch.apply2(t1, t2, func)
   local applyfunc = apply2func[tonumber(t1.__nDimension)][tonumber(t2.__nDimension)]
   applyfunc(t1, t2, func)
end

local apply3func = {}
for i=0,10 do
   apply3func[i] = {}
   for j=1,10 do
      apply3func[i][j] = {}
      for k=1,10 do
         apply3func[i][j][k] = loadstring(generate_apply({i,j,k}))()
      end
   end
end

function torch.apply3(t1, t2, t3, func)
   local applyfunc = apply3func[tonumber(t1.__nDimension)][tonumber(t2.__nDimension)][tonumber(t3.__nDimension)]
   applyfunc(t1, t2, t3, func)
end
