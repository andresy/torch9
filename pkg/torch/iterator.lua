local function generateiterator()
   local func = {}
   table.insert(func, [[
function torch.iterator(t)
   if t:isContiguous() then
      local done = false
      return function()
                if done then
                   return nil
                else
                   done = true
                   return t.__storage.__data + t.__storageOffset, t:nElement(), 1
                end
             end
]])
   for dim=1,10 do
      table.insert(func, string.format('elseif t.__nDimension == %d then', dim))
      for i=0,dim-1 do
         table.insert(func, string.format('local sz%d, st%d = tonumber(t.__size[%d]), tonumber(t.__stride[%d])', i, i, i, i))
      end
      table.insert(func, 'local data = t.__storage.__data + t.__storageOffset')
      table.insert(func, 'local k = -1')
      table.insert(func, 'return function()')
      table.insert(func, 'k = k + 1')
      local tst = {}
      for i=0,dim-2 do
         table.insert(tst, string.format('sz%d', i))
      end
      if #tst == 0 then
         table.insert(func, 'if k >= 1 then')
      else
         table.insert(func, string.format('if k >= %s then', table.concat(tst, '*')))
      end
      table.insert(func, 'return nil')
      table.insert(func, 'else')
      for i=0,dim-2 do
         local div = {}
         for j=i+1,dim-2 do
            table.insert(div, string.format('sz%d', j))
         end
         if #div > 0 then
            div = table.concat(div, '*')
         else
            div = nil
         end
         if i == 0 then
            if div then
               table.insert(func, string.format('local i%d = math.floor(k/(%s))', i, div))
               table.insert(func, string.format('local r = k %% (%s)', div))
            else
               table.insert(func, string.format('local i%d = k', i))
            end
         else
            if div then
               table.insert(func, string.format('local i%d = math.floor(r/(%s))', i, div))
               table.insert(func, string.format('r = k %% (%s)', div))
            else
               table.insert(func, string.format('local i%d = r', i))
            end
         end
      end
      local ret = {}
      for i=0,dim-2 do
         table.insert(ret, string.format(' + i%d*st%d', i, i))
      end
      table.insert(func, string.format('return data%s, sz%d, st%d', table.concat(ret, ''), dim-1, dim-1))
      table.insert(func, 'end') -- if k...
      table.insert(func, 'end') -- function()
   end
   
   table.insert(func, 'else')
   table.insert(func, 'error("the provided tensor has too many dimensions")')
   table.insert(func, 'end')
   table.insert(func, 'end')
   
   return table.concat(func, '\n')
end

--print(generateiterator())

loadstring(generateiterator())()

function torch.iterator2(t1, t2)
   local iter1 = torch.iterator(t1)
   local iter2 = torch.iterator(t2)

   local ptr1, sz1, st1 = iter1()
   local ptr2, sz2, st2 = iter2()
   return function()
             if ptr1 and ptr2 then
                if sz1 > sz2 then
                   local ptr2s, sz2s, st2s = ptr2, sz2, st2
                   local ptr1s = ptr1
                   sz1 = sz1 - sz2
                   ptr1 = ptr1 + sz2*st1
                   ptr2, sz2, st2 = iter2()
                   return ptr1s, st1, ptr2s, st2s, sz2s
                elseif sz1 < sz2 then
                   local ptr1s, sz1s, st1s = ptr1, sz1, st1
                   local ptr2s = ptr2
                   sz2 = sz2 - sz1
                   ptr2 = ptr2 + sz1*st2
                   ptr1, sz1, st1 = iter1()
                   return ptr1s, st1s, ptr2s, st2, sz1s
                else
                   local ptr1s, sz1s, st1s = ptr1, sz1, st1
                   local ptr2s, sz2s, st2s = ptr2, sz2, st2
                   ptr1, sz1, st1 = iter1()
                   ptr2, sz2, st2 = iter2()
                   return ptr1s, st1s, ptr2s, st2s, sz1s
                end
             else
                return
             end
          end
end
