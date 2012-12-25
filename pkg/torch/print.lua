local function storageformat(self)
   local intMode = true
   local expMin =  math.huge
   local expMax = -math.huge
   local type = torch.typename(self)
   for i=1,self:size() do
      local z = tonumber(self[i])
      if z ~= math.ceil(z) then
         intMode = false
      end
      expMin = math.min(expMin, math.abs(z))
      expMax = math.max(expMax, math.abs(z))
   end
   if expMin ~= 0 then
      expMin = math.floor(math.log10(expMin)) + 1
   end
   if expMax ~= 0 then
      expMax = math.floor(math.log10(expMax)) + 1
   end

   local format
   local scale
   local sz
   if intMode then
      if expMax > 9 then
         format = "%11.4e"
         sz = 11
      else
         format = "%SZd"
         sz = expMax + 1
      end
   else
      if expMax-expMin > 4 then
         format = "%SZ.4e"
         sz = 11
         if math.abs(expMax) > 99 or math.abs(expMin) > 99 then
            sz = sz + 1
         end
      else
         if expMax > 5 or expMax < 0 then
            format = "%SZ.4f"
            sz = 7
            scale = math.pow(10, expMax-1)
         else
            format = "%SZ.4f"
            if expMax == 0 then
               sz = 7
            else
               sz = expMax+6
            end
         end
      end
   end
   format = string.gsub(format, 'SZ', sz)
   if scale == 1 then
      scale = nil
   end
   return format, scale, sz
end

function torch.Storage:__tostring()
   local strt = {'\n'}
   local format, scale = storageformat(self)
   if format:sub(2,4) == 'nan' then format = '%f' end
   format = format .. '\n'
   if scale then
      table.insert(strt, string.format('%g *\n', scale))
      for i = 1,self:size() do
         table.insert(strt, string.format(format, self[i]/scale))
      end
   else
      for i = 1,self:size() do
         table.insert(strt, string.format(format, self[i]))
      end
   end
   table.insert(strt, string.format('[%s of size %d]\n',  torch.type(self), self:size()))
   local str = table.concat(strt)
   return str
end

local function printmatrix(self, indent)
   local format, scale, sz = storageformat(self:storage())
   if format:sub(2,4) == 'nan' then format = '%f' end
   scale = scale or 1
   indent = indent or ''
   local strt = {indent}
   local nColumnPerLine = math.floor((80-#indent)/(sz+1))
   local firstColumn = 1
   local lastColumn = -1
   while firstColumn <= self:size(2) do
      if firstColumn + nColumnPerLine - 1 <= self:size(2) then
         lastColumn = firstColumn + nColumnPerLine - 1
      else
         lastColumn = self:size(2)
      end
      if nColumnPerLine < self:size(2) then
         if firstColumn ~= 1 then
            table.insert(strt, '\n')
         end
         table.insert(strt, string.format('Columns %d to %d\n%s', firstColumn, lastColumn, indent))
      end
      if scale ~= 1 then
         table.insert(strt, string.format('%g *\n %s', scale, indent))
      end
      for l=1,self:size(1) do
         local row = self:select(1, l)
         for c=firstColumn,lastColumn do
            table.insert(strt, string.format(format, row[c]/scale))
            if c == lastColumn then
               table.insert(strt, '\n')
               if l~=self:size(1) then
                  if scale ~= 1 then
                     table.insert(strt, indent .. ' ')
                  else
                     table.insert(strt, indent)
                  end
               end
            else
               table.insert(strt, ' ')
            end
         end
      end
      firstColumn = lastColumn + 1
   end
   local str = table.concat(strt)
   return str
end

local function printtensor(self)
   local counter = torch.LongStorage(self:nDimension()-2)
   local strt = {''}
   local finished
   counter:fill(1)
   counter[1] = 0
   while true do
      for i=1,self:nDimension()-2 do
         counter[i] = counter[i] + 1
         if counter[i] > self:size(i) then
            if i == self:nDimension()-2 then
               finished = true
               break
            end
            counter[i] = 1
         else
            break
         end
      end
      if finished then
         break
      end
--      print(counter)
      if #strt > 1 then
         table.insert(strt, '\n')
      end
      table.insert(strt, '(')
      local tensor = self
      for i=1,self:nDimension()-2 do
         tensor = tensor:select(1, counter[i])
         table.insert(strt, counter[i] .. ',')
      end
      table.insert(strt, '.,.) = \n')
      table.insert(strt, printmatrix(tensor, ' '))
   end
   local str = table.concat(strt)
   return str
end

function torch.Tensor.__tostring(self)
   local str = '\n'
   local strt = {''}
   if self:nDimension() == 0 then
      table.insert(strt, string.format('[%s with no dimension]\n', torch.typename(self)))
   else
      if self:nDimension() == 1 then
         local format,scale,sz = storageformat(self:storage())
	 if format:sub(2,4) == 'nan' then format = '%f' end
         format = format .. '\n'
         if scale then
            table.insert(strt, string.format('%g *\n', scale))
            for i = 1,self:size(1) do
               table.insert(strt, string.format(format, self[i]/scale))
            end
         else
            for i = 1,self:size(1) do
               table.insert(strt, string.format(format, self[i]))
            end
         end
         table.insert(strt, string.format('[%s of dimension %d]\n', torch.type(self), self:size(1)))
      elseif self:nDimension() == 2 then
         table.insert(strt, printmatrix(self))
         table.insert(strt, string.format('[%s of dimension %dx%d]\n', torch.type(self), self:size(1), self:size(2)))
      else
         table.insert(strt, printtensor(self))
         table.insert(strt, string.format('[%s of dimension ', torch.type(self)))
         for i=1,self:nDimension() do
            table.insert(strt, self:size(i))
            if i ~= self:nDimension() then
               table.insert(strt, 'x')
            end
         end
         table.insert(strt, ']\n')
      end
   end
   local str = table.concat(strt)
   return str
end
