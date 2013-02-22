local src = arg[1]
local dst = arg[2]

local types = {'byte', 'char', 'short', 'int', 'long', 'float', 'double'}
local Types = {'Byte', 'Char', 'Short', 'Int', 'Long', 'Float', 'Double'}
local taccs = {'long', 'long', 'long', 'long', 'long', 'double', 'double'}

local f = io.open(src)
local txt = f:read('*all')
f:close()

for i=1,#types do
   local real, Real, accreal = types[i], Types[i], taccs[i]
   local txt = txt

   while txt:match('([%p%s])real([%p%s])') do
      txt = txt:gsub('([%p%s])real([%p%s])', '%1' .. real .. '%2')
   end

   while txt:match('([%p%s])accreal([%p%s])') do
      txt = txt:gsub('([%p%s])accreal([%p%s])', '%1' .. accreal .. '%2')
   end

   while txt:match('([%p%s])Storage([%p%s])') do
      txt = txt:gsub('([%p%s])Storage([%p%s])', '%1' .. Real .. 'Storage' .. '%2')
   end

   while txt:match('([%p%s])Tensor([%p%s])') do
      txt = txt:gsub('([%p%s])Tensor([%p%s])', '%1' .. Real .. 'Tensor' .. '%2')
   end

   local dst = dst:gsub('(%.[^%.]+)$', '_' .. real .. '%1')
   assert(dst ~= src, 'source and destination are same')
   local f = io.open(dst, 'w')
   f:write(txt)
   f:close()
end   

local basename, ext = dst:match('([^/\\]+)(%.[^%.]+)$')
if not basename or not ext then
   error('could not determine destination file basename/extension')
end

local txt = {}
if ext == '.lua' then
   local module = arg[3]
   assert(module, 'module name should be provided')
   for i=1,#types do
      table.insert(txt, string.format("require '%s.%s_%s'", module, basename, types[i]))
   end
else
   for i=1,#types do
      table.insert(txt, string.format('#include "%s_%s%s"', basename, types[i], ext))
   end
end
assert(dst ~= src, 'source and destination are same')
local f = io.open(dst, 'w')
f:write(table.concat(txt, '\n'))
f:close()
