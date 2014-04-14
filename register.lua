local argcheck = require 'argcheck'

local function tablecopyarg(tbl, method)
   local newtbl = {}
   for k,v in pairs(tbl) do
      if k ~= 'method' then
         newtbl[k] = v
      end
   end
   if method and tbl.method then
      for k,v in pairs(tbl.method) do
         newtbl[k] = v
      end
   end
   return newtbl
end

local function tablecopy(tbl, method)
   local newtbl = {}
   for k,v in pairs(tbl) do
      if k ~= 'name' then
         if type(k) == 'number' and type(v) == 'table' then
            newtbl[k] = tablecopyarg(v, method)
         else
            newtbl[k] = v
         end
      end
   end
   return newtbl
end

local function register(args, namespace, metatable)
   local name = args.name

   assert(name, 'missing function name')

   if namespace then
      local args_f = tablecopy(args)
      if namespace[name] then
         args_f.chain = namespace[name]
      end
      namespace[name] = argcheck(args_f)
   end

   if metatable then
      local args_m = tablecopy(args, true)
      if args_m[1] then
         args_m[1].name = "self"
      end
      if metatable[name] then
         args_m.chain = metatable[name]
      end
      metatable[name] = argcheck(args_m)
   end

end

return register
