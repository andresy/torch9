local torch = require 'torch'

function torch.type(obj)
   local tname = type(obj)
   if tname == 'table' then
      return obj.__typename or tname
   end
   return tname
end
torch.typename = torch.type -- backward compatibility... keep it or not?

function torch.istype(obj, typename)
   local tname = type(obj)
   if tname == 'table' then
      if obj.__typename then
         obj = getmetatable(obj)
         while type(obj) == 'table' do
            if obj.__typename == typename then
               return true
            else
               obj = getmetatable(obj)
            end
         end
         return false
      else
         return tname == typename
      end
   else
      return typename == tname
   end
end
