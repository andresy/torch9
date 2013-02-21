local torch = package.loaded.torch

local function namedispatch(...)
   local func = funcs[select(-1, ...)]
   if func then
      return func(unpack({...}, 1, select('#', ...)-1))
   else
      assert(torch and torch.Tensor, 'default torch.Tensor type does not exist')
      func = funcs[torch.Tensor.__typename]
      if not func then
         error('function not implemented for default type <%s>', torch.Tensor.__typename)
      end
      return func(...)
   end
end

return namedispatch
