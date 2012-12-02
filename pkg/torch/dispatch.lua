
dispatch = 
   argcheck{
   {{name="idx", type="number", default=1}},
   function(idx)
      local env = {idx=idx, funcs={}, type=type, select=select, error=error}
      local func
      if idx > 0 then
         func =
            function(...)
               local typename = type(select(idx, ...))
               local func = funcs[typename]
               if not func then
                  error('function not implemented for type ' .. typename)
               else
                  return func(...)
               end
            end
      else
         func = 
            function(...)
            end
      end
      setfenv(func, env)
      return func
   end,
   
   {{name="func", type="function"},
    {name="typename", type="string"},
    {name="functypename", type="function"}},
   function(func, typename, functypename)
      getfenv(func).funcs[typename] = functypename
   end
}

return dispatch
