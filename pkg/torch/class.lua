local torch = require 'torch'
local argcheck = require 'torch.argcheck'

local classes = {}

torch.class =
   argcheck(
   {{name="name", type="string"},
    {name="parentname", type="string", opt=true}},
   function(name, parentname)   
      assert(not classes[name], 'class <%s> already exists', name)
      local class = {__typename = name}
      class.__index = class
      class.__init = function()
                        local t = {}
                        setmetatable(t, class)
                        return t
                     end
      
      classes[name] = class
      
      if parentname then
         assert(classes[parentname], 'parent class <%s> does not exist', parentname)
         setmetatable(class, classes[parentname])
         return class, classes[parentname]
      else
         return class
      end
   end
)

torch.metatable =
   argcheck(
   {{name="name", type="string"}},
   function(name)
      return classes[name]
   end
)

torch.constructor =
   argcheck(
   {{name="metatable", type="table"},
    {name="constructor", type="string", default="new"}},
   function(metatable, constructor)
      assert(metatable[constructor], string.format('field constructor <%s> is nil', constructor))
      constructor = metatable[constructor]
      local ct = {}
      setmetatable(ct, {
                      __index=metatable,
                      __newindex=metatable,
                      __metatable=metatable,
                      __call=function(self, ...)
                                return constructor(...)
                             end
                   })
      return ct
   end
)
