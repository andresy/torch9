local torch = require 'torch'
local argcheck = require 'torch.argcheck'

local classes = {}

torch.class =
   argcheck(
   {{name="name", type="string"},
    {name="parentname", type="string", opt=true}},
   function(name, parentname)   
      assert(not classes[name], 'class <%s> already exists', name)

      local class = {__typename = name, __version=1}
      class.__index = class

      class.__init =
         function()
         end

      class.new =
         function(...)
            local self = {}
            setmetatable(self, class)
            self:__init(...)
            return self
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

torch.factory =
   argcheck(
   {{name="name", type="string"}},
   function(name)
      assert(classes[name], string.format('unknown class <%s>', name))
      local t = {}
      setmetatable(t, classes[name])
      return t
   end
)

torch.metatable =
   argcheck(
   {{name="name", type="string"}},
   function(name)
      return classes[name]
   end
)

local function constructor(metatable, constructor)
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

torch.constructor =
   argcheck(
   {{name="metatable", type="table"},
    {name="ctname", type="string", default="new"}},
    function(metatable, ctname)
       assert(metatable[ctname], string.format('constructor <%s> does not exist in metatable', ctname))
       return constructor(metatable, metatable[ctname])
    end,

   {{name="metatable", type="table"},
    {name="constructor", type="function"}},
   constructor
)
