local register_ = require 'torch.register'
local argcheck = require 'argcheck'
local torch = require 'torch.env'
local class = require 'class'
local ffi = require 'ffi'
local C = require 'torch.TH'

-- DEBUG: should register() be in argcheck?
local function register(args)
   return register_(args, torch, class.metatable('torch.Generator'))
end

local Generator = class('torch.Generator', nil, ffi.typeof('THGenerator&'))
torch.Generator = Generator

Generator.new = argcheck{
   call =
      function()
         local self = C.THGenerator_new()[0]
         ffi.gc(self, C.THGenerator_free)
         return self
      end
}

Generator.__factory = Generator.new

torch.__generator = torch.__generator or torch.Generator()

ffi.metatype('THGenerator', class.metatable('torch.Generator'))

register{
   name = "random",
   {name="generator", type="torch.Generator", opt=true, method={opt=false}},
   {name="b", type="number", opt=true},
   call =
      function(generator, b)
         generator = generator or torch.__generator
         if b then
            return tonumber(C.THRandom_random(generator)) % b
         else
            return tonumber(C.THRandom_random(generator))
         end
      end
}

register{
   name = "manualSeed",
   {name="generator", type="torch.Generator", opt=true, method={opt=false}},
   {name="seed", type="number"},
   call =
      function(generator, seed)
         generator = generator or torch.__generator
         C.THRandom_manualSeed(generator, seed)
         return generator
      end
}

register{
   name = "uniform",
   {name="generator", type="torch.Generator", opt=true, method={opt=false}},
   {name="a", type="number", default=0},
   {name="b", type="number", default=1},
   call =
      function(generator, a, b)
         generator = generator or torch.__generator
         return tonumber(C.THRandom_uniform(generator, a, b))
      end
}

register{
   name = "normal",
   {name="generator", type="torch.Generator", opt=true, method={opt=false}},
   {name="a", type="number", default=0},
   {name="b", type="number", default=1},
   call =
      function(generator, a, b)
         generator = generator or torch.__generator
         return tonumber(C.THRandom_normal(generator, a, b))
      end
}

register{
   name = "cauchy",
   {name="generator", type="torch.Generator", opt=true, method={opt=false}},
   {name="a", type="number", default=0},
   {name="b", type="number", default=1},
   call =
      function(generator, a, b)
         generator = generator or torch.__generator
         return tonumber(C.THRandom_cauchy(generator, a, b))
      end
}

register{
   name = "logNormal",
   {name="generator", type="torch.Generator", opt=true, method={opt=false}},
   {name="a", type="number", default=1},
   {name="b", type="number", default=2},
   call =
      function(generator, a, b)
         generator = generator or torch.__generator
         return tonumber(C.THRandom_logNormal(generator, a, b))
      end
}

register{
   name = "exponential",
   {name="generator", type="torch.Generator", opt=true, method={opt=false}},
   {name="a", type="number", default=1},
   call =
      function(generator, a)
         generator = generator or torch.__generator
         return tonumber(C.THRandom_exponential(generator, a))
      end
}

register{
   name = "geometric",
   {name="generator", type="torch.Generator", opt=true, method={opt=false}},
   {name="a", type="number"},
   call =
      function(generator, a)
         generator = generator or torch.__generator
         return tonumber(C.THRandom_geometric(generator, a))
      end
}

register{
   name = "bernoulli",
   {name="generator", type="torch.Generator", opt=true, method={opt=false}},
   {name="a", type="number", default=0.5},
   call =
      function(generator, a)
         generator = generator or torch.__generator
         return tonumber(C.THRandom_bernoulli(generator, a))
      end
}
