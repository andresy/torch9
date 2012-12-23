local torch = require 'torch'
local ffi = require 'ffi'

ffi.cdef[[
      unsigned long seed();
      void manualSeed(unsigned long the_seed_);
      unsigned long initialSeed();
      void nextState();
      unsigned long random();
]]

local th = ffi.load(paths.concat(paths.install_lua_path,
                                 'torch',
                                 ((jit.os == 'Windows') and '' or 'lib') .. 'maths' .. 
                                 ((jit.os == 'Windows') and '.dll' or ((jit.os == 'OSX') and '.dylib' or '.so'))))

torch.random = argcheck(
   {},
   function()
      return tonumber(th.random())
   end
)

torch.manualSeed = argcheck(
   {{name="seed", type="number"}},
   function(seed)
      return th.manualSeed(seed)
   end
)

torch.seed = argcheck(
   {{name="seed", type="number", opt=true}},
   function(seed)
      if seed then
         th.manualSeed(seed)
      else
         return tonumber(th.seed())
      end
   end
)

torch.uniform = argcheck(
   {{name="a", type="number", default=-1},
    {name="b", type="number", default=1}},
   function(a, b)
      return tonumber(th.random())/2^32 * (b-a) + a
   end
)

torch.normal = argcheck(
   {{name="mean", type="number", default=0},
    {name="std", type="number", default=1}},
   function(mean, std)
      local u, v
      repeat
         u = tonumber(th.random())/2^32
         v = 1.7156*(tonumber(th.random())/2^32-0.5)
         local x = u - 0.449871
         local y = math.abs(v) + 0.386595
         local q = x*x + y*(0.196*y-0.25472*x)
      until q <= 0.27597 or (q <= 0.27846 and v*v <= -4*math.log(u)*u*u)
      return mean + std*v/u
   end
)

torch.exponential = argcheck(
   {{name="lambda", type="number"}},
   function(lambda)
      return -1. / lambda * math.log(1-torch.uniform())
   end
)

torch.cauchy = argcheck(
   {{name="median", type="number"},
    {name="sigma", type="number"}},
   function(median, sigma)
      return median + sigma * math.tan(math.pi*(torch.uniform()-0.5))
   end
)

torch.cauchy = argcheck(
   {{name="median", type="number"},
    {name="sigma", type="number"}},
   function(median, sigma)
      return median + sigma * math.tan(math.pi*(torch.uniform()-0.5))
   end
)

torch.lognormal = argcheck(
   {{name="mean", type="number"},
    {name="std", type="number"}},
   function(mean, std)
      local zm = mean*mean;
      local zs = std*std;
      assert(std > 0, "standard deviation must be strictly positive")
      return math.exp(torch.normal(math.log(zm/math.sqrt(zs + zm)), math.sqrt(math.log(zs/zm+1)) ))
   end
)

torch.geometric = argcheck(
   {{name="p", type="number"}},
   function(p)
      assert(p > 0 and p < 1, "p must be > 0 and < 1")
      return math.floor(math.log(1-torch.uniform()) / math.log(p)) + 1
   end
)

torch.bernoulli = argcheck(
   {{name="p", type="number"}},
   function(p)
      assert(p >= 0 and p <= 1, "p must be >= 0 and <= 1")
      return torch.uniform() <= p
   end
)
