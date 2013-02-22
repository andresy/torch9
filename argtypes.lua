local argcheck = require 'argcheck'
local argtypes = getfenv(argcheck).argtypes

for _,Tensor in ipairs{'torch.ByteTensor',
                       'torch.CharTensor',
                       'torch.ShortTensor',
                       'torch.IntTensor',
                       'torch.LongTensor',
                       'torch.FloatTensor',
                       'torch.DoubleTensor'} do

   argtypes[Tensor] = {
      check = function(self)
                 if self.dim then
                    return string.format("type(%s) == '" .. Tensor .. "' and (%s).__nDimension == %d", self.luaname, self.luaname, self.dim)
                 else
                 return string.format("type(%s) == '" .. Tensor .. "'", self.luaname)
              end
           end,

      initdefault = function(self)
                       return string.format('%s = %s()', self.name, Tensor)
                    end
   }
end

for _,Storage in ipairs{'torch.ByteStorage',
                        'torch.CharStorage',
                        'torch.ShortStorage',
                        'torch.IntStorage',
                        'torch.LongStorage',
                        'torch.FloatStorage',
                        'torch.DoubleStorage'} do

   argtypes[Storage] = {
      check = function(self)
                 if self.size then
                    return string.format("type(%s) == '" .. Storage .. "' and (%s).__size == %d", self.luaname, self.luaname, self.size)
                 else
                 return string.format("type(%s) == '" .. Storage .. "'", self.luaname)
              end
           end,

      initdefault = function(self)
                       return string.format('%s = %s()', self.name, Storage)
                    end
   }
end
