local ffi = require 'ffi'
local argcheck = require 'argcheck'
local torch = require 'torch.env'
local class = require 'class'
local C = require 'torch.TH'
local register_ = require 'torch.registernumbers'

-- handle method/function
local function register(args)
   if args.nomethod and not args.nofunction then
      return register_(args, torch, nil)
   elseif args.nofunction and not args.nomethod then
      return register_(args, nil, class.metatable('torch.RealTensor'))
   else
      return register_(args, torch, class.metatable('torch.RealTensor'))
   end
end

if 'real' == 'float' or 'real' == 'double' then

   register{
      name = "gesv",
      {name="B", type="torch.RealTensor"},
      {name="A", type="torch.RealTensor"},
      call =
         function(B, A)
            local X = torch.RealTensor()
            local LU = torch.RealTensor()
            C.THRealTensor_gesv(X, LU, B, A)
            return X, LU
         end
   }

   register{
      nomethod = true,
      name = "gesv",
      {name="X", type="torch.RealTensor"},
      {name="LU", type="torch.RealTensor"},
      {name="B", type="torch.RealTensor"},
      {name="A", type="torch.RealTensor"},
      call =
         function(X, LU, B, A)
            C.THRealTensor_gesv(X, LU, B, A)
            return X, LU
         end
   }

   register{
      name = "gels",
      {name="B", type="torch.RealTensor"},
      {name="A", type="torch.RealTensor"},
      call =
         function(B, A)
            local X = torch.RealTensor()
            local LU = torch.RealTensor()
            C.THRealTensor_gels(X, LU, B, A)
            return X, LU
         end
   }

   register{
      nomethod = true,
      name = "gels",
      {name="X", type="torch.RealTensor"},
      {name="LU", type="torch.RealTensor"},
      {name="B", type="torch.RealTensor"},
      {name="A", type="torch.RealTensor"},
      call =
         function(X, LU, B, A)
            C.THRealTensor_gels(X, LU, B, A)
            return X, LU
         end
   }

   register{
      name = "symeig",
      {name="A", type="torch.RealTensor"},
      {name="opteig", type="string", default='N'},
      {name="opttriang", type="string", default='U'},
      call =
         function(A, opteig, opttriang)
            assert(opteig == 'N' or opteig == 'V', 'opteig: N or V expected')
            assert(opttriang == 'L' or opttriang == 'U', 'opttriang: L or U expected')
            local E = torch.RealTensor()
            local V = torch.RealTensor()
            C.THRealTensor_syev(E, V, A, opteig, opttriang)
            return E, V
         end
   }

   register{
      nomethod = true,
      name = "symeig",
      {name="E", type="torch.RealTensor"},
      {name="V", type="torch.RealTensor"},
      {name="A", type="torch.RealTensor"},
      {name="opteig", type="string", default='N'},
      {name="opttriang", type="string", default='U'},
      call =
         function(E, V, A, opteig, opttriang)
            assert(opteig == 'N' or opteig == 'V', 'opteig: N or V expected')
            assert(opttriang == 'L' or opttriang == 'U', 'opttriang: L or U expected')
            C.THRealTensor_syev(E, V, A, opteig, opttriang)
            return E, V
         end
   }

   register{
      name = "eig",
      {name="A", type="torch.RealTensor"},
      {name="opteig", type="string", default='N'},
      call =
         function(A, opteig)
            assert(opteig == 'N' or opteig == 'V', 'opteig: N or V expected')
            local E = torch.RealTensor()
            local V = torch.RealTensor()
            C.THRealTensor_geev(E, V, A, opteig)
            return E, V
         end
   }

   register{
      nomethod = true,
      name = "eig",
      {name="E", type="torch.RealTensor"},
      {name="V", type="torch.RealTensor"},
      {name="A", type="torch.RealTensor"},
      {name="opteig", type="string", default='N'},
      call =
         function(E, V, A, opteig, opttriang)
            assert(opteig == 'N' or opteig == 'V', 'opteig: N or V expected')
            C.THRealTensor_geev(E, V, A, opteig)
            return E, V
         end
   }

   register{
      name = "svd",
      {name="A", type="torch.RealTensor"},
      {name="opteig", type="string", default='S'},
      call =
         function(A, opteig)
            assert(opteig == 'S' or opteig == 'A', 'opteig: S or A expected')
            local U = torch.RealTensor()
            local S = torch.RealTensor()
            local V = torch.RealTensor()
            C.THRealTensor_gesvd(U, S, V, A, opteig)
            return U, S, V
         end
   }

   register{
      nomethod = true,
      name = "svd",
      {name="U", type="torch.RealTensor"},
      {name="S", type="torch.RealTensor"},
      {name="V", type="torch.RealTensor"},
      {name="A", type="torch.RealTensor"},
      {name="opteig", type="string", default='S'},
      call =
         function(U, S, V, A, opteig)
            assert(opteig == 'S' or opteig == 'A', 'opteig: S or A expected')
            C.THRealTensor_gesvd(U, S, V, A, opteig)
            return U, S, V
         end
   }

   for _, name in ipairs{'inverse', 'potri', 'potrf'} do
      local cname = name == 'inverse' and 'getri' or name
      local func = C['THRealTensor_' .. cname]
      register{
         name = name,
         {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
         {name="src", type="torch.RealTensor", method={opt=true}},
         call =
            function(dst, src)
               local res = src and dst or torch.RealTensor()
               src = src or dst
               func(res, src)
               return res
            end
      }
      
   end
   
end
