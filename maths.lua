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

local function defaulttensortype()
   return class.type(torch.Tensor)
end

register{
   name = "fill",
   {name="dst", type="torch.RealTensor"},
   {name="value", type="number"},
   call =
      function(dst, value)
         C.THRealTensor_fill(dst, value)
         return dst
      end
}

register{
   name = "zero",
   {name="dst", type="torch.RealTensor"},
   call =
      function(dst)
         C.THRealTensor_zero(dst)
         return dst
      end
}

register{
   name = "dot",
   {name="src1", type="torch.RealTensor"},
   {name="src2", type="torch.RealTensor"},
   call =
      function(src1, src2)
         return tonumber(C.THRealTensor_dot(src1, src2))
      end
}

register{
   name = "min",
   {name="src", type="torch.RealTensor"},
   call =
      function(src)
         return tonumber(C.THRealTensor_minall(src))
      end
}

register{
   name = "min",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="idx", type="torch.LongTensor", opt=true},
   {name="src", type="torch.RealTensor", method={opt=true}},
   {name="dim", type="number"},
   call =
      function(dst, idx, src, dim)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         idx = idx or torch.LongTensor()
         C.THRealTensor_min(res, idx, src, dim-1)
         idx:add(1)
         return res, idx
      end
}

register{
   name = "max",
   {name="src", type="torch.RealTensor"},
   call =
      function(src)
         return tonumber(C.THRealTensor_maxall(src))
      end
}

register{
   name = "max",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="idx", type="torch.LongRealTensor", opt=true},
   {name="src", type="torch.RealTensor", method={opt=true}},
   {name="dim", type="number"},
   call =
      function(dst, idx, src, dim)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         idx = idx or torch.LongTensor()
         C.THRealTensor_max(res, idx, src, dim-1)
         idx:add(1)
         return res, idx
      end
}

register{
   name = "sum",
   {name="src", type="torch.RealTensor"},
   call =
      function(src)
         return tonumber(C.THRealTensor_sumall(src))
      end
}

register{
   name = "sum",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src", type="torch.RealTensor", method={opt=true}},
   {name="dim", type="number"},
   call =
      function(dst, src, dim)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         C.THRealTensor_max(res, src, dim-1)
         return res
      end
}

register{
   name = "add",
   {name="dst", type="torch.RealTensor", opt=true,  method={opt=false}},
   {name="src", type="torch.RealTensor", method={defaulta="self"}},
   {name="value", type="number"},
   call =
      function(dst, src, value)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         res:resizeAs(src)
         C.THRealTensor_add(res, src, value)
         return res
      end
}

register{
   name = "add",
   {name="dst", type="torch.RealTensor", opt=true,   method={opt=false}},
   {name="src1", type="torch.RealTensor", opt=false, method={defaulta="self"}},
   {name="value", type="number", default=1},
   {name="src2", type="torch.RealTensor"},
   call =
      function(dst, src1, value, src2)
         local res = src1 and dst or torch.RealTensor()
         src1 = src1 or dst
         res:resizeAs(src1)
         C.THRealTensor_cadd(dst, src1, value, src2)
         return res
      end
}

register{
   name = "mul",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src", type="torch.RealTensor", method={defaulta="self"}},
   {name="value", type="number"},
   call =
      function(dst, src, value)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         res:resizeAs(src)
         C.THRealTensor_mul(res, src, value)
         return res
      end
}

register{
   name = "cmul",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src1", type="torch.RealTensor", method={defaulta="self"}},
   {name="src2", type="torch.RealTensor"},
   call =
      function(dst, src1, src2)
         local res = src1 and dst or torch.RealTensor()
         src1 = src1 or dst
         res:resizeAs(src1)
         C.THRealTensor_cmul(dst, src1, src2)
         return res
      end
}


register{
   name = "div",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src", type="torch.RealTensor", method={defaulta="self"}},
   {name="value", type="number"},
   call =
      function(dst, src, value)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         res:resizeAs(src)
         C.THRealTensor_div(res, src, value)
         return res
      end
}

register{
   name = "cdiv",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src1", type="torch.RealTensor", method={defaulta="self"}},
   {name="src2", type="torch.RealTensor"},
   call =
      function(dst, src1, src2)
         local res = src1 and dst or torch.RealTensor()
         src1 = src1 or dst
         res:resizeAs(src1)
         C.THRealTensor_cdiv(dst, src1, src2)
         return res
      end
}

register{
   name = "addcmul",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src", type="torch.RealTensor", method={defaulta="self"}},
   {name="value", type="number", default=1},
   {name="src1", type="torch.RealTensor"},
   {name="src2", type="torch.RealTensor"},
   call =
      function(dst, src, value, src1, src2)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         C.THRealTensor_addcmul(res, src, value, src1, src2)
         return res
      end
}

register{
   name = "addcdiv",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src", type="torch.RealTensor", method={defaulta="self"}},
   {name="value", type="number", default=1},
   {name="src1", type="torch.RealTensor"},
   {name="src2", type="torch.RealTensor"},
   call =
      function(dst, src, value, src1, src2)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         C.THRealTensor_addcdiv(res, src, value, src1, src2)
         return res
      end
}

for _, f in ipairs{
   {name="mv", addname="addmv", arg1="mat", arg2="vec"},
   {name="mm", addname="addmm", arg1="mat", arg2="mat"},
   {name="ger", addname="addr", arg1="vec1", arg2="vec2"}} do

   local func = C["THRealTensor_" .. f.addname]

   register{
      name = f.name,
      {name=f.arg1, type="torch.RealTensor"},
      {name=f.arg2, type="torch.RealTensor"},
      call =
         function(arg1, arg2)
            local res = torch.RealTensor()
            func(res, 0, res, 1, arg1, arg2)
            return res
         end
   }

   register{
      name = f.addname,
      {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
      {name="src", type='torch.RealTensor', method={defaulta="self"}},
      {name="alpha", type='number', default=1},
      {name="mat", type='torch.RealTensor'}, -- could check dim
      {name="vec", type='torch.RealTensor'},
      call =
         function(dst, src, alpha, mat, vec)
            local res = src and dst or torch.RealTensor()
            src = src or self
            func(res, 1, src, alpha, mat, vec)
            return res
         end
   }

   register{
      name = f.addname,
      {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
      {name="beta", type='number'},
      {name="src", type='torch.RealTensor', method={defaulta="self"}},
      {name="alpha", type='number'},
      {name="mat", type='torch.RealTensor'}, -- could check dim
      {name="vec", type='torch.RealTensor'},
      call =
         function(dst, beta, src, alpha, mat, vec)
            local res = src and dst or torch.RealTensor()
            src = src or self
            func(res, beta, src, alpha, mat, vec)
            return res
         end
   }

end

register{
   name = "numel",
   {name="src", type="torch.RealTensor"},
   call =
      function(src)
         return tonumber(THRealTensor_nElement(src))
      end
}

-- NYI
-- register{
--    name = "prod",
--    {name="src", type="torch.RealTensor"},
--    call =
-- }

register{
   name = "prod",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src", type="torch.RealTensor", method={opt=true}},
   {name="dim", type="number"},
   call =
      function(dst, src, dim)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         C.THRealTensor_prod(res, src, dim-1)
         return res
      end
}

register{
   name = "cumsum",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src", type="torch.RealTensor", method={opt=true}},
   {name="dim", type="number"},
   call =
      function(dst, src, dim)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         C.THRealTensor_cumsum(res, src, dim-1)
         return res
      end
}

register{
   name = "cumprod",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src", type="torch.RealTensor", method={opt=true}},
   {name="dim", type="number"},
   call =
      function(dst, src, dim)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         C.THRealTensor_cumprod(res, src, dim-1)
         return res
      end
}

register{
   name = "sign",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src", type="torch.RealTensor", method={opt=true}},
   call =
      function(dst, src)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         C.THRealTensor_sign(res, src)
         return res
      end
}

register{
   name = "trace",
   {name="src", type="torch.RealTensor"},
   call =
      function(src)
         return tonumber(C.THRealTensor_trace(src))
      end
}

register{
   name = "cross",
   {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
   {name="src1", type="torch.RealTensor"},
   {name="src2", type="torch.RealTensor"},
   {name="dim", type="number", default=0},
   call =
      function(dst, src1, src2, dim)
         dst = dst or torch.RealTensor()
         C.THRealTensor_cross(dst, src1, src2, dim-1)
         return dst
      end
}

register{
   name = "diag",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="src", type='torch.RealTensor', method={opt=true}},
   {name="k", type='number', default=0},
   call =
      function(dst, src, k)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         C.THRealTensor_diag(dst, src, k)
         return res
      end
}

register{
   name = "range",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="xmin", type='number'},
   {name="xmax", type='number'},
   {name="step", type='number', default=1},
   call =
      function(dst, xmin, xmax, step)
         local res = dst or torch.RealTensor()
         C.THRealTensor_range(res, xmin, xmax, step)
         return res
      end
}

register{
   name = "randperm",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="generator", type='torch.Generator', opt=true},
   {name="n", type='number'},
   call =
      function(dst, generator, n)
         local res = dst or torch.RealTensor()
         generator = generator or torch.__generator
         C.THRealTensor_randperm(res, generator, n)
         C.THRealTensor_add(res, res, 1)
         return res
      end
}

register{
   name = "reshape",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="src", type='torch.RealTensor', method={opt=true}},
   {name="size", type='numbers'},
   call =
      function(dst, src, size)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         C.THRealTensor_reshape(res, src, size)
         return res
      end
}

register{
   name = "sort",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="idx", type='torch.LongTensor', opt=true},
   {name="src", type='torch.RealTensor', method={opt=true}},
   {name="dim", type='number', opt=true},
   {name="descend", type='boolean', default=false},
   call =
      function(dst, idx, src, dim, descend)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         idx = idx or torch.LongTensor()
         dim = dim or src:nDimension()
         C.THRealTensor_sort(res, idx, src, dim-1, descend and 1 or 0)
         idx:add(1)
         return res, idx
      end
}

register{
   name = "tril",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="src", type='torch.RealTensor', method={opt=true}},
   {name="k", type='number', default=0},
   call =
      function(dst, src, k)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         C.THRealTensor_tril(res, src, k)
         return res
      end
}

register{
   name = "triu",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="src", type='torch.RealTensor', method={opt=true}},
   {name="k", type='number', default=0},
   call =
      function(dst, src, k)
         local res = src and dst or torch.RealTensor()
         src = src or dst
         C.THRealTensor_triu(res, src, k)
         return res
      end
}

register{
   name = "cat",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="src1", type='torch.RealTensor', method={opt=true}},
   {name="src2", type='torch.RealTensor'},
   {name="dim", type='number', opt=true},
   call =
      function(dst, src1, src2, dim)
         local res = src1 and dst or torch.RealTensor()
         src1 = src1 or dst
         dim = dim or src1:nDimension()
         C.THRealTensor_cat(res, src1, src2, dim-1)
         return res
      end
}

-- comparison
for _,name in ipairs{'lt','gt','le','ge','eq','ne'} do
   local func_val = C['THRealTensor_' .. name .. 'Value']
   local func_valT = C['THRealTensor_' .. name .. 'ValueT']
   local func_tens = C['THRealTensor_' .. name .. 'Tensor']
   local func_tensT = C['THRealTensor_' .. name .. 'TensorT']

   register{
      nomethod = true,
      name = name,
      {name="dst", type='torch.ByteTensor', opt=true},
      {name="src", type='torch.RealTensor'},
      {name="value", type='number'},
      call =
         function(dst, src, value)
            local res = dst or torch.ByteTensor()
            func_val(res, src, value) -- DEBUG: TH args is not right (it is not like a method...)
            return res
         end
   }

   register{
      nofunction = true,
      name = name,
      {name="src", type='torch.RealTensor'},
      {name="value", type='number'},
      {name="dst", type='torch.ByteTensor', opt=true},
      call =
         function(src, value, dst)
            local res = dst or torch.ByteTensor()
            func_val(res, src, value)
            return res
         end
   }

   register{
      nomethod = true,
      name = name,
      {name="dst", type='torch.RealTensor', opt=true},
      {name="src", type='torch.RealTensor'},
      {name="value", type='number'},
      call =
         function(dst, src, value)
            local res = dst or torch.RealTensor()
            func_valT(res, src, value) -- DEBUG: TH args is not right (it is not like a method...)
            return res
         end
   }

   register{
      nofunction = true,
      name = name,
      {name="src", type='torch.RealTensor'},
      {name="value", type='number'},
      {name="dst", type='torch.RealTensor', opt=true},
      call =
         function(src, value, dst)
            local res = dst or torch.RealTensor()
            func_valT(res, src, value)
            return res
         end
   }

   register{
      nomethod = true,
      name = name,
      {name="dst", type='torch.ByteTensor', opt=true},
      {name="src1", type='torch.RealTensor'},
      {name="src2", type='torch.RealTensor'},
      call =
         function(dst, src1, src2)
            local res = dst or torch.ByteTensor()
            func_tens(res, src1, src2) -- DEBUG: TH args is not right (it is not like a method...)
            return res
         end
   }

   register{
      nofunction = true,
      name = name,
      {name="src1", type='torch.RealTensor'},
      {name="src2", type='torch.RealTensor'},
      {name="dst", type='torch.ByteTensor', opt=true},
      call =
         function(src1, src2, dst)
            local res = dst or torch.ByteTensor()
            func_tens(res, src1, src2)
            return res
         end
   }

   register{
      nomethod = true,
      name = name,
      {name="dst", type='torch.RealTensor', opt=true},
      {name="src1", type='torch.RealTensor'},
      {name="src2", type='torch.RealTensor'},
      call =
         function(dst, src1, src2)
            local res = dst or torch.RealTensor()
            func_tensT(res, src1, src2) -- DEBUG: TH args is not right (it is not like a method...)
            return res
         end
   }

   register{
      nofunction = true,
      name = name,
      {name="src1", type='torch.RealTensor'},
      {name="src2", type='torch.RealTensor'},
      {name="dst", type='torch.RealTensor', opt=true},
      call =
         function(src1, src2, dst)
            local res = dst or torch.RealTensor()
            func_tensT(res, src1, src2)
            return res
         end
   }

end

-- copy
register{
   name = "copy",
   {name="dst", type='torch.RealTensor'},
   {name="src", type='torch.RealTensor'},
   call =
      function(dst, src)
         C.THRealTensor_copy(dst, src)
         return dst
      end
}

register{
   name = "copy",
   {name="dst", type='torch.RealTensor'},
   {name="src", type='torch.ByteTensor'},
   call =
      function(dst, src)
         C.THRealTensor_copyByte(dst, src)
         return dst
      end
}

register{
   name = "copy",
   {name="dst", type='torch.RealTensor'},
   {name="src", type='torch.CharTensor'},
   call =
      function(dst, src)
         C.THRealTensor_copyChar(dst, src)
         return dst
      end
}

register{
   name = "copy",
   {name="dst", type='torch.RealTensor'},
   {name="src", type='torch.ShortTensor'},
   call =
      function(dst, src)
         C.THRealTensor_copyShort(dst, src)
         return dst
      end
}

register{
   name = "copy",
   {name="dst", type='torch.RealTensor'},
   {name="src", type='torch.IntTensor'},
   call =
      function(dst, src)
         C.THRealTensor_copyInt(dst, src)
         return dst
      end
}

register{
   name = "copy",
   {name="dst", type='torch.RealTensor'},
   {name="src", type='torch.LongTensor'},
   call =
      function(dst, src)
         C.THRealTensor_copyLong(dst, src)
         return dst
      end
}

register{
   name = "copy",
   {name="dst", type='torch.RealTensor'},
   {name="src", type='torch.FloatTensor'},
   call =
      function(dst, src)
         C.THRealTensor_copyFloat(dst, src)
         return dst
      end
}

register{
   name = "copy",
   {name="dst", type='torch.RealTensor'},
   {name="src", type='torch.DoubleTensor'},
   call =
      function(dst, src)
         C.THRealTensor_copyDouble(dst, src)
         return dst
      end
}

-- creation
register{
   name = "zeros",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="size", type="table"},
   {name="typename", type="string", defaultf=defaulttensortype}, -- namedispatch
   call =
      function(dst, size, typename)
         if dst then
            size = torch.LongStorage(size)
            C.THRealTensor_resize(dst, size, nil)
            C.THRealTensor_zero(dst)
         else
            dst = class.metatable(typename).new()
            dst:zeros(size)
         end
         return dst
      end
}

register{
   name = "zeros",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="dim1", type="number"},
   {name="dim2", type="number", default=0},
   {name="dim3", type="number", default=0},
   {name="dim4", type="number", default=0},
   {name="typename", type="string", defaultf=defaulttensortype}, -- namedispatch
   call =
      function(dst, dim1, dim2, dim3, dim4, typename)
         if dst then
            C.THRealTensor_resize4d(dst, dim1, dim2, dim3, dim4)
            C.THRealTensor_zero(dst)
         else
            dst = class.metatable(typename).new()
            dst:zeros(dim1, dim2, dim3, dim4)
         end
         return dst
      end
}

register{
   name = "ones",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="size", type="table"},
   {name="typename", type="string", defaultf=defaulttensortype}, -- namedispatch
   call =
      function(dst, size, typename)
         if dst then
            size = torch.LongStorage(size)
            C.THRealTensor_resize(dst, size, nil)
            C.THRealTensor_fill(dst, 1)
         else
            dst = class.metatable(typename).new()
            dst:ones(size)
         end
         return dst
      end
}

register{
   name = "ones",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="dim1", type="number"},
   {name="dim2", type="number", default=0},
   {name="dim3", type="number", default=0},
   {name="dim4", type="number", default=0},
   {name="typename", type="string", defaultf=defaulttensortype}, -- namedispatch
   call =
      function(dst, dim1, dim2, dim3, dim4, typename)
         if dst then
            C.THRealTensor_resize4d(dst, dim1, dim2, dim3, dim4)
            C.THRealTensor_fill(dst, 1)
         else
            dst = class.metatable(typename).new()
            dst:ones(dim1, dim2, dim3, dim4)
         end
         return dst
      end
}

register{
   name = "rand",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="size", type="table"},
   {name="typename", type="string", defaultf=defaulttensortype}, -- namedispatch
   call =
      function(dst, size, typename)
         if dst then
            size = torch.LongStorage(size)
            C.THRealTensor_resize(dst, size, nil)
            C.THRealTensor_uniform(dst, torch.__generator, 0, 1)
         else
            dst = class.metatable(typename).new()
            dst:rand(size)
         end
         return dst
      end
}

register{
   name = "rand",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="dim1", type="number"},
   {name="dim2", type="number", default=0},
   {name="dim3", type="number", default=0},
   {name="dim4", type="number", default=0},
   {name="typename", type="string", defaultf=defaulttensortype}, -- namedispatch
   call =
      function(dst, dim1, dim2, dim3, dim4, typename)
         if dst then
            C.THRealTensor_resize4d(dst, dim1, dim2, dim3, dim4)
            C.THRealTensor_uniform(dst, torch.__generator, 0, 1)
         else
            dst = class.metatable(typename).new()
            dst:rand(dim1, dim2, dim3, dim4)
         end
         return dst
      end
}

register{
   name = "randn",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="size", type="table"},
   {name="typename", type="string", defaultf=defaulttensortype}, -- namedispatch
   call =
      function(dst, size, typename)
         if dst then
            size = torch.LongStorage(size)
            C.THRealTensor_resize(dst, size, nil)
            C.THRealTensor_normal(dst, torch.__generator, 0, 1)
         else
            dst = class.metatable(typename).new()
            dst:randn(size)
         end
         return dst
      end
}

register{
   name = "randn",
   {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
   {name="dim1", type="number"},
   {name="dim2", type="number", default=0},
   {name="dim3", type="number", default=0},
   {name="dim4", type="number", default=0},
   {name="typename", type="string", defaultf=defaulttensortype}, -- namedispatch
   call =
      function(dst, dim1, dim2, dim3, dim4, typename)
         if dst then
            C.THRealTensor_resize4d(dst, dim1, dim2, dim3, dim4)
            C.THRealTensor_normal(dst, torch.__generator, 0, 1)
         else
            dst = class.metatable(typename).new()
            dst:randn(dim1, dim2, dim3, dim4)
         end
         return dst
      end
}

-- apply and map
register{
   name = "apply",
   {name="src", type="torch.RealTensor"},
   {name="func", type="function"},
   call =
      function(src, func)
         local function rawfunc(sz, x, inc)
            for i=0,sz-1 do
               local res = func(tonumber(x[i*inc]))
               if res then
                  x[i*inc] = res
               end
            end
         end
         torch.rawapply(src, rawfunc)
      end
}

-- float only
if "real" == "double" or "real" == "float" then

   register{
      name = "linspace",
      {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
      {name="a", type="number"},
      {name="b", type="number"},
      {name="n", type="number", default=100},
      {name="typename", type="string", defaultf=defaulttensortype}, -- namedispatch
      call =
         function(dst, a, b, n, typename)
            if dst then
               C.THRealTensor_linspace(dst, a, b, n)
            else
               dst = class.metatable(typename).new()
               dst:linspace(a, b, n)
            end
            return dst
         end
   }

   register{
      name = "logspace",
      {name="dst", type='torch.RealTensor', opt=true, method={opt=false}},
      {name="a", type="number"},
      {name="b", type="number"},
      {name="n", type="number", default=100},
      {name="typename", type="string", defaultf=defaulttensortype}, -- namedispatch
      call =
         function(dst, a, b, n, typename)
            if dst then
               C.THRealTensor_logspace(dst, a, b, n)
            else
               dst = class.metatable(typename).new()
               dst:logspace(a, b, n)
            end
            return dst
         end
   }

   register{
      name = "mean",
      {name="src", type="torch.RealTensor"},
      call = C.THRealTensor_meanall
   }

   register{
      name = "mean",
      {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},  -- could be torch.DoubleRealTensor for other types
      {name="src", type="torch.RealTensor", method={opt=true}},
      {name="dim", type="number"},
      call =
         function(dst, src, dim)
            local res = src and dst or torch.RealTensor()
            src = src or dst
            C.THRealTensor_mean(dst, src, n, dim-1)
            return res
         end
   }

   register{
      name = "std",
      {name="src", type="torch.RealTensor"},
      {name="flag", type="boolean", default=false}, -- NYI
      call =
         function(src, flag)
            return C.THRealTensor_stdall(src)
         end
   }
   
   register{
      name = "std",
      {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
      {name="src", type="torch.RealTensor", method={opt=true}},
      {name="dim", type="number"},
      {name="flag", type="boolean", default=false},
      call =
         function(dst, src, dim, flag)
            local res = src and dst or torch.RealTensor()
            src = src or dst
            C.THRealTensor_std(dst, src, dim-1, flag and 1 or 0)
            return res
         end
   }

   register{
      name = "var",
      {name="src", type="torch.RealTensor"},
      {name="flag", type="boolean", default=false},
      call =
         function(src, flag)
            return C.THRealTensor_varall(src) -- NYI
         end
   }

   register{
      name = "var",
      {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
      {name="src", type="torch.RealTensor", method={opt=true}},
      {name="dim", type="number"},
      {name="flag", type="boolean", default=false},
      call =
         function(dst, src, dim, flag)
            local res = src and dst or torch.RealTensor()
            src = src or dst
            C.THRealTensor_var(dst, src, dim-1, flag and 1 or 0)
            return res
         end
   }

   register{
      name = "norm",
      {name="src", type="torch.RealTensor"},
      {name="n", type="number", default=2},
      call = C.THRealTensor_normall
   }

   register{
      name = "norm",
      {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
      {name="src", type="torch.RealTensor", method={opt=true}},
      {name="n", type="number", default=2},
      {name="dim", type="number"},
      call =
         function(dst, src, n, dim)
            local res = src and dst or torch.RealTensor()
            src = src or dst
            C.THRealTensor_norm(dst, src, n, dim-1)
            return res
         end
   }

   register{
      name = "dist",
      {name="dst", type="torch.RealTensor"},
      {name="src", type="torch.RealTensor"},
      {name="n", type="number", default=2},
      call =
         function(dst, src, n)
            local res = src and dst or torch.RealTensor()
            src = src or dst
            local dist = C.THRealTensor_dist(dst, src, n)
            return dist
         end
   }

   register{
      name = "histc",
      {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
      {name="src", type="torch.RealTensor", method={opt=true}},
      {name="nbin", type="number", default=100},
      {name="minvalue", type="number", opt=0},
      {name="maxvalue", type="number", opt=0},
      call =
         function(dst, src, nbin, minvalue, maxvalue)
            local res = src and dst or torch.RealTensor()
            src = src or dst
            C.THRealTensor_histc(dst, src, nbin, minvalue, maxvalue)
            return res
         end
   }

   for _,name in ipairs{'log', 'log1p', 'exp', 'cos', 'acos', 'cosh', 'sin', 'asin',
                        'sinh', 'tan', 'atan', 'tanh', 'sqrt', 'ceil', 'floor', 'abs'} do

      local func = C['THRealTensor_' .. name]
      local func_number = math[name]

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

      register{
         name = name,
         {name="x", type="number"},
         call = func_number
      }
   end

   register{
      name = "atan2",
      {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
      {name="src1", type="torch.RealTensor", method={opt=true}},
      {name="src2", type="torch.RealTensor"},
      call =
         function(dst, src1, src2)
            local res = src1 and dst or torch.RealTensor()
            src1 = src1 or dst
            C.THRealTensor_atan2(res, src1, src2)
            return res
         end
   }

   register{
      name = "atan2",
      {name="x", type="number"},
      {name="y", type="number"},
      call = math.atan2
   }

   register{
      name = "pow",
      {name="dst", type="torch.RealTensor", opt=true, method={opt=false}},
      {name="src", type="torch.RealTensor", method={defaulta="self"}},
      {name="value", type="number"},
      call =
         function(dst, src, value)
            local res = src and dst or torch.RealTensor()
            src = src or dst
            C.THRealTensor_pow(res, src)
            return res
         end
   }

end
