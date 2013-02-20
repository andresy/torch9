local ffi = require 'ffi'
local namedispatch = require 'torch.namedispatch'
local dispatch = require 'torch.dispatch'
local argcheck = require 'torch.argcheck'
local torch = require 'torch'
local C = require 'torch.clib'

print('we loaded real')

local register =
   argcheck{
   
   {{name="name", type="string"},
    {name="args", type="table"},
    {name="dispatchidx", type="number", default=1},
    {name="dispatchfunc", type="function", opt=true}},
   
   function (name, args, dispatchidx, dispatchfunc)
      local func, method = argcheck(args)
      torch[name] = torch[name] or (
         dispatchfunc and dispatch(dispatchfunc) or dispatch(dispatchidx)
      )
      dispatch(torch[name], "torch.Tensor", func)
      torch.Tensor[name] = method
   end
}

register(
   "fill",
   {
      {
         {name="dst", type="torch.Tensor"},
         {name="value", type="number"},
         self="dst"
      },
      function(dst, value)
         torch.apply(dst,
                     function(sz, dst, str)
                        C.th_fill_real(sz, value, dst, str)
                     end)
         return dst
      end
   }
)

register(
   "zero",
   {
      {
         {name="dst", type="torch.Tensor"},
         self="dst"
      },
      function(dst)
         torch.apply(dst, C.th_zero_real)
         return dst
      end
   }
)

register(
   "dot",
   {
      {
         {name="src1", type="torch.Tensor"},
         {name="src2", type="torch.Tensor"},
         self="src1"
      },
      function(src1, src2)
         local sum = 0
         torch.apply2(src1, src2,
                      function(sz, src1, strsrc1, src2, strsrc2)
                         sum = sum + C.th_dot_real(sz, src1, strsrc1, src2, strsrc2)
                      end)
         return sum
      end
   }
)

register(
   "min",
   {
      {
         {name="src", type="torch.Tensor"},
         self="src"
      },
      function(src)
         local min = math.huge
         local minptr = ffi.new('real[1]')
         local idsrcptr = ffi.new('long[1]')
         torch.apply(src,
                     function(sz, src, str)
                        C.th_min_real(sz, src, str, minptr, idsrcptr)
                        min = math.min(min, minptr[0])
                     end)
         return min
      end,
      
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="idx", type="torch.LongTensor", opt=true},
         {name="src", type="torch.Tensor", method={opt=true}},
         {name="dim", type="number"},
         self="dst"
      },
      function(dst, idx, src, dim)
         local res = src and dst or torch.Tensor()
         src = src or dst
         idx = idx or torch.LongTensor()
         
         local size = src:size()
         size[dim] = 1
         res:resize(size)
         idx:resize(size)
         torch.dimapply3(idx, src, res, dim,
                         function(idxsz, idx, idxst,
                                  srcsz, src, srcst,
                                  ressz, res, resst)
                            C.th_min_real(srcsz, src, srcst, res, idx)
                         end)
         return res, idx
      end
   }
)

register(
   "max",
   {
      {
         {name="src", type="torch.Tensor"},
         self="src"
      },
      function(src)
         local max = -math.huge
         local maxptr = ffi.new('real[1]')
         local idsrcptr = ffi.new('long[1]')
         torch.apply(src,
                     function(sz, src, str)
                        C.th_max_real(sz, src, str, maxptr, idsrcptr)
                        max = math.max(max, maxptr[0])
                     end)
         return max
      end,
      
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="idx", type="torch.LongTensor", opt=true},
         {name="src", type="torch.Tensor", method={opt=true}},
         {name="dim", type="number"},
         self="dst"
      },
      function(dst, idx, src, dim)
         local res = src and dst or torch.Tensor()
         src = src or dst
         idx = idx or torch.LongTensor()
         
         local size = src:size()
         size[dim] = 1
         res:resize(size)
         idx:resize(size)
         torch.dimapply3(idx, src, res, dim,
                         function(idxsz, idx, idxst,
                                  srcsz, src, srcst,
                                  ressz, res, resst)
                            C.th_max_real(srcsz, src, srcst, res, idx)
                         end)
         return res, idx
      end
   }
)

register(
   "sum",
   {
      {
         {name="src", type="torch.Tensor"},
         self="src"
      },
      function(src)
         local sum = 0
         torch.apply(src, function(sz, src, str)
                             sum = sum + C.th_sum_real(sz, src, str)
                          end)
         return sum
      end,
      
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="src", type="torch.Tensor", method={opt=true}},
         {name="dim", type="number"},
         self="dst"
      },
      function(dst, src, dim)
         local res = src and dst or torch.Tensor()
         src = src or dst

         local size = src:size()
         size[dim] = 1
         res:resize(size)
         torch.dimapply2(src, res, dim,
                         function(srcsz, src, srcst,
                                  ressz, res, resst)
                            res[0] = C.th_sum_real(srcsz, src, srcst)
                         end)
         return res
      end
   }
)

register(
   "prod",
   {
      {
         {name="src", type="torch.Tensor"},
         self="src"
      },
      function(src)
         local prod = (src:nElement() > 0) and 1 or 0
         torch.apply(src,
                     function(sz, src, str)
                        prod = prod * tonumber( C.th_prod_real(sz, src, str) )
                     end)
         return prod
      end,
      
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="src", type="torch.Tensor", method={opt=true}},
         {name="dim", type="number"},
         self="dst"
      },
      function(dst, src, dim)
         local res = src and dst or torch.Tensor()
         src = src or dst

         local size = src:size()
         size[dim] = 1
         res:resize(size)
         torch.dimapply2(src, res, dim,
                         function(srcsz, src, srcst,
                                  ressz, res, resst)
                            res[0] = C.th_prod_real(srcsz, src, srcst)
                         end)
         return res
      end
   }
)

register(
   "cumsum",
   {
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="src", type="torch.Tensor", method={opt=true}},
         {name="dim", type="number"},
         self="dst"
      },
      function(dst, src, dim)
         local res = src and dst or torch.Tensor()
         src = src or dst

         res:resizeAs(src)
         torch.dimapply2(src, res, dim,
                         function(szsrc, src, incsrc,
                                  szres, res, incres)
                            C.th_cumsum_real(szsrc, src, incsrc, res, incres)
                         end)
         return res
      end
   }
)

register(
   "cumprod",
   {
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="src", type="torch.Tensor", method={opt=true}},
         {name="dim", type="number"},
         self="dst"
      },
      function(dst, src, dim)
         local res = src and dst or torch.Tensor()
         src = src or dst

         res:resizeAs(src)
         torch.dimapply2(src, res, dim,
                         function(szsrc, src, incsrc,
                                  szres, res, incres)
                            C.th_cumprod_real(szsrc, src, incsrc, res, incres)
                         end)
         return res
      end
   }
)

-- float only

register(
   "norm",
   {
      {
         {name="src", type="torch.Tensor"},
         {name="n", type="number", default=2},
         self="src"
      },
      function(src, n)
         local norm = 0
         torch.apply(src,
                     function(sz, src, str)
                        norm = norm + C.th_norm_real(sz, n, 0, src, str)
                     end)
         return math.pow(norm, 1/n)
      end,
      
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="src", type="torch.Tensor", method={opt=true}},
         {name="n", type="number", default=2},
         {name="dim", type="number"},
         self="dst"
      },
      function(dst, src, n, dim)
         local res = src and dst or torch.Tensor()
         src = src or dst

         local size = src:size()
         size[dim] = 1
         res:resize(size)
         torch.dimapply2(src, res, dim,
                         function(srcsz, src, srcst,
                                  ressz, res, resst)
                            res[0] = C.th_norm_real(srcsz, n, 1, src, srcst)
                         end)
         return res
      end
   }
)

register(
   "mean",
   {
      {
         {name="src", type="torch.Tensor"},
         self="src"
      },
      function(src)
         local sum = 0
         torch.apply(src, function(sz, src, str)
                             sum = sum + C.th_sum_real(sz, src, str)
                          end)
         return sum / src:nElement()
      end,
      
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},  -- could be torch.DoubleTensor for other types
         {name="src", type="torch.Tensor", method={opt=true}},
         {name="dim", type="number"}
      },
      function(dst, src, dim)
         local res = src and dst or torch.Tensor()
         src = src or dst

         local size = src:size()
         size[dim] = 1
         res:resize(size)
         torch.dimapply2(src, res, dim,
                         function(srcsz, src, srcst,
                                  ressz, res, resst)
                            res[0] = C.th_sum_real(srcsz, src, srcst) / srcsz
                         end)
         return dst
      end
   }
)

register(
   "std",
   {
      {
         {name="src", type="torch.Tensor"},
         {name="flag", type="boolean", default=false},
         self="src"
      },
      function(src, flag)
         local sum = 0
         local sum2 = 0
         local n = src:nElement()
         torch.apply(src,
                     function(sz, src, str)
                        sum = sum + C.th_sum_real(sz, src, str)
                        sum2 = sum2 + C.th_sum2_real(sz, src, str)
                     end)
         if flag then
            return math.sqrt((sum2 - sum*sum/n)/n)
         else
            return math.sqrt((sum2 - sum*sum/n)/(n-1))
         end
      end,
      
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},  -- could be torch.DoubleTensor for other types
         {name="src", type="torch.Tensor", method={opt=true}},
         {name="dim", type="number"},
         {name="flag", type="boolean", default=false},
         self="dst"
      },
      function(dst, src, dim, flag)
         local res = src and dst or torch.Tensor()
         src = src or dst

         local size = src:size()
         size[dim] = 1
         res:resize(size)
         local sumptr = ffi.new('real[1]')
         local sum2ptr = ffi.new('real[1]')
         if flag then
            torch.dimapply2(src, res, dim,
                            function(srcsz, src, srcst,
                                     ressz, res, resst)
                               C.th_sum_sum2_real(srcsz, src, srcst, sumptr, sum2ptr)
                               res[0] = math.sqrt((sum2ptr[0] - sumptr[0]*sumptr[0]/srcsz)/srcsz)
                            end)
         else
            torch.dimapply2(src, res, dim,
                            function(srcsz, src, srcst,
                                     ressz, res, resst)
                               C.th_sum_sum2_real(srcsz, src, srcst, sumptr, sum2ptr)
                               res[0] = math.sqrt((sum2ptr[0] - sumptr[0]*sumptr[0]/srcsz)/(srcsz-1))
                            end)
         end
         return res
      end
   }
)

register(
   "var",
   {
      {
         {name="src", type="torch.Tensor"},
         {name="flag", type="boolean", default=false},
         self="src"
      },
      function(src, flag)
         local sum = 0
         local sum2 = 0
         local n = src:nElement()
         torch.apply(src, function(sz, src, str)
                             sum = sum + C.th_sum_real(sz, src, str)
                             sum2 = sum2 + C.th_sum2_real(sz, src, str)
                          end)
         if flag then
            return (sum2 - sum*sum/n)/n
         else
            return (sum2 - sum*sum/n)/(n-1)
         end
      end,
      
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},  -- could be torch.DoubleTensor for other types
         {name="src", type="torch.Tensor", method={opt=true}},
         {name="dim", type="number"},
         {name="flag", type="boolean", default=false},
         self="dst"
      },
      function(dst, src, dim, flag)
         local res = src and dst or torch.Tensor()
         src = src or dst

         local size = src:size()
         size[dim] = 1
         res:resize(size)
         local sumptr = ffi.new('real[1]')
         local sum2ptr = ffi.new('real[1]')
         if flag then
            torch.dimapply2(src, res, dim,
                            function(srcsz, src, srcst,
                                     ressz, res, resst)
                               C.th_sum_sum2_real(srcsz, src, srcst, sumptr, sum2ptr)
                               res[0] = (sum2ptr[0] - sumptr[0]*sumptr[0]/srcsz)/srcsz
                            end)
         else
            torch.dimapply2(src, res, dim,
                            function(srcsz, src, srcst,
                                     ressz, res, resst)
                               C.th_sum_sum2_real(srcsz, src, srcst, sumptr, sum2ptr)
                               res[0] = (sum2ptr[0] - sumptr[0]*sumptr[0]/srcsz)/(srcsz-1)
                            end)
         end
         return res
      end
   }
)

register(
   "add",
   {
      {
         {name="dst", type="torch.Tensor", opt=true,  method={opt=false}},
         {name="src", type="torch.Tensor", method={defaulta="self"}},
         {name="value", type="number"},
         self="dst"
      },
      function(dst, src, value)
         local res = src and dst or torch.Tensor()
         src = src or self

         res:resizeAs(x)
         torch.apply2(src, res,
                      function(sz, src, strsrc, res, strres)
                         C.th_add_real(sz, value, src, strsrc, res, strres)
                      end)
         return res
      end,

      {{name="dst", type="torch.Tensor", opt=true,   method={opt=false}},
       {name="src1", type="torch.Tensor", opt=false, method={defaulta="self"}},
       {name="value", type="number", default=1},
       {name="src2", type="torch.Tensor"},
       self="dst"},
      function(dst, src1, value, src2)
         local res = src1 and dst or torch.Tensor()
         src1 = src1 or dst

         res:resizeAs(src1)
         torch.apply3(src1, src2, res,
                      function(sz, src1, src1st, src2, src2st, res, resst)
                         C.th_cadd_real(sz, src1, src1st, value, src2, src2st, res, resst)
                      end)
         return res
      end}
)

register(
   "mul",
   {
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="src", type="torch.Tensor", method={defaulta="self"}},
         {name="value", type="number"},
         self="dst"
      },
      function(dst, src, value)
         local res = src and dst or torch.Tensor()
         src = src or dst

         res:resizeAs(src)
         torch.apply2(src, res,
                      function(sz, x, strx, y, stry)
                         C.th_mul_real(sz, value, x, strx, y, stry)
                      end)
         return res
      end
   }
)

register(
   "cmul",
   {
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="src1", type="torch.Tensor", method={defaulta="self"}},
         {name="src2", type="torch.Tensor"},
         self="dst"
      },
      function(dst, src1, src2)
         local res = src1 and dst or torch.Tensor()
         src1 = src1 or dst

         res:resizeAs(src1)
         torch.apply3(src1, src2, res, C.th_cmul_real)
         return res
      end
   }
)


register(
   "div",
   {
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="src", type="torch.Tensor", method={defaulta="self"}},
         {name="value", type="number"},
         self="dst"
      },
      function(dst, src, value)
         local res = src and dst or torch.Tensor()
         src = src or dst

         res:resizeAs(src)
         torch.apply2(src, res,
                      function(sz, src, strsrc, res, strres)
                         C.th_div_real(sz, value, src, strsrc, res, strres)
                      end)
         return res
      end
   }
)

register(
   "cdiv",
   {
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="src1", type="torch.Tensor", method={defaulta="self"}},
         {name="src2", type="torch.Tensor"},
         self="dst"
      },
      function(dst, src1, src2)
         local res = src1 and dst or torch.Tensor()
         src1 = src1 or dst

         res:resizeAs(src1)
         torch.apply3(src1, src2, res, C.th_cdiv_real)
         return res
      end
   }
)

register(
   "addcmul",
   {
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="value", type="number", default=1},
         {name="src1", type="torch.Tensor", method={defaulta="self"}},
         {name="src2", type="torch.Tensor"},
         self="dst"
      },
      function(dst, value, src1, src2)
         local res = src1 and dst or torch.Tensor()
         src1 = src1 or dst

         res:resizeAs(src1)
         torch.apply3(src1, src2, res,
                      function(sz, src1, src1st, src2, src2st, res, resst)
                         C.th_addcmul_real(sz, value, src1, src1st, src2, src2st, res, resst)
                      end)
         return res
      end
   }
)

register(
   "addcdiv",
   {
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="value", type="number", default=1},
         {name="src1", type="torch.Tensor", method={defaulta="self"}},
         {name="src2", type="torch.Tensor"},
         self="dst"
      },
      function(dst, value, src1, src2)
         local res = src1 and dst or torch.Tensor()
         src1 = src1 or dst

         res:resizeAs(src1)
         torch.apply3(src1, src2, res,
                      function(sz, src1, src1st, src2, src2st, res, resst)
                         C.th_addcdiv_real(sz, value, src1, src1st, src2, src2st, res, resst)
                      end)
         return res
      end
   }
)

register(
   "trace",
   {
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="src", type="torch.Tensor", method={opt=true}},
         self="dst"
      },
      function(dst, src)
         local res = src and dst or torch.Tensor()
         src = src or dst

         assert(src.__nDimension == 2, 'matrix expected')

         res:resize(math.min(src:size(1), src:size(2)))
         return tonumber(C.th_sum_real(math.min(src:size(1), src:size(2)),
                                       src:data(),
                                       src:stride(1)+src:stride(2)
                                 )
                      )
      end
   }
)

for _,name in ipairs{'log', 'log1p', 'exp', 'cos', 'acos', 'cosh', 'sin', 'asin',
                     'sinh', 'tan', 'atan', 'tanh', 'sqrt', 'ceil', 'floor', 'abs'} do

   local func = C['th_' .. name .. '_real']
   register(
      name,
      {
         {
            {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
            {name="src", type="torch.Tensor", method={defaulta="self"}},
            self="dst"
         },
         function(dst, src)
            local res = src and dst or torch.Tensor()
            src = src or dst

            res:resizeAs(src)
            torch.apply2(src, res, func)
            return res
         end
      }
   )
end

register(
   "pow",
   {
      {
         {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
         {name="src", type="torch.Tensor", method={defaulta="self"}},
         {name="value", type="number"},
         self="dst"
      },
      function(dst, src, value)
         local res = src and dst or torch.Tensor()
         src = src or dst

         res:resizeAs(src)
         torch.apply2(src, res,
                      function(sz, src, srcst, res, resst)
                         C.th_pow_real(sz, value, src, srcst, res, resst)
                      end)
         return res
      end
   }
)

register(
   "zeros",
   {
      {
         {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
         {name="size", type="numbers"},
         self="dst"
      },
      function(dst, size)
         local res = dst or torch.Tensor()
         res:resize(size)
         res:zero()
         return res
      end
   },
   namedispatch
)

register(
   "ones",
   {
      {
         {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
         {name="size", type="numbers"},
         self="dst"
      },
      function(dst, size)
         local res = dst or torch.Tensor()
         res:resize(size)
         res:fill(1)
         return res
      end
   },
   namedispatch
)

-- arg... a copy, or a view?
register(
   "diag",
   {
      {
         {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
         {name="src", type='torch.Tensor', method={opt=true}},
         {name="k", type='number', default=0},
         self="dst"
      },
      function(dst, src, k)
         local res = src and dst or torch.Tensor()
         src = src or dst

         assert(src.__nDimension == 1 or src.__nDimension == 2, "matrix or vector expected")
         if src.__nDimension == 1 then
            local sz = src.__size[0] + (k >= 0 and k or -k)
            res:resize(sz, sz):zero()
            C.th_copy_real(sz,
                           src:data(),
                           src.__stride[0],
                           res:data() + (k >= 0 and k*res.__stride[1] or -k*res.__stride[0]),
                           res.__stride[0]+res.__stride[1])
         else
            local sz
            if k >= 0 then
               sz = math.min(tonumber(src.__size[0]), tonumber(src.__size[1])-k)
            else
               sz = math.min(tonumber(src.__size[0])+k, tonumber(src.__size[1]))
            end
            res:resize(sz)
            C.th_copy_real(sz,
                           src:data() + (k >= 0 and k*src.__stride[1] or -k*src.__stride[0]),
                           src.__stride[0]+src.__stride[1],
                           res:data(),
                           res.__stride[0])
         end
         return res
      end
   }
)

register(
   "addmv",
   {
      {
         {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
         {name="beta", type='number', default=1},
         {name="src", type='torch.Tensor', method={defaulta="self"}},
         {name="alpha", type='number', default=1},
         {name="mat", type='torch.Tensor', dim=2},
         {name="vec", type='torch.Tensor', dim=1},
         self="dst"
      },
      function(dst, beta, src, alpha, mat, vec)
         local res = src and dst or torch.Tensor()
         src = src or self

         assert(mat.__size[1] == vec.__size[0], "size mismatch")
         assert(src.__nDimension == 1 and src.__size[0] == mat.__size[0], "size mismatch")

         if res ~= src then
            res:resizeAs(src)
            res:copy(src)
         end

         if mat.__stride[0] == 1 then
            C.th_gemv_real(string.byte('n'),
                           mat.__size[0], mat.__size[1],
                           alpha,
                           mat:data(), mat.__stride[1],
                           vec:data(), vec.__stride[0],
                           beta,
                           res:data(), res.__stride[0]);
         elseif mat.__stride[1] == 1 then
            C.th_gemv_real(string.byte('t'),
                           mat.__size[1], mat.__size[0],
                           alpha,
                           mat:data(), mat.__stride[0],
                           vec:data(), vec.__stride[0],
                           beta,
                           res:data(), res.__stride[0]);
         else
            mat = mat:contiguous()
            C.th_gemv_real(string.byte('t'),
                           mat.__size[1], mat.__size[0],
                           alpha,
                           mat:data(), mat.__stride[0],
                           vec:data(), vec.__stride[0],
                           beta,
                           res:data(), res.__stride[0]);
         end

         return res
      end
   }
)

register(
   "rand",
   {
      {
         {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
         {name="size", type="numbers"},
         self="dst"
      },
      function(dst, size)
         local res = dst or torch.Tensor()
         res:resize(size)
         torch.apply(res,
                     function(sz, dst, str)
                        for i=0,sz-1 do
                           dst[i*str] = torch.random()/2^32
                        end
                     end)
         return res
      end
   },
   namedispatch
)

register(
   "randn",
   {
      {
         {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
         {name="size", type="numbers"},
         self="dst"
      },
      function(dst, size)
         local res = dst or torch.Tensor()
         res:resize(size)
         torch.apply(res,
                     function(sz, dst, str)
                        for i=0,sz-1 do
                           dst[i*str] = torch.normal()
                        end
                     end)
         return res
      end
   },
   namedispatch
)

register(
   "copy",
   {
      {
         {name="dst", type='torch.Tensor'},
         {name="src", type='torch.Tensor'},
         self="dst"
      },
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_real)
      end,

      {
         {name="dst", type='torch.Tensor'},
         {name="src", type='torch.ByteTensor'},
         self="dst"
      },
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_byte)
      end,

      {
         {name="dst", type='torch.Tensor'},
         {name="src", type='torch.CharTensor'},
         self="dst"
      },
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_char)
      end,

      {
         {name="dst", type='torch.Tensor'},
         {name="src", type='torch.ShortTensor'},
         self="dst"
      },
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_short)
      end,

      {
         {name="dst", type='torch.Tensor'},
         {name="src", type='torch.IntTensor'},
         self="dst"
      },
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_int)
      end,

      {
         {name="dst", type='torch.Tensor'},
         {name="src", type='torch.LongTensor'},
         self="dst"
      },
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_long)
      end,

      {
         {name="dst", type='torch.Tensor'},
         {name="src", type='torch.FloatTensor'},
         self="dst"
      },
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_float)
      end,

      {
         {name="dst", type='torch.Tensor'},
         {name="src", type='torch.DoubleTensor'},
         self="dst"
      },
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_double)
      end
   }
)
