local ffi = require 'ffi'
local namedispatch = require 'torch.namedispatch'
local dispatch = require 'torch.dispatch'
local torch = require 'torch'

-- DEBUG:
-- if returning stuff like th.sum_float directly, then one should do tonumber(), i think

ffi.cdef[[
      void zero_real(real *x, long str, long sz);
      void fill_real(real *x, long str, long sz, real value);
      void copy_real(real *y, long stry, real *x, long strx, long sz);
      real dot_real(real *x, long strx, real *y, long stry, long sz);
      void min_real(real *min_, long *idx_, real *x, long strx, long sz);
      void max_real(real *max_, long *idx_, real *x, long strx, long sz);
      real sum_real(real *x, long strx, long sz);
      void prod_real(real *prod_, real *x, long strx, long sz);
      real norm_real(real *x, long strx, long sz, real n, int dopow);
      void cumsum_real(real *cumsum, long cumsumst, long cumsumsz, real *x, long strx, long sz);
      void cumprod_real(real *cumprod, long cumprodst, long cumprodsz, real *x, long strx, long sz);
      real sum2_real(real *x, long strx, long sz);
      void sum_sum2_real(real *sum_, real *sum2_, real *x, long strx, long sz);
      void add_real(real *y, long stry, real *x, long strx, long sz, real value);
      void cadd_real(real *z, long strz, real *y, long stry, real *x, long strx, long sz, real value);
      void mul_real(real *y, long stry, real *x, long strx, long sz, real value);
      void cmul_real(real *z, long strz, real *y, long stry, real *x, long strx, long sz);
      void div_real(real *y, long stry, real *x, long strx, long sz, real value);
      void cdiv_real(real *z, long strz, real *y, long stry, real *x, long strx, long sz);
      void addcmul_real(real *z, long strz, real *y, long stry, real *x, long strx, long sz, real value);
      void addcdiv_real(real *z, long strz, real *y, long stry, real *x, long strx, long sz, real value);
]]

local th = ffi.load(paths.concat(paths.install_lua_path,
                                 'torch',
                                 ((jit.os == 'Windows') and '' or 'lib') .. 'maths' .. 
                                 ((jit.os == 'Windows') and '.dll' or ((jit.os == 'OSX') and '.dylib' or '.so'))))
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
                     function(dst, str, sz)
                        th.fill_real(dst, str, sz, value)
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
         torch.apply(dst, th.zero_real)
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
                      function(src1, strsrc1, src2, strsrc2, sz)
                         sum = sum + th.dot_real(src1, strsrc1, src2, strsrc2, sz)
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
                     function(src, str, sz)
                        th.min_real(minptr, idsrcptr, src, str, sz)
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
         torch.dimapply3(res, idx, src, dim,
                         function(res, resst, ressz,
                                  idx, idxst, idxsz,
                                  src, srcst, srcsz)
                            th.min_real(res, idx, src, srcst, srcsz)
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
                     function(src, str, sz)
                        th.max_real(maxptr, idsrcptr, src, str, sz)
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
         torch.dimapply3(res, idx, src, dim,
                         function(res, resst, ressz,
                                  idx, idxst, idxsz,
                                  src, srcst, srcsz)
                            th.max_real(res, idx, src, srcst, srcsz)
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
         torch.apply(src, function(src, str, sz)
                             sum = sum + th.sum_real(src, str, sz)
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
         torch.dimapply2(res, src, dim,
                         function(res, resst, ressz,
                                  src, srcst, srcsz)
                            res[0] = th.sum_real(src, srcst, srcsz)
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
         local prodptr = ffi.new('real[1]')
         torch.apply(src,
                     function(src, str, sz)
                        th.prod_real(prodptr, src, str, sz)
                        prod = prod * prodptr[0]
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
         torch.dimapply2(res, src, dim,
                         function(res, resst, ressz,
                                  src, srcst, srcsz)
                            th.prod_real(res, src, srcst, srcsz)
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
         torch.dimapply2(res, src, dim, th.cumsum_real)
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
         torch.dimapply2(res, src, dim, th.cumprod_real)
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
                     function(src, str, sz)
                        norm = norm + th.norm_real(src, str, sz, n, 0)
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
         torch.dimapply2(res, src, dim,
                         function(res, resst, ressz,
                                  src, srcst, srcsz)
                            th.norm_real(res, src, srcst, srcsz, n, 1)
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
         torch.apply(src, function(src, str, sz)
                             sum = sum + th.sum_real(src, str, sz)
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
         torch.dimapply2(res, src, dim,
                         function(res, resst, ressz,
                                  src, srcst, srcsz)
                            res[0] = th.sum_real(src, srcst, srcsz) / srcsz
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
                     function(src, str, sz)
                        sum = sum + th.sum_real(src, str, sz)
                        sum2 = sum2 + th.sum2_real(src, str, sz)
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
            torch.dimapply2(res, src, dim,
                            function(res, resst, ressz,
                                     src, srcst, srcsz)
                               th.sum_sum2_real(sumptr, sum2ptr, src, srcst, srcsz)
                               res[0] = math.sqrt((sum2ptr[0] - sumptr[0]*sumptr[0]/srcsz)/srcsz)
                            end)
         else
            torch.dimapply2(res, src, dim,
                            function(res, resst, ressz,
                                     src, srcst, srcsz)
                               th.sum_sum2_real(sumptr, sum2ptr, src, srcst, srcsz)
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
         torch.apply(src, function(src, str, sz)
                             sum = sum + th.sum_real(src, str, sz)
                             sum2 = sum2 + th.sum2_real(src, str, sz)
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
            torch.dimapply2(res, src, dim,
                            function(res, resst, ressz,
                                     src, srcst, srcsz)
                               th.sum_sum2_real(sumptr, sum2ptr, src, srcst, srcsz)
                               res[0] = (sum2ptr[0] - sumptr[0]*sumptr[0]/srcsz)/srcsz
                            end)
         else
            torch.dimapply2(res, src, dim,
                            function(res, resst, ressz,
                                     src, srcst, srcsz)
                               th.sum_sum2_real(sumptr, sum2ptr, src, srcst, srcsz)
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
         {name="src", type="torch.Tensor", opt=false, method={defaulta="self"}},
         {name="value", type="number"},
         self="dst"
      },
      function(dst, src, value)
         local res = src and dst or torch.Tensor()
         src = src or self

         res:resizeAs(x)
         torch.apply2(res, src,
                      function(res, strres, src, strsrc, sz)
                         th.add_real(res, strres, src, strsrc, sz, value)
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
         torch.apply3(res, src1, src2,
                      function(res, resst, src1, src1st, src2, src2st, sz)
                         th.cadd_real(res, resst, src1, src1st, src2, src2st, sz, value)
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
         torch.apply2(res, src,
                      function(y, stry, x, strx, sz)
                         th.mul_real(y, stry, x, strx, sz, value)
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
         torch.apply3(res, src1, src2, th.cmul_real)
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
         torch.apply2(res, src,
                      function(res, strres, src, strsrc, sz)
                         th.div_real(res, strres, src, strsrc, sz, value)
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
         torch.apply3(res, src1, src2, th.cdiv_real)
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
         torch.apply3(res, src1, src2,
                      function(res, resst, src1, src1st, src2, src2st, sz)
                         th.addcmul_real(res, resst, src1, src1st, src2, src2st, sz, value)
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
         torch.apply3(res, src1, src2,
                      function(res, resst, src1, src1st, src2, src2st, sz)
                         th.addcdiv_real(res, resst, src1, src1st, src2, src2st, sz, value)
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
         return th.sum_real(src:data(),
                            src:stride(1)+src:stride(2),
                            math.min(src:size(1), src:size(2)))
      end
   }
)

for _,name in ipairs{'log', 'log1p', 'exp', 'cos', 'acos', 'cosh', 'sin', 'asin',
                     'sinh', 'tan', 'atan', 'tanh', 'sqrt', 'ceil', 'floor', 'abs'} do

   ffi.cdef(string.format('void %s_float(float *y, long stry, float *x, long strx, long sz);', name)) -- DEBUG: see below

   local func = th[name .. '_float'] -- DEBUG: *MUST* be real here, but i did not defined them yet ;)
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
            torch.apply2(res, src, func)
            return res
         end
      }
   )
end

ffi.cdef('void pow_float(float *y, long stry, float *x, long strx, long sz, float value);')

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
         torch.apply2(res, src,
                      function(res, resst, src, srcst, sz)
                         th.pow_real(res, resst, src, srcst, sz, value)
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
            th.copy_real(res:data() + (k >= 0 and k*res.__stride[1] or -k*res.__stride[0]),
                         res.__stride[0]+res.__stride[1],
                         src:data(),
                         src.__stride[0],
                         sz)
         else
            local sz
            if k >= 0 then
               sz = math.min(tonumber(src.__size[0]), tonumber(src.__size[1])-k)
            else
               sz = math.min(tonumber(src.__size[0])+k, tonumber(src.__size[1]))
            end
            res:resize(sz)
            th.copy_real(res:data(),
                         res.__stride[0],
                         src:data() + (k >= 0 and k*src.__stride[1] or -k*src.__stride[0]),
                         src.__stride[0]+src.__stride[1],
                         sz)
         end
         return res
      end
   }
)
