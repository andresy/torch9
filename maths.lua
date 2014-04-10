local ffi = require 'ffi'
local argcheck = require 'argcheck'
local torch = require 'torch.env'
local class = require 'class'
local C = require 'torch.clib'
local register_ = require 'torch.register'

local function register(args)
   return register_(args, torch, class.metatable('torch.Tensor'))
end

-- numbers: we could emulate it (right now it is done by hand)

-- DEBUG: Real would save this one
local function defaulttensortype()
   return class.type(torch.Tensor)
end

register{
   name = "fill",
   {name="dst", type="torch.Tensor"},
   {name="value", type="number"},
   call =
      function(dst, value)
         torch.apply(dst,
                     function(sz, dst, inc)
                        C.th_fill_real(sz, value, dst, inc)
                     end)
         return dst
      end
}

register{
   name = "zero",
   {name="dst", type="torch.Tensor"},
   call =
      function(dst)
         torch.apply(dst, C.th_zero_real)
         return dst
      end
}

register{
   name = "dot",
   {name="src1", type="torch.Tensor"},
   {name="src2", type="torch.Tensor"},
   call =
      function(src1, src2)
         local sum = 0
         torch.apply2(src1, src2,
                      function(sz, src1, incsrc1, src2, incsrc2)
                         sum = sum + C.th_dot_real(sz, src1, incsrc1, src2, incsrc2)
                      end)
         return sum
      end
}

register{
   name = "min",
   {name="src", type="torch.Tensor"},
   call =
      function(src)
         local min = math.huge
         local minptr = ffi.new('real[1]')
         local idsrcptr = ffi.new('long[1]')
         torch.apply(src,
                     function(sz, src, inc)
                        C.th_min_real(sz, src, inc, minptr, idsrcptr)
                        min = math.min(min, minptr[0])
                     end)
         return min
      end
}

register{
   name = "min",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="idx", type="torch.LongTensor", opt=true},
   {name="src", type="torch.Tensor", method={opt=true}},
   {name="dim", type="number"},
   call =
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

register{
   name = "max",
   {name="src", type="torch.Tensor"},
   call =
      function(src)
         local max = -math.huge
         local maxptr = ffi.new('real[1]')
         local idsrcptr = ffi.new('long[1]')
         torch.apply(src,
                     function(sz, src, inc)
                        C.th_max_real(sz, src, inc, maxptr, idsrcptr)
                        max = math.max(max, maxptr[0])
                     end)
         return max
      end
}

register{
   name = "max",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="idx", type="torch.LongTensor", opt=true},
   {name="src", type="torch.Tensor", method={opt=true}},
   {name="dim", type="number"},
   call =
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

register{
   name = "sum",
   {name="src", type="torch.Tensor"},
   call =
      function(src)
         local sum = 0
         torch.apply(src,
                     function(sz, src, inc)
                        sum = sum + C.th_sum_real(sz, src, inc)
                     end)
         return sum
      end
}

register{
   name = "sum",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="src", type="torch.Tensor", method={opt=true}},
   {name="dim", type="number"},
   call =
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

register{
   name = "prod",
   {name="src", type="torch.Tensor"},
   call =
      function(src)
         local prod = (src:nElement() > 0) and 1 or 0
         torch.apply(src,
                     function(sz, src, inc)
                        prod = prod * tonumber( C.th_prod_real(sz, src, inc) )
                     end)
         return prod
      end
}

register{
   name = "prod",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="src", type="torch.Tensor", method={opt=true}},
   {name="dim", type="number"},
   call =
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

register{
   name = "cumsum",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="src", type="torch.Tensor", method={opt=true}},
   {name="dim", type="number"},
   call =
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

register{
   name = "cumprod",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="src", type="torch.Tensor", method={opt=true}},
   {name="dim", type="number"},
   call =
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

-- float only

register{
   name = "norm",
   {name="src", type="torch.Tensor"},
   {name="n", type="number", default=2},
   call =
      function(src, n)
         local norm = 0
         torch.apply(src,
                     function(sz, src, inc)
                        norm = norm + C.th_norm_real(sz, n, 0, src, inc)
                     end)
         return math.pow(norm, 1/n)
      end
}

register{
   name = "norm",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="src", type="torch.Tensor", method={opt=true}},
   {name="n", type="number", default=2},
   {name="dim", type="number"},
   call =
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

register{
   name = "mean",
   {name="src", type="torch.Tensor"},
   call =
      function(src)
         local sum = 0
         torch.apply(src, function(sz, src, inc)
                             sum = sum + C.th_sum_real(sz, src, inc)
                          end)
         return sum / src:nElement()
      end
}

register{
   name = "mean",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},  -- could be torch.DoubleTensor for other types
   {name="src", type="torch.Tensor", method={opt=true}},
   {name="dim", type="number"},
   call =
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

register{
   name = "std",
   {name="src", type="torch.Tensor"},
   {name="flag", type="boolean", default=false},
   call =
      function(src, flag)
         local sum = 0
         local sum2 = 0
         local n = src:nElement()
         torch.apply(src,
                     function(sz, src, inc)
                        sum = sum + C.th_sum_real(sz, src, inc)
                        sum2 = sum2 + C.th_sum2_real(sz, src, inc)
                     end)
         if flag then
            return math.sqrt((sum2 - sum*sum/n)/n)
         else
            return math.sqrt((sum2 - sum*sum/n)/(n-1))
         end
      end
}
      
register{
   name = "std",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},  -- could be torch.DoubleTensor for other types
   {name="src", type="torch.Tensor", method={opt=true}},
   {name="dim", type="number"},
   {name="flag", type="boolean", default=false},
   call =
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

register{
   name = "var",
   {name="src", type="torch.Tensor"},
   {name="flag", type="boolean", default=false},
   call =
      function(src, flag)
         local sum = 0
         local sum2 = 0
         local n = src:nElement()
         torch.apply(src, function(sz, src, inc)
                             sum = sum + C.th_sum_real(sz, src, inc)
                             sum2 = sum2 + C.th_sum2_real(sz, src, inc)
                          end)
         if flag then
            return (sum2 - sum*sum/n)/n
         else
            return (sum2 - sum*sum/n)/(n-1)
         end
      end
}

register{
   name = "var",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},  -- could be torch.DoubleTensor for other types
   {name="src", type="torch.Tensor", method={opt=true}},
   {name="dim", type="number"},
   {name="flag", type="boolean", default=false},
   call =
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

register{
   name = "add",
   {name="dst", type="torch.Tensor", opt=true,  method={opt=false}},
   {name="src", type="torch.Tensor", method={defaulta="self"}},
   {name="value", type="number"},
   call =
      function(dst, src, value)
         local res = src and dst or torch.Tensor()
         src = src or self

         res:resizeAs(x)
         torch.apply2(src, res,
                      function(sz, src, incsrc, res, incres)
                         C.th_add_real(sz, value, src, incsrc, res, incres)
                      end)
         return res
      end
}

register{
   name = "add",
   {name="dst", type="torch.Tensor", opt=true,   method={opt=false}},
   {name="src1", type="torch.Tensor", opt=false, method={defaulta="self"}},
   {name="value", type="number", default=1},
   {name="src2", type="torch.Tensor"},
   call =
      function(dst, src1, value, src2)
         local res = src1 and dst or torch.Tensor()
         src1 = src1 or dst

         res:resizeAs(src1)
         torch.apply3(src1, src2, res,
                      function(sz, src1, src1st, src2, src2st, res, resst)
                         C.th_cadd_real(sz, src1, src1st, value, src2, src2st, res, resst)
                      end)
         return res
      end
}

register{
   name = "mul",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="src", type="torch.Tensor", method={defaulta="self"}},
   {name="value", type="number"},
   call =
      function(dst, src, value)
         local res = src and dst or torch.Tensor()
         src = src or dst

         res:resizeAs(src)
         torch.apply2(src, res,
                      function(sz, src, incsrc, res, incres)
                         C.th_mul_real(sz, value, src, incsrc, res, incres)
                      end)
         return res
      end
}

register{
   name = "cmul",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="src1", type="torch.Tensor", method={defaulta="self"}},
   {name="src2", type="torch.Tensor"},
   call =
      function(dst, src1, src2)
         local res = src1 and dst or torch.Tensor()
         src1 = src1 or dst

         res:resizeAs(src1)
         torch.apply3(src1, src2, res, C.th_cmul_real)
         return res
      end
}


register{
   name = "div",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="src", type="torch.Tensor", method={defaulta="self"}},
   {name="value", type="number"},
   call =
      function(dst, src, value)
         local res = src and dst or torch.Tensor()
         src = src or dst

         res:resizeAs(src)
         torch.apply2(src, res,
                      function(sz, src, incsrc, res, incres)
                         C.th_div_real(sz, value, src, incsrc, res, incres)
                      end)
         return res
      end
}

register{
   name = "cdiv",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="src1", type="torch.Tensor", method={defaulta="self"}},
   {name="src2", type="torch.Tensor"},
   call =
      function(dst, src1, src2)
         local res = src1 and dst or torch.Tensor()
         src1 = src1 or dst

         res:resizeAs(src1)
         torch.apply3(src1, src2, res, C.th_cdiv_real)
         return res
      end
}

register{
   name = "addcmul",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="value", type="number", default=1},
   {name="src1", type="torch.Tensor", method={defaulta="self"}},
   {name="src2", type="torch.Tensor"},
   call =
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

register{
   name = "addcdiv",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="value", type="number", default=1},
   {name="src1", type="torch.Tensor", method={defaulta="self"}},
   {name="src2", type="torch.Tensor"},
   call =
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

register{
   name = "trace",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="src", type="torch.Tensor", method={opt=true}},
   call =
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

for _,name in ipairs{'log', 'log1p', 'exp', 'cos', 'acos', 'cosh', 'sin', 'asin',
                     'sinh', 'tan', 'atan', 'tanh', 'sqrt', 'ceil', 'floor', 'abs'} do

   local func = C['th_' .. name .. '_real']
   register{
      name = name,
      {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
      {name="src", type="torch.Tensor", method={defaulta="self"}},
      call =
         function(dst, src)
            local res = src and dst or torch.Tensor()
            src = src or dst

            res:resizeAs(src)
            torch.apply2(src, res, func)
            return res
         end
   }
end

register{
   name = "pow",
   {name="dst", type="torch.Tensor", opt=true, method={opt=false}},
   {name="src", type="torch.Tensor", method={defaulta="self"}},
   {name="value", type="number"},
   call =
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

local function zeros(dst, size, typename)
   local res = dst or class.metatable(typename).new()
   res:resize(size)
   res:zero()
   return res
end

register{
   name = "zeros",
   {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
   {name="size", type="table"},
   {name="typename", type="string", defaultf=defaulttensortype}, -- namedispatch
   call = zeros
}

register{
   name = "zeros",
   {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
   {name="dim1", type="number"},
   {name="dim2", type="number", opt=true},
   {name="dim3", type="number", opt=true},
   {name="dim4", type="number", opt=true},
   {name="typename", type="string", defaultf=defaulttensortype}, -- namedispatch
   call =
      function(dst, dim1, dim2, dim3, dim4, typename)
         return zeros(dst, {dim1, dim2, dim3, dim4}, typename)
      end
}

local function ones(dst, size, typename)
   local res = dst or class.metatable(typename).new()
   res:resize(size)
   res:fill(1)
   return res
end

register{
   name = "ones",
   {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
   {name="size", type="table"},
   {name="typename", type="string",  defaultf=defaulttensortype}, -- namedispatch
   call = ones
}

register{
   name = "ones",
   {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
   {name="dim1", type="number"},
   {name="dim2", type="number", opt=true},
   {name="dim3", type="number", opt=true},
   {name="dim4", type="number", opt=true},
   {name="typename", type="string", default=defaulttensortype}, -- namedispatch
   call =
      function(dst, dim1, dim2, dim3, dim4, typename)
         return ones(dst, {dim1, dim2, dim3, dim4}, typename)
      end
}

-- arg... a copy, or a view?
register{
   name = "diag",
   {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
   {name="src", type='torch.Tensor', method={opt=true}},
   {name="k", type='number', default=0},
   call =
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

register{
   name = "addmv",
   {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
   {name="beta", type='number', default=1},
   {name="src", type='torch.Tensor', method={defaulta="self"}},
   {name="alpha", type='number', default=1},
   {name="mat", type='torch.Tensor', dim=2},
   {name="vec", type='torch.Tensor', dim=1},
   call =
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

local function rand(dst, size, typename)
   local res = dst or class.metatable(typename).new()
   res:resize(size)
   torch.apply(res,
               function(sz, dst, inc)
                  for i=0,sz-1 do
                     dst[i*inc] = torch.random()/2^32
                  end
               end)
   return res
end

register{
   name = "rand",
   {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
   {name="size", type="table"},
   {name="typename", type="string", defaultf=defaultensortype}, -- namedispatch
   call = rand
}

register{
   name = "rand",
   {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
   {name="dim1", type="number"},
   {name="dim2", type="number", opt=true},
   {name="dim3", type="number", opt=true},
   {name="dim4", type="number", opt=true},
   {name="typename", type="string", default=defaulttensortype}, -- namedispatch
   call =
      function(dst, dim1, dim2, dim3, dim4, typename)
         return rand(dst, {dim1, dim2, dim3, dim4}, typename)
      end
}

local function randn(dst, size, typename)
   local res = dst or class.metatable(typename).new()
   res:resize(size)
   torch.apply(res,
               function(sz, dst, inc)
                  for i=0,sz-1 do
                     dst[i*inc] = torch.normal()
                  end
               end)
   return res
end

register{
   name = "randn",
   {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
   {name="size", type="table"},
   {name="typename", type="string", defaultf=defaultensortype}, -- namedispatch
   call = randn
}

register{
   name = "randn",
   {name="dst", type='torch.Tensor', opt=true, method={opt=false}},
   {name="dim1", type="number"},
   {name="dim2", type="number", opt=true},
   {name="dim3", type="number", opt=true},
   {name="dim4", type="number", opt=true},
   {name="typename", type="string", default=defaulttensortype}, -- namedispatch
   call =
      function(dst, dim1, dim2, dim3, dim4, typename)
         return randn(dst, {dim1, dim2, dim3, dim4}, typename)
      end
}

register{
   name = "copy",
   {name="dst", type='torch.Tensor'},
   {name="src", type='torch.Tensor'},
   call =
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_real)
      end
}

register{
   name = "copy",
   {name="dst", type='torch.Tensor'},
   {name="src", type='torch.ByteTensor'},
   call =
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_byte)
      end
}

register{
   name = "copy",
   {name="dst", type='torch.Tensor'},
   {name="src", type='torch.CharTensor'},
   call =
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_char)
      end
}

register{
   name = "copy",
   {name="dst", type='torch.Tensor'},
   {name="src", type='torch.ShortTensor'},
   call =
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_short)
      end
}

register{
   name = "copy",
   {name="dst", type='torch.Tensor'},
   {name="src", type='torch.IntTensor'},
   call =
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_int)
      end
}

register{
   name = "copy",
   {name="dst", type='torch.Tensor'},
   {name="src", type='torch.LongTensor'},
   call =
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_long)
      end
}

register{
   name = "copy",
   {name="dst", type='torch.Tensor'},
   {name="src", type='torch.FloatTensor'},
   call =
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_float)
      end
}

register{
   name = "copy",
   {name="dst", type='torch.Tensor'},
   {name="src", type='torch.DoubleTensor'},
   call =
      function(dst, src)
         torch.apply2(src, dst,
                      C.th_copy_real_double)
      end
}
