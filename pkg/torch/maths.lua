local ffi = require 'ffi'

ffi.cdef[[
      void zero_float(float *x, long str, long sz);
      void fill_float(float *x, long str, long sz, float value);
      void copy_float(float *y, long stry, float *x, long strx, long sz);
      float dot_float(float *x, long strx, float *y, long stry, long sz);
      void min_float(float *min_, long *idx_, float *x, long strx, long sz);
      void max_float(float *max_, long *idx_, float *x, long strx, long sz);
      float sum_float(float *x, long strx, long sz);
      void prod_float(float *prod_, float *x, long strx, long sz);
      float norm_float(float *x, long strx, long sz, float n, int dopow);
      void cumsum_float(float *cumsum, long cumsumst, long cumsumsz, float *x, long strx, long sz);
      void cumprod_float(float *cumprod, long cumprodst, long cumprodsz, float *x, long strx, long sz);
      float sum2_float(float *x, long strx, long sz);
      void sum_sum2_float(float *sum_, float *sum2_, float *x, long strx, long sz);
      void add_float(float *y, long stry, float *x, long strx, long sz, float value);
      void cadd_float(float *z, long strz, float *y, long stry, float *x, long strx, long sz, float value);
      void mul_float(float *y, long stry, float *x, long strx, long sz, float value);
      void cmul_float(float *z, long strz, float *y, long stry, float *x, long strx, long sz);
      void div_float(float *y, long stry, float *x, long strx, long sz, float value);
      void cdiv_float(float *z, long strz, float *y, long stry, float *x, long strx, long sz);
      void addcmul_float(float *z, long strz, float *y, long stry, float *x, long strx, long sz, float value);
      void addcdiv_float(float *z, long strz, float *y, long stry, float *x, long strx, long sz, float value);
]]

local th = ffi.load(paths.concat(paths.install_lua_path,
                                 'torch',
                                 ((jit.os == 'Windows') and '' or 'lib') .. 'maths' .. 
                                 ((jit.os == 'Windows') and '.dll' or ((jit.os == 'OSX') and '.dylib' or '.so'))))
print('we loaded real')

local function register(name, ...)
   local func, method = argcheck(...)
   torch[name] = torch[name] or dispatch()
   dispatch(torch[name], "torch.Tensor", func)
   torch.Tensor[name] = method
end

register(
   "fill",
   {{name="dst", type="torch.Tensor"},
    {name="value", type="number"},
    self="dst"},
   function(dst, value)
      torch.apply(dst, function(x, str, sz)
                          th.fill_real(x, str, sz, value)
                       end)
      return dst
   end
)

register(
   "zero",
   {{name="dst", type="torch.Tensor"},
    self="dst"},
   function(dst)
      torch.apply(dst, th.zero_real)
      return dst
   end
)

register(
   "dot",
   {{name="vec1", type="torch.Tensor"},
    {name="vec2", type="torch.Tensor"},
    self="vec1"},
   function(vec1, vec2)
      local sum = 0
      torch.apply2(vec1, vec2, function(x, strx, y, stry, sz)
                                  sum = sum + th.dot_real(x, strx, y, stry, sz)
                               end)
      return sum
   end
)

register(
   "min",
   {{{name="src", type="torch.Tensor"},
     self="src"},
    function(src)
       local min = math.huge
       local minptr = ffi.new('real[1]')
      local idxptr = ffi.new('long[1]')
      torch.apply(src, function(x, str, sz)
                          th.min_real(minptr, idxptr, x, str, sz)
                          min = math.min(min, minptr[0])
                       end)
      return min
   end,
    
    {{name="dst", type="torch.Tensor", default=true},
     {name="idx", type="torch.LongTensor", default=true},
     {name="src", type="torch.Tensor"},
     {name="dim", type="number"},
     self="src"},
    function(dst, idx, src, dim)
       local size = src:size()
       size[dim] = 1
       dst:resize(size)
       idx:resize(size)
       torch.dimapply3(dst, idx, src, dim, function(dst, dstst, dstsz,
                                                    idx, idxst, idxsz,
                                                    src, srcst, srcsz)
                                              th.min_real(dst, idx, src, srcst, srcsz)
                                           end)
       return dst, idx
    end}
)

register(
   "max",
   {{{name="src", type="torch.Tensor"},
     self="src"},
    function(src)
       local max = -math.huge
       local maxptr = ffi.new('real[1]')
       local idxptr = ffi.new('long[1]')
       torch.apply(src, function(x, str, sz)
                           th.max_real(maxptr, idxptr, x, str, sz)
                           max = math.max(max, maxptr[0])
                        end)
       return max
    end,
    
    {{name="dst", type="torch.Tensor", default=true},
     {name="idx", type="torch.LongTensor", default=true},
     {name="src", type="torch.Tensor"},
     {name="dim", type="number"},
     self="src"},
    function(dst, idx, src, dim)
       local size = src:size()
       size[dim] = 1
       dst:resize(size)
       idx:resize(size)
       torch.dimapply3(dst, idx, src, dim, function(dst, dstst, dstsz,
                                                    idx, idxst, idxsz,
                                                    src, srcst, srcsz)
                                              th.max_real(dst, idx, src, srcst, srcsz)
                                           end)
       return dst, idx
    end}
)

register(
   "sum",
   {{{name="src", type="torch.Tensor"},
     self="src"},
    function(src)
       local sum = 0
       torch.apply(src, function(x, str, sz)
                           sum = sum + th.sum_real(x, str, sz)
                        end)
       return sum
    end,
    
    {{name="dst", type="torch.Tensor", default=true},
     {name="src", type="torch.Tensor"},
     {name="dim", type="number"},
     self="src"},
    function(dst, src, dim)
       local size = src:size()
       size[dim] = 1
       dst:resize(size)
       torch.dimapply2(dst, src, dim, function(dst, dstst, dstsz,
                                               src, srcst, srcsz)
                                         dst[0] = th.sum_real(src, srcst, srcsz)
                                      end)
       return dst
    end}
)

register(
   "prod",
   {{{name="src", type="torch.Tensor"},
     self="src"},
    function(src)
       local prod = (src:nElement() > 0) and 1 or 0
       local prodptr = ffi.new('real[1]')
       torch.apply(src, function(x, str, sz)
                           th.prod_real(prodptr, x, str, sz)
                           prod = prod * prodptr[0]
                        end)
       return prod
    end,
    
    {{name="dst", type="torch.Tensor", default=true},
     {name="src", type="torch.Tensor"},
     {name="dim", type="number"},
     self="src"},
    function(dst, src, dim)
       local size = src:size()
       size[dim] = 1
       dst:resize(size)
       torch.dimapply2(dst, src, dim, function(dst, dstst, dstsz,
                                               src, srcst, srcsz)
                                         th.prod_real(dst, src, srcst, srcsz)
                                      end)
       return dst
    end}
)

register(
   "cumsum",
   {{name="dst", type="torch.Tensor", default=true},
    {name="src", type="torch.Tensor"},
    {name="dim", type="number"},
    self="src"},
   function(dst, src, dim)
      dst:resizeAs(src)
      torch.dimapply2(dst, src, dim, th.cumsum_real)
      return dst
   end
)

register(
   "cumprod",
   {{name="dst", type="torch.Tensor", default=true},
    {name="src", type="torch.Tensor"},
    {name="dim", type="number"},
    self="src"},
   function(dst, src, dim)
      dst:resizeAs(src)
      torch.dimapply2(dst, src, dim, th.cumprod_real)
      return dst
   end
)

-- float only

register(
   "norm",
   {{{name="src", type="torch.Tensor"},
     {name="n", type="number", default=2},
     self="src"},
    function(src, n)
       local norm = 0
       torch.apply(src, function(x, str, sz)
                           norm = norm + th.norm_real(x, str, sz, n, 0)
                        end)
       return math.pow(norm, 1/n)
    end,
    
    {{name="dst", type="torch.Tensor", default=true},
     {name="src", type="torch.Tensor"},
     {name="n", type="number", default=2},
     {name="dim", type="number"},
     self="src"},
    function(dst, src, n, dim)
       local size = src:size()
       size[dim] = 1
       dst:resize(size)
       torch.dimapply2(dst, src, dim, function(dst, dstst, dstsz,
                                               src, srcst, srcsz)
                                         th.norm_real(dst, src, srcst, srcsz, n, 1)
                                      end)
       return dst
    end}
)

register(
   "mean",
   {{{name="src", type="torch.Tensor"},
     self="src"},
    function(src)
       local sum = 0
       torch.apply(src, function(x, str, sz)
                           sum = sum + th.sum_real(x, str, sz)
                        end)
       return sum / src:numElement()
    end,
    
    {{name="dst", type="torch.Tensor", default=true}, -- could be torch.DoubleTensor for other types
       {name="src", type="torch.Tensor"},
       {name="dim", type="number"}},
    function(dst, src, dim)
       local size = src:size()
       size[dim] = 1
       dst:resize(size)
       torch.dimapply2(dst, src, dim, function(dst, dstst, dstsz,
                                               src, srcst, srcsz)
                                         dst[0] = th.sum_real(src, srcst, srcsz) / srcsz
                                      end)
       return dst
    end}
)

register(
   "std",
   {{{name="src", type="torch.Tensor"},
     {name="flag", type="boolean", default=false},
     self="src"},
    function(src, flag)
       local sum = 0
       local sum2 = 0
       local n = src:nElement()
       torch.apply(src, function(x, str, sz)
                           sum = sum + th.sum_real(x, str, sz)
                           sum2 = sum2 + th.sum2_real(x, str, sz)
                        end)
       if flag then
          return math.sqrt((sum2 - sum*sum/n)/n)
       else
          return math.sqrt((sum2 - sum*sum/n)/(n-1))
       end
    end,
    
    {{name="dst", type="torch.Tensor", default=true}, -- could be torch.DoubleTensor for other types
       {name="src", type="torch.Tensor"},
       {name="dim", type="number"},
       {name="flag", type="boolean", default=false},
       self="src"},
    function(dst, src, dim, flag)
       local size = src:size()
       size[dim] = 1
       dst:resize(size)
       local sumptr = ffi.new('real[1]')
       local sum2ptr = ffi.new('real[1]')
       if flag then
          torch.dimapply2(dst, src, dim, function(dst, dstst, dstsz,
                                                  src, srcst, srcsz)
                                            th.sum_sum2_real(sumptr, sum2ptr, src, srcst, srcsz)
                                            dst[0] = math.sqrt((sum2ptr[0] - sumptr[0]*sumptr[0]/srcsz)/srcsz)
                                         end)
       else
          torch.dimapply2(dst, src, dim, function(dst, dstst, dstsz,
                                                  src, srcst, srcsz)
                                            th.sum_sum2_real(sumptr, sum2ptr, src, srcst, srcsz)
                                            dst[0] = math.sqrt((sum2ptr[0] - sumptr[0]*sumptr[0]/srcsz)/(srcsz-1))
                                         end)
       end
       return dst
    end}
)

register(
   "var",
   {{{name="src", type="torch.Tensor"},
     {name="flag", type="boolean", default=false},
     self="src"},
    function(src, flag)
       local sum = 0
       local sum2 = 0
       local n = src:nElement()
       torch.apply(src, function(x, str, sz)
                           sum = sum + th.sum_real(x, str, sz)
                           sum2 = sum2 + th.sum2_real(x, str, sz)
                        end)
       if flag then
          return (sum2 - sum*sum/n)/n
       else
          return (sum2 - sum*sum/n)/(n-1)
       end
            end,
    
    {{name="dst", type="torch.Tensor", default=true}, -- could be torch.DoubleTensor for other types
       {name="src", type="torch.Tensor"},
       {name="dim", type="number"},
       {name="flag", type="boolean", default=false},
       self="src"},
    function(dst, src, dim, flag)
       local size = src:size()
       size[dim] = 1
       dst:resize(size)
       local sumptr = ffi.new('real[1]')
       local sum2ptr = ffi.new('real[1]')
       if flag then
          torch.dimapply2(dst, src, dim, function(dst, dstst, dstsz,
                                                  src, srcst, srcsz)
                                            th.sum_sum2_real(sumptr, sum2ptr, src, srcst, srcsz)
                                            dst[0] = (sum2ptr[0] - sumptr[0]*sumptr[0]/srcsz)/srcsz
                                         end)
       else
          torch.dimapply2(dst, src, dim, function(dst, dstst, dstsz,
                                                  src, srcst, srcsz)
                                            th.sum_sum2_real(sumptr, sum2ptr, src, srcst, srcsz)
                                            dst[0] = (sum2ptr[0] - sumptr[0]*sumptr[0]/srcsz)/(srcsz-1)
                                         end)
       end
       return dst
    end}
)

-- warning: should have a different one for the method
torch.Tensor.add =
   argcheck{{{name="dst", type="torch.Tensor", default=true},
             {name="src", type="torch.Tensor"},
             {name="value", type="number"}},
             function(dst, src, value)
                dst:resizeAs(src)
                torch.apply2(dst, src, function(y, stry, x, strx, sz)
                                          th.add_real(y, stry, x, strx, sz, value)
                                       end)
                return dst
             end,

             {{name="dst", type="torch.Tensor", default=true},
              {name="src1", type="torch.Tensor"},
              {name="value", type="number", default=1},
              {name="src2", type="torch.Tensor"}},
             function(dst, src1, value, src2)
                dst:resizeAs(src1)
                torch.apply3(dst, src1, src2, function(dst, dstst, src1, src1st, src2, src2st, sz)
                                                 th.cadd_real(dst, dstst, src1, src1st, src2, src2st, sz, value)
                                              end)
                return dst
             end}

torch.Tensor.mul =
   argcheck{{{name="dst", type="torch.Tensor", default=true},
             {name="src", type="torch.Tensor"},
             {name="value", type="number"}},
            function(dst, src, value)
               dst:resizeAs(src)
               torch.apply2(dst, src, function(y, stry, x, strx, sz)
                                         th.mul_real(y, stry, x, strx, sz, value)
                                      end)
               return dst
            end}

torch.Tensor.cmul = 
   argcheck{{{name="dst", type="torch.Tensor", default=true},
             {name="src1", type="torch.Tensor"},
             {name="src2", type="torch.Tensor"}},
            function(dst, src1, src2)
               dst:resizeAs(src1)
               torch.apply3(dst, src1, src2, th.cmul_real)
               return dst
            end}


torch.Tensor.div =
   argcheck{{{name="dst", type="torch.Tensor", default=true},
             {name="src", type="torch.Tensor"},
             {name="value", type="number"}},
             function(dst, src, value)
                dst:resizeAs(src)
                torch.apply2(dst, src, function(y, stry, x, strx, sz)
                                          th.div_real(y, stry, x, strx, sz, value)
                                       end)
                return dst
             end}

torch.Tensor.cdiv = 
   argcheck{{{name="dst", type="torch.Tensor", default=true},
             {name="src1", type="torch.Tensor"},
             {name="src2", type="torch.Tensor"}},
             function(dst, src1, src2)
                dst:resizeAs(src1)
                torch.apply3(dst, src1, src2, th.cdiv_real)
                return dst
             end}

torch.Tensor.addcmul = 
   argcheck{{{name="dst", type="torch.Tensor"},
             {name="value", type="number", default=1},
             {name="src1", type="torch.Tensor"},
             {name="src2", type="torch.Tensor"}},
             function(dst, value, src1, src2)
                dst:resizeAs(src1)
                torch.apply3(dst, src1, src2, function(dst, dstst, src1, src1st, src2, src2st, sz)
                                                 th.addcmul_real(dst, dstst, src1, src1st, src2, src2st, sz, value)
                                              end)
                return dst
             end}

torch.Tensor.addcdiv = 
   argcheck{{{name="dst", type="torch.Tensor"},
             {name="value", type="number", default=1},
             {name="src1", type="torch.Tensor"},
             {name="src2", type="torch.Tensor"}},
             function(dst, value, src1, src2)
                dst:resizeAs(src1)
                torch.apply3(dst, src1, src2, function(dst, dstst, src1, src1st, src2, src2st, sz)
                                                 th.addcdiv_real(dst, dstst, src1, src1st, src2, src2st, sz, value)
                                              end)
                return dst
             end}

torch.Tensor.trace =
   argcheck({{name="src", type="torch.Tensor", dim=2}},
            function(src)
               return th.sum_float(src:data(),
                                   src:stride(1)+src:stride(2),
                                   math.min(src:size(1), src:size(2)))
            end)

for _,name in ipairs{'log', 'log1p', 'exp', 'cos', 'acos', 'cosh', 'sin', 'asin',
                     'sinh', 'tan', 'atan', 'tanh', 'sqrt', 'ceil', 'floor', 'abs'} do

   ffi.cdef(string.format('void %s_float(float *y, long stry, float *x, long strx, long sz);', name)) -- DEBUG: see below

   local func = th[name .. '_float'] -- DEBUG: *MUST* be real here, but i did not defined them yet ;)
   torch.Tensor[name] =
      argcheck({{name="dst", type="torch.Tensor", default=true},
                {name="src", type="torch.Tensor"}},
               function(dst, src)
                  dst:resizeAs(src)
                  torch.apply2(dst, src, func)
                  return dst
            end)

end

ffi.cdef('void pow_float(float *y, long stry, float *x, long strx, long sz, float value);')

torch.Tensor.pow =
   argcheck({{name="dst", type="torch.Tensor", default=true},
             {name="src", type="torch.Tensor"},
             {name="value", type="number"}},
            function(dst, src, value)
               dst:resizeAs(src)
               torch.apply2(dst, src, function(dst, dstst, src, srcst, sz)
                                         th.pow_real(dst, dstst, src, srcst, sz, value)
                                      end)
               return dst
            end)

torch.Tensor.zeros =
   argcheck({{name="dst", type='torch.Tensor', default=true},
             {name="size", type="numbers"}},
            function(dst, size)
               dst:resize(size)
               dst:zero()
               return dst
            end)

torch.Tensor.ones =
   argcheck({{name="dst", type='torch.Tensor', default=true},
             {name="size", type="numbers"}},
            function(dst, size)
               dst:resize(size)
               dst:fill(1)
               return dst
            end)

torch.Tensor.diag = argcheck{
   {{name="dst", type='torch.Tensor', default=true},
    {name="src", type='torch.Tensor'},
    {name="k", type='number', default=0}},
   function(dst, src)
      assert(src.__nDimension == 1 or src.__nDimension == 2, "matrix or vector expected")
      if src.__nDimension == 1 then
         local sz = src.__size[0] + (k >= 0 and k or -k)
         dst:resize(sz, sz):zero()
         th.copy(dst:data() + (k >= 0 and k*dst.__stride[1] or -k*dst.__stride[0]),
                 dst.__stride[0]+dst.__stride[1],
                 src:data(),
                 src.__stride[0],
                 sz)
      else
         local sz
         if k >= 0 then
            sz = math.min(src.__size[0], src.__size[1]-k)
         else
            sz = math.min(src.__size[0]+k, src.__size[1])
         end
         dst:resize(sz)
         th.copy(dst:data(),
                 dst.__stride[0],
                 src:data() + (k >= 0 and k*src.__stride[1] or -k*src.__stride[0]),
                 src.__stride[0]+src.__stride[1],
                 sz)
      end
      return dst
   end
}
