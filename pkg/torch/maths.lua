local ffi = require 'ffi'

ffi.cdef[[
      void zero_float(float *x, long str, long sz);
      void fill_float(float *x, long str, long sz, float value);
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
]]

local th = ffi.load(paths.concat(paths.install_lua_path,
                                 'torch',
                                 ((jit.os == 'Windows') and '' or 'lib') .. 'maths' .. 
                                 ((jit.os == 'Windows') and '.dll' or ((jit.os == 'OSX') and '.dylib' or '.so'))))
print('we loaded real')

torch.Tensor.fill =
   argcheck({{name="dst", type="torch.Tensor"},
             {name="value", type="number"}},
            function(dst, value)
               torch.apply(dst, function(x, str, sz)
                                   th.fill_real(x, str, sz, value)
                                end)
               return dst
            end)

torch.Tensor.zero =
   argcheck({{name="dst", type="torch.Tensor"}},
            function(dst)
               torch.apply(dst, th.zero_real)
               return dst
            end)

torch.Tensor.dot =
   argcheck({{name="vec1", type="torch.Tensor"},
             {name="vec2", type="torch.Tensor"}},
            function(vec1, vec2)
               local sum = 0
               torch.apply2(vec1, vec2, function(x, strx, y, stry, sz)
                                           sum = sum + th.dot_real(x, strx, y, stry, sz)
                                        end)
               return sum
            end)

torch.Tensor.min =
   argcheck{{{name="src", type="torch.Tensor"}},
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
             {name="dim", type="number"}},
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

torch.Tensor.max =
   argcheck{{{name="src", type="torch.Tensor"}},
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
             {name="dim", type="number"}},
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

torch.Tensor.sum =
   argcheck{{{name="src", type="torch.Tensor"}},
            function(src)
               local sum = 0
               torch.apply(src, function(x, str, sz)
                                   sum = sum + th.sum_real(x, str, sz)
                                end)
               return sum
            end,

            {{name="dst", type="torch.Tensor", default=true},
             {name="src", type="torch.Tensor"},
             {name="dim", type="number"}},
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

torch.Tensor.prod =
   argcheck{{{name="src", type="torch.Tensor"}},
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
             {name="dim", type="number"}},
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

torch.Tensor.cumsum =
   argcheck({{name="dst", type="torch.Tensor", default=true},
             {name="src", type="torch.Tensor"},
             {name="dim", type="number"}},
            function(dst, src, dim)
               dst:resizeAs(src)
               torch.dimapply2(dst, src, dim, th.cumsum_real)
               return dst
            end)

torch.Tensor.cumprod =
   argcheck({{name="dst", type="torch.Tensor", default=true},
             {name="src", type="torch.Tensor"},
             {name="dim", type="number"}},
            function(dst, src, dim)
               dst:resizeAs(src)
               torch.dimapply2(dst, src, dim, th.cumprod_real)
               return dst
            end)

-- float only

torch.Tensor.norm =
   argcheck{{{name="src", type="torch.Tensor"},
             {name="n", type="number", default=2}},
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
             {name="dim", type="number"}},
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

torch.Tensor.mean =
   argcheck{{{name="src", type="torch.Tensor"}},
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

torch.Tensor.std =
   argcheck{{{name="src", type="torch.Tensor"},
             {name="flag", type="boolean", default=false}},
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
             {name="flag", type="boolean", default=false}},
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

torch.Tensor.var =
   argcheck{{{name="src", type="torch.Tensor"},
             {name="flag", type="boolean", default=false}},
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
             {name="flag", type="boolean", default=false}},
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

torch.Tensor.zeros =
   argcheck({{name="size", type="numbers"}},
            function(size)
               print(size)
            end)
