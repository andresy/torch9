local function findclib()
   for path in string.gmatch(package.cpath, '[^%;]+') do
      path = path:gsub('%?', 'libtorch')
      local f = io.open(path)
      if f then
         f:close()
         return path
      end
   end
end

local ffi = require 'ffi'

for _, ctype in ipairs{'byte', 'char', 'short', 'int', 'long', 'float', 'double'} do
   local defs = string.gsub([[
void th_swap_real(long n, real *x, long incx, real *y, long incy);
void th_scal_real(long n, real a, real *x, long incx);
void th_copy_real(long n, real *x, long incx, real *y, long incy);
void th_axpy_real(long n, real a, real *x, long incx, real *y, long incy);
real th_dot_real(long n, real *x, long incx, real *y, long incy);
void th_gemv_real(char trans, long m, long n, real alpha, real *a, long lda, real *x, long incx, real beta, real *y, long incy);
void th_ger_real(long m, long n, real alpha, real *x, long incx, real *y, long incy, real *a, long lda);
void th_gemm_real(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta, real *c, long ldc);
void th_zero_real(long sz, real *x, long inc);
void th_fill_real(long sz, real value, real *x, long inc);
void th_min_real(long sz, real *x, long incx, real *min_, long *idx_);
void th_max_real(long sz, real *x, long incx, real *max_, long *idx_);
real th_sum_real(long sz, real *x, long incx);
real th_sum2_real(long sz, real *x, long incx);
void th_sum_sum2_real(long sz, real *x, long incx, real *sum_, real *sum2_);
real th_prod_real(long sz, real *x, long incx);
void th_cumsum_real(long sz, real *x, long incx, real *cumsum, long inccumsum);
void th_cumprod_real(long sz, real *x, long incx, real *cumprod, long inccumprod);
real th_norm_real(long sz, real n, int dopow, real *x, long incx);
void th_add_real(long sz, real value, real *x, long incx, real *y, long incy);
void th_cadd_real(long sz, real *x, long incx, real value, real *y, long incy, real *z, long incz);
void th_mul_real(long sz, real *x, long incx, real value, real *y, long incy);
void th_cmul_real(long sz, real *x, long incx, real *y, long incy, real *z, long incz);
void th_div_real(long sz, real value, real *x, long incx, real *y, long incy);
void th_cdiv_real(long sz, real *x, long incx, real *y, long incy, real *z, long incz);
void th_addcmul_real(long sz, real value, real *x, long incx, real *y, long incy, real *z, long incz);
void th_addcdiv_real(long sz, real value, real *x, long incx, real *y, long incy, real *z, long incz);
void th_log_real(long sz, real *x, long incx, real *y, long incy);
void th_log1p_real(long sz, real *x, long incx, real *y, long incy);
void th_exp_real(long sz, real *x, long incx, real *y, long incy);
void th_cos_real(long sz, real *x, long incx, real *y, long incy);
void th_acos_real(long sz, real *x, long incx, real *y, long incy);
void th_cosh_real(long sz, real *x, long incx, real *y, long incy);
void th_sin_real(long sz, real *x, long incx, real *y, long incy);
void th_asin_real(long sz, real *x, long incx, real *y, long incy);
void th_sinh_real(long sz, real *x, long incx, real *y, long incy);
void th_tan_real(long sz, real *x, long incx, real *y, long incy);
void th_atan_real(long sz, real *x, long incx, real *y, long incy);
void th_tanh_real(long sz, real *x, long incx, real *y, long incy);
void th_sqrt_real(long sz, real *x, long incx, real *y, long incy);
void th_ceil_real(long sz, real *x, long incx, real *y, long incy);
void th_floor_real(long sz, real *x, long incx, real *y, long incy);
void th_abs_real(long sz, real *x, long incx, real *y, long incy);
void th_pow_real(long sz, real value, real *x, long incx, real *y, long incy);

void th_copy_real_byte(long sz, byte *x, long incx, real *y, long incy);
void th_copy_real_char(long sz, char *x, long incx, real *y, long incy);
void th_copy_real_short(long sz, short *x, long incx, real *y, long incy);
void th_copy_real_int(long sz, int *x, long incx, real *y, long incy);
void th_copy_real_long(long sz, long *x, long incx, real *y, long incy);
void th_copy_real_float(long sz, float *x, long incx, real *y, long incy);
void th_copy_real_double(long sz, double *x, long incx, real *y, long incy);

unsigned long th_seed();
void th_manualseed(unsigned long the_seed_);
unsigned long th_initialseed();
void th_nextstate();
unsigned long th_random();
]], 'real', ctype)
   ffi.cdef(defs)
end

local clibpath = findclib()

assert(clibpath, 'torch C library not found')

return ffi.load(clibpath)
