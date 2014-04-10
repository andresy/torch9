#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct TH_Storage
{
   real *__data;
   long __size;
   int __refcount;
   char __flag;

} TH_Storage;


void TH_Storage_gc(TH_Storage *self)
{
  self->__refcount = self->__refcount - 1;
  if(self->__refcount == 0)
  {
    if(self->__data)
      free(self->__data);
    free(self);
  }
}

void th_zero_real(long sz, real *x, long inc)
{
  if(inc == 1)
    memset(x, 0, sizeof(real)*sz);
  else
  {
    long i;    
    for(i = 0; i < sz; i++)
      x[i*inc] = 0;
  }
}

void th_fill_real(long sz, real value, real *x, long inc)
{
  long i;

  for(i = 0; i < sz; i++)
    x[i*inc] = value;
}

void th_min_real(long sz, real *x, long incx, real *min_, long *idx_)
{
  long i;
  real min = (sz > 0 ? x[0] : 0);
  long idx = 1; /* lua */
  for(i = 1; i < sz; i++)
  {
    real z = x[i*incx];
    if(z < min)
    {
      min = z;
      idx = i+1; /* lua */
    }
  }
  *min_ = min;
  *idx_ = idx;
}

void th_max_real(long sz, real *x, long incx, real *max_, long *idx_)
{
  long i;
  real max = (sz > 0 ? x[0] : 0);
  long idx = 1; /* lua */
  for(i = 1; i < sz; i++)
  {
    real z = x[i*incx];
    if(z > max)
    {
      max = z;
      idx = i+1; /* lua */
    }
  }
  *max_ = max;
  *idx_ = idx;
}

real th_sum_real(long sz, real *x, long incx)
{
  long i;
  accreal sum = 0;
  for(i = 0; i < sz; i++)
    sum += x[i*incx];
  return (real)sum;
}

real th_sum2_real(long sz, real *x, long incx)
{
  long i;
  accreal sum2 = 0;
  for(i = 0; i < sz; i++)
  {
    real z = x[i*incx];
    sum2 += z*z;
  }
  return (real)sum2;
}

void th_sum_sum2_real(long sz, real *x, long incx, real *sum_, real *sum2_)
{
  long i;
  accreal sum = 0;
  accreal sum2 = 0;
  for(i = 0; i < sz; i++)
  {
    real z = x[i*incx];
    sum += z;
    sum2 += z*z;
  }
  *sum_ = sum;
  *sum2_ = sum2;
}

real th_prod_real(long sz, real *x, long incx)
{
  long i;
  accreal prod = (sz > 0 ? x[0] : 0);
  for(i = 1; i < sz; i++)
    prod *= x[i*incx];
  return (real)prod;
}

void th_cumsum_real(long sz, real *x, long incx, real *cumsum, long inccumsum)
{
  long i;
  accreal sum = 0;
  for(i = 0; i < sz; i++)
  {
    sum += x[i*incx];
    cumsum[i*inccumsum] = (real)sum;
  }
}

void th_cumprod_real(long sz, real *x, long incx, real *cumprod, long inccumprod)
{
  long i;
  accreal prod = 1;
  for(i = 0; i < sz; i++)
  {
    prod *= x[i*incx];
    cumprod[i*inccumprod] = (real)prod;
  }
}

real th_norm_real(long sz, real n, int dopow, real *x, long incx)
{
  long i;
  accreal sum = 0;
  if(n == 1.0)
  {
    for(i = 0; i < sz; i++)
      sum += fabs(x[i*incx]);
  }
  else if(n == 2.0)
  {
    for(i = 0; i < sz; i++)
    {
      real z = x[i*incx];
      sum += z*z;
    }
  }
  else
  {
    for(i = 0; i < sz; i++)
      sum += pow(fabs(x[i*incx]), n);
  }

  if(dopow)
    return (real)pow(sum, 1/n);
  else
    return (real)sum;
}

void th_add_real(long sz, real value, real *x, long incx, real *y, long incy)
{
  long i;
  for(i = 0; i < sz; i++)
    y[i*incy] = x[i*incx] + value;
}

void th_cadd_real(long sz, real *x, long incx, real value, real *y, long incy, real *z, long incz)
{
  long i;
  for(i = 0; i < sz; i++)
    z[i*incz] = x[i*incx] + value*y[i*incy];
}

void th_mul_real(long sz, real value, real *x, long incx, real *y, long incy)
{
  long i;
  for(i = 0; i < sz; i++)
    y[i*incy] = x[i*incx] * value;
}

void th_cmul_real(long sz, real *x, long incx, real *y, long incy, real *z, long incz)
{
  long i;
  for(i = 0; i < sz; i++)
    z[i*incz] = x[i*incx] * y[i*incy];
}

void th_div_real(long sz, real value, real *x, long incx, real *y, long incy)
{
  long i;
  for(i = 0; i < sz; i++)
    y[i*incy] = x[i*incx] / value;
}

void th_cdiv_real(long sz, real *x, long incx, real *y, long incy, real *z, long incz)
{
  long i;
  for(i = 0; i < sz; i++)
    z[i*incz] = x[i*incx] / y[i*incy];
}

void th_addcmul_real(long sz, real value, real *x, long incx, real *y, long incy, real *z, long incz)
{
  long i;
  for(i = 0; i < sz; i++)
    z[i*incz] += value * x[i*incx] * y[i*incy];
}

void th_addcdiv_real(long sz, real value, real *x, long incx, real *y, long incy, real *z, long incz)
{
  long i;
  for(i = 0; i < sz; i++)
    z[i*incz] += value * x[i*incx] / y[i*incy];
}

#define BASIC_FUNC(NAME)                                                \
  void th_##NAME##_real(long sz, real *x, long incx, real *y, long incy) \
  {                                                                     \
    long i;                                                             \
    for(i = 0; i < sz; i++)                                             \
      y[i*incy] = NAME(x[i*incx]);                                      \
  }                                                                     \


BASIC_FUNC(log)
BASIC_FUNC(log1p)
BASIC_FUNC(exp)
BASIC_FUNC(cos)
BASIC_FUNC(acos)
BASIC_FUNC(cosh)
BASIC_FUNC(sin)
BASIC_FUNC(asin)
BASIC_FUNC(sinh)
BASIC_FUNC(tan)
BASIC_FUNC(atan)
BASIC_FUNC(tanh)
BASIC_FUNC(sqrt)
BASIC_FUNC(ceil)
BASIC_FUNC(floor)

void th_abs_real(long sz, real *x, long incx, real *y, long incy)
{
  long i;
  for(i = 0; i < sz; i++)
    y[i*incy] = fabs(x[i*incx]);
}

void th_pow_real(long sz, real value, real *x, long incx, real *y, long incy)
{
  long i;
  for(i = 0; i < sz; i++)
    y[i*incy] = pow(x[i*incx], value);
}

#define COPY_REAL(REAL)                                                 \
  void th_copy_real_##REAL(long sz, REAL *x, long incx, real *y, long incy) \
  {                                                                     \
    long i;                                                             \
    for(i = 0; i < sz; i++)                                             \
      y[i*incy] = (real)x[i*incx];                                        \
  }                                                                     \

COPY_REAL(byte)
COPY_REAL(char)
COPY_REAL(short)
COPY_REAL(int)
COPY_REAL(long)
COPY_REAL(float)
COPY_REAL(double)
