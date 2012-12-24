#include <math.h>
#include <string.h>

void zero_real(real *x, long str, long sz)
{
  if(str == 1)
    memset(x, 0, sizeof(real)*sz);
  else
  {
    long i;    
    for(i = 0; i < sz; i++)
      x[i*str] = 0;
  }
}

void fill_real(real *x, long str, long sz, real value)
{
  long i;

  for(i = 0; i < sz; i++)
    x[i*str] = value;
}

void copy_real(real *y, long stry, real *x, long strx, long sz)
{
  long i;

  for(i = 0; i < sz; i++)
    y[i*stry] = x[i*strx];
}

real dot_real(real *x, long strx, real *y, long stry, long sz)
{
  long i;
  accreal sum = 0;
  for(i = 0; i < sz; i++)
    sum += x[i*strx]*y[i*stry];
  return (real)sum;
}

void min_real(real *min_, long *idx_, real *x, long strx, long sz)
{
  long i;
  real min = (sz > 0 ? x[0] : 0);
  long idx = 1; /* lua */
  for(i = 1; i < sz; i++)
  {
    real z = x[i*strx];
    if(z < min)
    {
      min = z;
      idx = i+1; /* lua */
    }
  }
  *min_ = min;
  *idx_ = idx;
}

void max_real(real *max_, long *idx_, real *x, long strx, long sz)
{
  long i;
  real max = (sz > 0 ? x[0] : 0);
  long idx = 1; /* lua */
  for(i = 1; i < sz; i++)
  {
    real z = x[i*strx];
    if(z > max)
    {
      max = z;
      idx = i+1; /* lua */
    }
  }
  *max_ = max;
  *idx_ = idx;
}

real sum_real(real *x, long strx, long sz)
{
  long i;
  accreal sum = 0;
  for(i = 0; i < sz; i++)
    sum += x[i*strx];
  return (real)sum;
}

real sum2_real(real *x, long strx, long sz)
{
  long i;
  accreal sum2 = 0;
  for(i = 0; i < sz; i++)
  {
    real z = x[i*strx];
    sum2 += z*z;
  }
  return (real)sum2;
}

void sum_sum2_real(real *sum_, real *sum2_, real *x, long strx, long sz)
{
  long i;
  accreal sum = 0;
  accreal sum2 = 0;
  for(i = 0; i < sz; i++)
  {
    real z = x[i*strx];
    sum += z;
    sum2 += z*z;
  }
  *sum_ = sum;
  *sum2_ = sum2;
}

void prod_real(real *prod_, real *x, long strx, long sz)
{
  long i;
  accreal prod = (sz > 0 ? x[0] : 0);
  for(i = 1; i < sz; i++)
    prod *= x[i*strx];
  *prod_ = (real)prod;
}

void cumsum_real(real *cumsum, long cumsumst, long cumsumsz, real *x, long strx, long sz)
{
  long i;
  accreal sum = 0;
  for(i = 0; i < sz; i++)
  {
    sum += x[i*strx];
    cumsum[i*cumsumst] = (real)sum;
  }
}

void cumprod_real(real *cumprod, long cumprodst, long cumprodsz, real *x, long strx, long sz)
{
  long i;
  accreal prod = 1;
  for(i = 0; i < sz; i++)
  {
    prod *= x[i*strx];
    cumprod[i*cumprodst] = (real)prod;
  }
}

real norm_real(real *x, long strx, long sz, real n, int dopow)
{
  long i;
  accreal sum = 0;
  if(n == 1.0)
  {
    for(i = 0; i < sz; i++)
      sum += fabs(x[i*strx]);
  }
  else if(n == 2.0)
  {
    for(i = 0; i < sz; i++)
    {
      real z = x[i*strx];
      sum += z*z;
    }
  }
  else
  {
    for(i = 0; i < sz; i++)
      sum += pow(fabs(x[i*strx]), n);
  }

  if(dopow)
    return (real)pow(sum, 1/n);
  else
    return (real)sum;
}

void add_real(real *y, long stry, real *x, long strx, long sz, real value)
{
  long i;
  for(i = 0; i < sz; i++)
    y[i*stry] = x[i*strx] + value;
}

void cadd_real(real *z, long strz, real *y, long stry, real *x, long strx, long sz, real value)
{
  long i;
  for(i = 0; i < sz; i++)
    z[i*strz] = y[i*strx] + value*x[i*stry];
}

void mul_real(real *y, long stry, real *x, long strx, long sz, real value)
{
  long i;
  for(i = 0; i < sz; i++)
    y[i*stry] = x[i*strx] * value;
}

void cmul_real(real *z, long strz, real *y, long stry, real *x, long strx, long sz)
{
  long i;
  for(i = 0; i < sz; i++)
    z[i*strz] = x[i*strx] * y[i*stry];
}

void div_real(real *y, long stry, real *x, long strx, long sz, real value)
{
  long i;
  for(i = 0; i < sz; i++)
    y[i*stry] = x[i*strx] / value;
}

void cdiv_real(real *z, long strz, real *y, long stry, real *x, long strx, long sz)
{
  long i;
  for(i = 0; i < sz; i++)
    z[i*strz] = x[i*strx] / y[i*stry];
}

void addcmul_real(real *z, long strz, real *y, long stry, real *x, long strx, long sz, real value)
{
  long i;
  for(i = 0; i < sz; i++)
    z[i*strz] += value * x[i*strx] * y[i*stry];
}

void addcdiv_real(real *z, long strz, real *y, long stry, real *x, long strx, long sz, real value)
{
  long i;
  for(i = 0; i < sz; i++)
    z[i*strz] += value * x[i*strx] / y[i*stry];
}

#define BASIC_FUNC(NAME)                                                \
  void NAME##_real(real *y, long stry, real *x, long strx, long sz)  \
  {                                                                     \
    long i;                                                             \
    for(i = 0; i < sz; i++)                                             \
      y[i*stry] = NAME(x[i*strx]);                                      \
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

void abs_real(real *y, long stry, real *x, long strx, long sz)
{
  long i;
  for(i = 0; i < sz; i++)
    y[i*stry] = fabs(x[i*strx]);
}

void pow_real(real *y, long stry, real *x, long strx, long sz, real value)
{
  long i;
  for(i = 0; i < sz; i++)
    y[i*stry] = pow(x[i*strx], value);
}

#define COPY_REAL(REAL)                                                 \
  void copy_real_##REAL(real *y, long sty, REAL *x, long stx, long sz)  \
  {                                                                     \
    long i;                                                             \
    for(i = 0; i < sz; i++)                                             \
      y[i*sty] = (real)x[i*stx];                                        \
  }                                                                     \

COPY_REAL(byte)
COPY_REAL(char)
COPY_REAL(short)
COPY_REAL(int)
COPY_REAL(long)
COPY_REAL(float)
COPY_REAL(double)
