#include <math.h>
#include <string.h>

void zero_float(float *x, long str, long sz)
{
  if(str == 1)
    memset(x, 0, sizeof(float)*sz);
  else
  {
    long i;    
    for(i = 0; i < sz; i++)
      x[i*str] = 0;
  }
}

void fill_float(float *x, long str, long sz, float value)
{
  long i;

  for(i = 0; i < sz; i++)
    x[i*str] = value;
}

float dot_float(float *x, long strx, float *y, long stry, long sz)
{
  long i;
  double sum = 0;
  for(i = 0; i < sz; i++)
    sum += x[i*strx]*y[i*stry];
  return (float)sum;
}

void min_float(float *min_, long *idx_, float *x, long strx, long sz)
{
  long i;
  float min = (sz > 0 ? x[0] : 0);
  long idx = 1; /* lua */
  for(i = 1; i < sz; i++)
  {
    float z = x[i*strx];
    if(z < min)
    {
      min = z;
      idx = i+1; /* lua */
    }
  }
  *min_ = min;
  *idx_ = idx;
}

void max_float(float *max_, long *idx_, float *x, long strx, long sz)
{
  long i;
  float max = (sz > 0 ? x[0] : 0);
  long idx = 1; /* lua */
  for(i = 1; i < sz; i++)
  {
    float z = x[i*strx];
    if(z > max)
    {
      max = z;
      idx = i+1; /* lua */
    }
  }
  *max_ = max;
  *idx_ = idx;
}

float sum_float(float *x, long strx, long sz)
{
  long i;
  double sum = 0;
  for(i = 0; i < sz; i++)
    sum += x[i*strx];
  return (float)sum;
}

float sum2_float(float *x, long strx, long sz)
{
  long i;
  double sum2 = 0;
  for(i = 0; i < sz; i++)
  {
    float z = x[i*strx];
    sum2 += z*z;
  }
  return (float)sum2;
}

void sum_sum2_float(float *sum_, float *sum2_, float *x, long strx, long sz)
{
  long i;
  double sum = 0;
  double sum2 = 0;
  for(i = 0; i < sz; i++)
  {
    float z = x[i*strx];
    sum += z;
    sum2 += z*z;
  }
  *sum_ = sum;
  *sum2_ = sum2;
}

void prod_float(float *prod_, float *x, long strx, long sz)
{
  long i;
  double prod = (sz > 0 ? x[0] : 0);
  for(i = 1; i < sz; i++)
    prod *= x[i*strx];
  *prod_ = (float)prod;
}

void cumsum_float(float *cumsum, long cumsumst, long cumsumsz, float *x, long strx, long sz)
{
  long i;
  double sum = 0;
  for(i = 0; i < sz; i++)
  {
    sum += x[i*strx];
    cumsum[i*cumsumst] = (float)sum;
  }
}

void cumprod_float(float *cumprod, long cumprodst, long cumprodsz, float *x, long strx, long sz)
{
  long i;
  double prod = 1;
  for(i = 0; i < sz; i++)
  {
    prod *= x[i*strx];
    cumprod[i*cumprodst] = (float)prod;
  }
}

float norm_float(float *x, long strx, long sz, float n, int dopow)
{
  long i;
  double sum = 0;
  if(n == 1.0)
  {
    for(i = 0; i < sz; i++)
      sum += fabs(x[i*strx]);
  }
  else if(n == 2.0)
  {
    for(i = 0; i < sz; i++)
    {
      float z = x[i*strx];
      sum += z*z;
    }
  }
  else
  {
    for(i = 0; i < sz; i++)
      sum += pow(fabs(x[i*strx]), n);
  }

  if(dopow)
    return (float)pow(sum, 1/n);
  else
    return (float)sum;
}
