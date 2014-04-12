local ffi = require 'ffi'

ffi.cdef[[

extern double THLog1p(const double x);
extern void THError(const char *fmt, ...);
extern void THSetErrorHandler( void (*torchErrorHandlerFunction)(const char *msg, void *data), void *data );
extern void THArgCheck(int condition, int argNumber, const char *msg);
extern void THSetArgErrorHandler( void (*torchArgErrorHandlerFunction)(int argNumber, const char *msg, void *data), void *data );
extern void* THAlloc(long size);
extern void* THRealloc(void *ptr, long size);
extern void THFree(void *ptr);














extern void THByteBlas_swap(long n, unsigned char *x, long incx, unsigned char *y, long incy);
extern void THByteBlas_scal(long n, unsigned char a, unsigned char *x, long incx);
extern void THByteBlas_copy(long n, unsigned char *x, long incx, unsigned char *y, long incy);
extern void THByteBlas_axpy(long n, unsigned char a, unsigned char *x, long incx, unsigned char *y, long incy);
extern unsigned char THByteBlas_dot(long n, unsigned char *x, long incx, unsigned char *y, long incy);


extern void THByteBlas_gemv(char trans, long m, long n, unsigned char alpha, unsigned char *a, long lda, unsigned char *x, long incx, unsigned char beta, unsigned char *y, long incy);
extern void THByteBlas_ger(long m, long n, unsigned char alpha, unsigned char *x, long incx, unsigned char *y, long incy, unsigned char *a, long lda);


extern void THByteBlas_gemm(char transa, char transb, long m, long n, long k, unsigned char alpha, unsigned char *a, long lda, unsigned char *b, long ldb, unsigned char beta, unsigned char *c, long ldc);





extern void THCharBlas_swap(long n, char *x, long incx, char *y, long incy);
extern void THCharBlas_scal(long n, char a, char *x, long incx);
extern void THCharBlas_copy(long n, char *x, long incx, char *y, long incy);
extern void THCharBlas_axpy(long n, char a, char *x, long incx, char *y, long incy);
extern char THCharBlas_dot(long n, char *x, long incx, char *y, long incy);


extern void THCharBlas_gemv(char trans, long m, long n, char alpha, char *a, long lda, char *x, long incx, char beta, char *y, long incy);
extern void THCharBlas_ger(long m, long n, char alpha, char *x, long incx, char *y, long incy, char *a, long lda);


extern void THCharBlas_gemm(char transa, char transb, long m, long n, long k, char alpha, char *a, long lda, char *b, long ldb, char beta, char *c, long ldc);





extern void THShortBlas_swap(long n, short *x, long incx, short *y, long incy);
extern void THShortBlas_scal(long n, short a, short *x, long incx);
extern void THShortBlas_copy(long n, short *x, long incx, short *y, long incy);
extern void THShortBlas_axpy(long n, short a, short *x, long incx, short *y, long incy);
extern short THShortBlas_dot(long n, short *x, long incx, short *y, long incy);


extern void THShortBlas_gemv(char trans, long m, long n, short alpha, short *a, long lda, short *x, long incx, short beta, short *y, long incy);
extern void THShortBlas_ger(long m, long n, short alpha, short *x, long incx, short *y, long incy, short *a, long lda);


extern void THShortBlas_gemm(char transa, char transb, long m, long n, long k, short alpha, short *a, long lda, short *b, long ldb, short beta, short *c, long ldc);





extern void THIntBlas_swap(long n, int *x, long incx, int *y, long incy);
extern void THIntBlas_scal(long n, int a, int *x, long incx);
extern void THIntBlas_copy(long n, int *x, long incx, int *y, long incy);
extern void THIntBlas_axpy(long n, int a, int *x, long incx, int *y, long incy);
extern int THIntBlas_dot(long n, int *x, long incx, int *y, long incy);


extern void THIntBlas_gemv(char trans, long m, long n, int alpha, int *a, long lda, int *x, long incx, int beta, int *y, long incy);
extern void THIntBlas_ger(long m, long n, int alpha, int *x, long incx, int *y, long incy, int *a, long lda);


extern void THIntBlas_gemm(char transa, char transb, long m, long n, long k, int alpha, int *a, long lda, int *b, long ldb, int beta, int *c, long ldc);





extern void THLongBlas_swap(long n, long *x, long incx, long *y, long incy);
extern void THLongBlas_scal(long n, long a, long *x, long incx);
extern void THLongBlas_copy(long n, long *x, long incx, long *y, long incy);
extern void THLongBlas_axpy(long n, long a, long *x, long incx, long *y, long incy);
extern long THLongBlas_dot(long n, long *x, long incx, long *y, long incy);


extern void THLongBlas_gemv(char trans, long m, long n, long alpha, long *a, long lda, long *x, long incx, long beta, long *y, long incy);
extern void THLongBlas_ger(long m, long n, long alpha, long *x, long incx, long *y, long incy, long *a, long lda);


extern void THLongBlas_gemm(char transa, char transb, long m, long n, long k, long alpha, long *a, long lda, long *b, long ldb, long beta, long *c, long ldc);





extern void THFloatBlas_swap(long n, float *x, long incx, float *y, long incy);
extern void THFloatBlas_scal(long n, float a, float *x, long incx);
extern void THFloatBlas_copy(long n, float *x, long incx, float *y, long incy);
extern void THFloatBlas_axpy(long n, float a, float *x, long incx, float *y, long incy);
extern float THFloatBlas_dot(long n, float *x, long incx, float *y, long incy);


extern void THFloatBlas_gemv(char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy);
extern void THFloatBlas_ger(long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda);


extern void THFloatBlas_gemm(char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc);





extern void THDoubleBlas_swap(long n, double *x, long incx, double *y, long incy);
extern void THDoubleBlas_scal(long n, double a, double *x, long incx);
extern void THDoubleBlas_copy(long n, double *x, long incx, double *y, long incy);
extern void THDoubleBlas_axpy(long n, double a, double *x, long incx, double *y, long incy);
extern double THDoubleBlas_dot(long n, double *x, long incx, double *y, long incy);


extern void THDoubleBlas_gemv(char trans, long m, long n, double alpha, double *a, long lda, double *x, long incx, double beta, double *y, long incy);
extern void THDoubleBlas_ger(long m, long n, double alpha, double *x, long incx, double *y, long incy, double *a, long lda);


extern void THDoubleBlas_gemm(char transa, char transb, long m, long n, long k, double alpha, double *a, long lda, double *b, long ldb, double beta, double *c, long ldc);














extern void THByteLapack_gesv(int n, int nrhs, unsigned char *a, int lda, int *ipiv, unsigned char *b, int ldb, int* info);

extern void THByteLapack_gels(char trans, int m, int n, int nrhs, unsigned char *a, int lda, unsigned char *b, int ldb, unsigned char *work, int lwork, int *info);

extern void THByteLapack_syev(char jobz, char uplo, int n, unsigned char *a, int lda, unsigned char *w, unsigned char *work, int lwork, int *info);

extern void THByteLapack_geev(char jobvl, char jobvr, int n, unsigned char *a, int lda, unsigned char *wr, unsigned char *wi, unsigned char* vl, int ldvl, unsigned char *vr, int ldvr, unsigned char *work, int lwork, int *info);

extern void THByteLapack_gesvd(char jobu, char jobvt, int m, int n, unsigned char *a, int lda, unsigned char *s, unsigned char *u, int ldu, unsigned char *vt, int ldvt, unsigned char *work, int lwork, int *info);

extern void THByteLapack_getrf(int m, int n, unsigned char *a, int lda, int *ipiv, int *info);

extern void THByteLapack_getri(int n, unsigned char *a, int lda, int *ipiv, unsigned char *work, int lwork, int* info);



void THByteLapack_potrf(char uplo, int n, unsigned char *a, int lda, int *info);

void THByteLapack_potri(char uplo, int n, unsigned char *a, int lda, int *info);

void THByteLapack_potrs(char uplo, int n, int nrhs, unsigned char *a, int lda, unsigned char *b, int ldb, int *info);





extern void THCharLapack_gesv(int n, int nrhs, char *a, int lda, int *ipiv, char *b, int ldb, int* info);

extern void THCharLapack_gels(char trans, int m, int n, int nrhs, char *a, int lda, char *b, int ldb, char *work, int lwork, int *info);

extern void THCharLapack_syev(char jobz, char uplo, int n, char *a, int lda, char *w, char *work, int lwork, int *info);

extern void THCharLapack_geev(char jobvl, char jobvr, int n, char *a, int lda, char *wr, char *wi, char* vl, int ldvl, char *vr, int ldvr, char *work, int lwork, int *info);

extern void THCharLapack_gesvd(char jobu, char jobvt, int m, int n, char *a, int lda, char *s, char *u, int ldu, char *vt, int ldvt, char *work, int lwork, int *info);

extern void THCharLapack_getrf(int m, int n, char *a, int lda, int *ipiv, int *info);

extern void THCharLapack_getri(int n, char *a, int lda, int *ipiv, char *work, int lwork, int* info);



void THCharLapack_potrf(char uplo, int n, char *a, int lda, int *info);

void THCharLapack_potri(char uplo, int n, char *a, int lda, int *info);

void THCharLapack_potrs(char uplo, int n, int nrhs, char *a, int lda, char *b, int ldb, int *info);





extern void THShortLapack_gesv(int n, int nrhs, short *a, int lda, int *ipiv, short *b, int ldb, int* info);

extern void THShortLapack_gels(char trans, int m, int n, int nrhs, short *a, int lda, short *b, int ldb, short *work, int lwork, int *info);

extern void THShortLapack_syev(char jobz, char uplo, int n, short *a, int lda, short *w, short *work, int lwork, int *info);

extern void THShortLapack_geev(char jobvl, char jobvr, int n, short *a, int lda, short *wr, short *wi, short* vl, int ldvl, short *vr, int ldvr, short *work, int lwork, int *info);

extern void THShortLapack_gesvd(char jobu, char jobvt, int m, int n, short *a, int lda, short *s, short *u, int ldu, short *vt, int ldvt, short *work, int lwork, int *info);

extern void THShortLapack_getrf(int m, int n, short *a, int lda, int *ipiv, int *info);

extern void THShortLapack_getri(int n, short *a, int lda, int *ipiv, short *work, int lwork, int* info);



void THShortLapack_potrf(char uplo, int n, short *a, int lda, int *info);

void THShortLapack_potri(char uplo, int n, short *a, int lda, int *info);

void THShortLapack_potrs(char uplo, int n, int nrhs, short *a, int lda, short *b, int ldb, int *info);





extern void THIntLapack_gesv(int n, int nrhs, int *a, int lda, int *ipiv, int *b, int ldb, int* info);

extern void THIntLapack_gels(char trans, int m, int n, int nrhs, int *a, int lda, int *b, int ldb, int *work, int lwork, int *info);

extern void THIntLapack_syev(char jobz, char uplo, int n, int *a, int lda, int *w, int *work, int lwork, int *info);

extern void THIntLapack_geev(char jobvl, char jobvr, int n, int *a, int lda, int *wr, int *wi, int* vl, int ldvl, int *vr, int ldvr, int *work, int lwork, int *info);

extern void THIntLapack_gesvd(char jobu, char jobvt, int m, int n, int *a, int lda, int *s, int *u, int ldu, int *vt, int ldvt, int *work, int lwork, int *info);

extern void THIntLapack_getrf(int m, int n, int *a, int lda, int *ipiv, int *info);

extern void THIntLapack_getri(int n, int *a, int lda, int *ipiv, int *work, int lwork, int* info);



void THIntLapack_potrf(char uplo, int n, int *a, int lda, int *info);

void THIntLapack_potri(char uplo, int n, int *a, int lda, int *info);

void THIntLapack_potrs(char uplo, int n, int nrhs, int *a, int lda, int *b, int ldb, int *info);





extern void THLongLapack_gesv(int n, int nrhs, long *a, int lda, int *ipiv, long *b, int ldb, int* info);

extern void THLongLapack_gels(char trans, int m, int n, int nrhs, long *a, int lda, long *b, int ldb, long *work, int lwork, int *info);

extern void THLongLapack_syev(char jobz, char uplo, int n, long *a, int lda, long *w, long *work, int lwork, int *info);

extern void THLongLapack_geev(char jobvl, char jobvr, int n, long *a, int lda, long *wr, long *wi, long* vl, int ldvl, long *vr, int ldvr, long *work, int lwork, int *info);

extern void THLongLapack_gesvd(char jobu, char jobvt, int m, int n, long *a, int lda, long *s, long *u, int ldu, long *vt, int ldvt, long *work, int lwork, int *info);

extern void THLongLapack_getrf(int m, int n, long *a, int lda, int *ipiv, int *info);

extern void THLongLapack_getri(int n, long *a, int lda, int *ipiv, long *work, int lwork, int* info);



void THLongLapack_potrf(char uplo, int n, long *a, int lda, int *info);

void THLongLapack_potri(char uplo, int n, long *a, int lda, int *info);

void THLongLapack_potrs(char uplo, int n, int nrhs, long *a, int lda, long *b, int ldb, int *info);





extern void THFloatLapack_gesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb, int* info);

extern void THFloatLapack_gels(char trans, int m, int n, int nrhs, float *a, int lda, float *b, int ldb, float *work, int lwork, int *info);

extern void THFloatLapack_syev(char jobz, char uplo, int n, float *a, int lda, float *w, float *work, int lwork, int *info);

extern void THFloatLapack_geev(char jobvl, char jobvr, int n, float *a, int lda, float *wr, float *wi, float* vl, int ldvl, float *vr, int ldvr, float *work, int lwork, int *info);

extern void THFloatLapack_gesvd(char jobu, char jobvt, int m, int n, float *a, int lda, float *s, float *u, int ldu, float *vt, int ldvt, float *work, int lwork, int *info);

extern void THFloatLapack_getrf(int m, int n, float *a, int lda, int *ipiv, int *info);

extern void THFloatLapack_getri(int n, float *a, int lda, int *ipiv, float *work, int lwork, int* info);



void THFloatLapack_potrf(char uplo, int n, float *a, int lda, int *info);

void THFloatLapack_potri(char uplo, int n, float *a, int lda, int *info);

void THFloatLapack_potrs(char uplo, int n, int nrhs, float *a, int lda, float *b, int ldb, int *info);





extern void THDoubleLapack_gesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb, int* info);

extern void THDoubleLapack_gels(char trans, int m, int n, int nrhs, double *a, int lda, double *b, int ldb, double *work, int lwork, int *info);

extern void THDoubleLapack_syev(char jobz, char uplo, int n, double *a, int lda, double *w, double *work, int lwork, int *info);

extern void THDoubleLapack_geev(char jobvl, char jobvr, int n, double *a, int lda, double *wr, double *wi, double* vl, int ldvl, double *vr, int ldvr, double *work, int lwork, int *info);

extern void THDoubleLapack_gesvd(char jobu, char jobvt, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt, double *work, int lwork, int *info);

extern void THDoubleLapack_getrf(int m, int n, double *a, int lda, int *ipiv, int *info);

extern void THDoubleLapack_getri(int n, double *a, int lda, int *ipiv, double *work, int lwork, int* info);



void THDoubleLapack_potrf(char uplo, int n, double *a, int lda, int *info);

void THDoubleLapack_potri(char uplo, int n, double *a, int lda, int *info);

void THDoubleLapack_potrs(char uplo, int n, int nrhs, double *a, int lda, double *b, int ldb, int *info);






static inline void THFloatVector_fill(float *x, const float c, const long n) {
  long i = 0;

  for(; i < n-4; i += 4)
  {
    x[i] = c;
    x[i+1] = c;
    x[i+2] = c;
    x[i+3] = c;
  }

  for(; i < n; i++)
    x[i] = c;
}

static inline void THFloatVector_add(float *y, const float *x, const float c, const long n)
{
  long i = 0;

  for(;i < n-4; i += 4)
  {
    y[i] += c * x[i];
    y[i+1] += c * x[i+1];
    y[i+2] += c * x[i+2];
    y[i+3] += c * x[i+3];
  }

  for(; i < n; i++)
    y[i] += c * x[i];
}

static inline void THFloatVector_diff(float *z, const float *x, const float *y, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    z[i] = x[i] - y[i];
    z[i+1] = x[i+1] - y[i+1];
    z[i+2] = x[i+2] - y[i+2];
    z[i+3] = x[i+3] - y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] - y[i];
}

static inline void THFloatVector_scale(float *y, const float c, const long n)
{
  long i = 0;

  for(; i < n-4; i +=4)
  {
    y[i] *= c;
    y[i+1] *= c;
    y[i+2] *= c;
    y[i+3] *= c;
  }

  for(; i < n; i++)
    y[i] *= c;
}

static inline void THFloatVector_mul(float *y, const float *x, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    y[i] *= x[i];
    y[i+1] *= x[i+1];
    y[i+2] *= x[i+2];
    y[i+3] *= x[i+3];
  }

  for(; i < n; i++)
    y[i] *= x[i];
}




static inline void THDoubleVector_fill(double *x, const double c, const long n) {
  long i = 0;

  for(; i < n-4; i += 4)
  {
    x[i] = c;
    x[i+1] = c;
    x[i+2] = c;
    x[i+3] = c;
  }

  for(; i < n; i++)
    x[i] = c;
}

static inline void THDoubleVector_add(double *y, const double *x, const double c, const long n)
{
  long i = 0;

  for(;i < n-4; i += 4)
  {
    y[i] += c * x[i];
    y[i+1] += c * x[i+1];
    y[i+2] += c * x[i+2];
    y[i+3] += c * x[i+3];
  }

  for(; i < n; i++)
    y[i] += c * x[i];
}

static inline void THDoubleVector_diff(double *z, const double *x, const double *y, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    z[i] = x[i] - y[i];
    z[i+1] = x[i+1] - y[i+1];
    z[i+2] = x[i+2] - y[i+2];
    z[i+3] = x[i+3] - y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] - y[i];
}

static inline void THDoubleVector_scale(double *y, const double c, const long n)
{
  long i = 0;

  for(; i < n-4; i +=4)
  {
    y[i] *= c;
    y[i+1] *= c;
    y[i+2] *= c;
    y[i+3] *= c;
  }

  for(; i < n; i++)
    y[i] *= c;
}

static inline void THDoubleVector_mul(double *y, const double *x, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    y[i] *= x[i];
    y[i+1] *= x[i+1];
    y[i+2] *= x[i+2];
    y[i+3] *= x[i+3];
  }

  for(; i < n; i++)
    y[i] *= x[i];
}








static inline void THByteVector_fill(unsigned char *x, const unsigned char c, const long n) {
  long i = 0;

  for(; i < n-4; i += 4)
  {
    x[i] = c;
    x[i+1] = c;
    x[i+2] = c;
    x[i+3] = c;
  }

  for(; i < n; i++)
    x[i] = c;
}

static inline void THByteVector_add(unsigned char *y, const unsigned char *x, const unsigned char c, const long n)
{
  long i = 0;

  for(;i < n-4; i += 4)
  {
    y[i] += c * x[i];
    y[i+1] += c * x[i+1];
    y[i+2] += c * x[i+2];
    y[i+3] += c * x[i+3];
  }

  for(; i < n; i++)
    y[i] += c * x[i];
}

static inline void THByteVector_diff(unsigned char *z, const unsigned char *x, const unsigned char *y, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    z[i] = x[i] - y[i];
    z[i+1] = x[i+1] - y[i+1];
    z[i+2] = x[i+2] - y[i+2];
    z[i+3] = x[i+3] - y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] - y[i];
}

static inline void THByteVector_scale(unsigned char *y, const unsigned char c, const long n)
{
  long i = 0;

  for(; i < n-4; i +=4)
  {
    y[i] *= c;
    y[i+1] *= c;
    y[i+2] *= c;
    y[i+3] *= c;
  }

  for(; i < n; i++)
    y[i] *= c;
}

static inline void THByteVector_mul(unsigned char *y, const unsigned char *x, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    y[i] *= x[i];
    y[i+1] *= x[i+1];
    y[i+2] *= x[i+2];
    y[i+3] *= x[i+3];
  }

  for(; i < n; i++)
    y[i] *= x[i];
}




static inline void THCharVector_fill(char *x, const char c, const long n) {
  long i = 0;

  for(; i < n-4; i += 4)
  {
    x[i] = c;
    x[i+1] = c;
    x[i+2] = c;
    x[i+3] = c;
  }

  for(; i < n; i++)
    x[i] = c;
}

static inline void THCharVector_add(char *y, const char *x, const char c, const long n)
{
  long i = 0;

  for(;i < n-4; i += 4)
  {
    y[i] += c * x[i];
    y[i+1] += c * x[i+1];
    y[i+2] += c * x[i+2];
    y[i+3] += c * x[i+3];
  }

  for(; i < n; i++)
    y[i] += c * x[i];
}

static inline void THCharVector_diff(char *z, const char *x, const char *y, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    z[i] = x[i] - y[i];
    z[i+1] = x[i+1] - y[i+1];
    z[i+2] = x[i+2] - y[i+2];
    z[i+3] = x[i+3] - y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] - y[i];
}

static inline void THCharVector_scale(char *y, const char c, const long n)
{
  long i = 0;

  for(; i < n-4; i +=4)
  {
    y[i] *= c;
    y[i+1] *= c;
    y[i+2] *= c;
    y[i+3] *= c;
  }

  for(; i < n; i++)
    y[i] *= c;
}

static inline void THCharVector_mul(char *y, const char *x, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    y[i] *= x[i];
    y[i+1] *= x[i+1];
    y[i+2] *= x[i+2];
    y[i+3] *= x[i+3];
  }

  for(; i < n; i++)
    y[i] *= x[i];
}




static inline void THShortVector_fill(short *x, const short c, const long n) {
  long i = 0;

  for(; i < n-4; i += 4)
  {
    x[i] = c;
    x[i+1] = c;
    x[i+2] = c;
    x[i+3] = c;
  }

  for(; i < n; i++)
    x[i] = c;
}

static inline void THShortVector_add(short *y, const short *x, const short c, const long n)
{
  long i = 0;

  for(;i < n-4; i += 4)
  {
    y[i] += c * x[i];
    y[i+1] += c * x[i+1];
    y[i+2] += c * x[i+2];
    y[i+3] += c * x[i+3];
  }

  for(; i < n; i++)
    y[i] += c * x[i];
}

static inline void THShortVector_diff(short *z, const short *x, const short *y, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    z[i] = x[i] - y[i];
    z[i+1] = x[i+1] - y[i+1];
    z[i+2] = x[i+2] - y[i+2];
    z[i+3] = x[i+3] - y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] - y[i];
}

static inline void THShortVector_scale(short *y, const short c, const long n)
{
  long i = 0;

  for(; i < n-4; i +=4)
  {
    y[i] *= c;
    y[i+1] *= c;
    y[i+2] *= c;
    y[i+3] *= c;
  }

  for(; i < n; i++)
    y[i] *= c;
}

static inline void THShortVector_mul(short *y, const short *x, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    y[i] *= x[i];
    y[i+1] *= x[i+1];
    y[i+2] *= x[i+2];
    y[i+3] *= x[i+3];
  }

  for(; i < n; i++)
    y[i] *= x[i];
}




static inline void THIntVector_fill(int *x, const int c, const long n) {
  long i = 0;

  for(; i < n-4; i += 4)
  {
    x[i] = c;
    x[i+1] = c;
    x[i+2] = c;
    x[i+3] = c;
  }

  for(; i < n; i++)
    x[i] = c;
}

static inline void THIntVector_add(int *y, const int *x, const int c, const long n)
{
  long i = 0;

  for(;i < n-4; i += 4)
  {
    y[i] += c * x[i];
    y[i+1] += c * x[i+1];
    y[i+2] += c * x[i+2];
    y[i+3] += c * x[i+3];
  }

  for(; i < n; i++)
    y[i] += c * x[i];
}

static inline void THIntVector_diff(int *z, const int *x, const int *y, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    z[i] = x[i] - y[i];
    z[i+1] = x[i+1] - y[i+1];
    z[i+2] = x[i+2] - y[i+2];
    z[i+3] = x[i+3] - y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] - y[i];
}

static inline void THIntVector_scale(int *y, const int c, const long n)
{
  long i = 0;

  for(; i < n-4; i +=4)
  {
    y[i] *= c;
    y[i+1] *= c;
    y[i+2] *= c;
    y[i+3] *= c;
  }

  for(; i < n; i++)
    y[i] *= c;
}

static inline void THIntVector_mul(int *y, const int *x, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    y[i] *= x[i];
    y[i+1] *= x[i+1];
    y[i+2] *= x[i+2];
    y[i+3] *= x[i+3];
  }

  for(; i < n; i++)
    y[i] *= x[i];
}




static inline void THLongVector_fill(long *x, const long c, const long n) {
  long i = 0;

  for(; i < n-4; i += 4)
  {
    x[i] = c;
    x[i+1] = c;
    x[i+2] = c;
    x[i+3] = c;
  }

  for(; i < n; i++)
    x[i] = c;
}

static inline void THLongVector_add(long *y, const long *x, const long c, const long n)
{
  long i = 0;

  for(;i < n-4; i += 4)
  {
    y[i] += c * x[i];
    y[i+1] += c * x[i+1];
    y[i+2] += c * x[i+2];
    y[i+3] += c * x[i+3];
  }

  for(; i < n; i++)
    y[i] += c * x[i];
}

static inline void THLongVector_diff(long *z, const long *x, const long *y, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    z[i] = x[i] - y[i];
    z[i+1] = x[i+1] - y[i+1];
    z[i+2] = x[i+2] - y[i+2];
    z[i+3] = x[i+3] - y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] - y[i];
}

static inline void THLongVector_scale(long *y, const long c, const long n)
{
  long i = 0;

  for(; i < n-4; i +=4)
  {
    y[i] *= c;
    y[i+1] *= c;
    y[i+2] *= c;
    y[i+3] *= c;
  }

  for(; i < n; i++)
    y[i] *= c;
}

static inline void THLongVector_mul(long *y, const long *x, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    y[i] *= x[i];
    y[i+1] *= x[i+1];
    y[i+2] *= x[i+2];
    y[i+3] *= x[i+3];
  }

  for(; i < n; i++)
    y[i] *= x[i];
}





extern const double THLog2Pi;
extern const double THLogZero;
extern const double THLogOne;

extern double THLogAdd(double log_a, double log_b);
extern double THLogSub(double log_a, double log_b);
extern double THExpMinusApprox(const double x);







typedef struct THGenerator {

  unsigned long the_initial_seed;
  int left;
  int initf;
  unsigned long *next;
  unsigned long state[624];



  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid;
} THGenerator;




extern THGenerator * THGenerator_new();


extern void THGenerator_free(THGenerator *gen);


extern unsigned long THRandom_seed(THGenerator *_generator);


extern void THRandom_manualSeed(THGenerator *_generator, unsigned long the_seed_);


extern unsigned long THRandom_initialSeed(THGenerator *_generator);


extern unsigned long THRandom_random(THGenerator *_generator);


extern double THRandom_uniform(THGenerator *_generator, double a, double b);




extern double THRandom_normal(THGenerator *_generator, double mean, double stdv);





extern double THRandom_exponential(THGenerator *_generator, double lambda);




extern double THRandom_cauchy(THGenerator *_generator, double median, double sigma);





extern double THRandom_logNormal(THGenerator *_generator, double mean, double stdv);





extern int THRandom_geometric(THGenerator *_generator, double p);


extern int THRandom_bernoulli(THGenerator *_generator, double p);


extern void THRandom_getState(THGenerator *_generator, unsigned long *state, long *offset, long *_left);


extern void THRandom_setState(THGenerator *_generator, unsigned long *state, long offset, long _left);











typedef struct THAllocator {
  void* (*malloc)(void*, long);
  void* (*realloc)(void*, void*, long);
  void (*free)(void*, void*);
} THAllocator;




extern THAllocator THDefaultAllocator;



typedef struct THMapAllocatorContext_ THMapAllocatorContext;
THMapAllocatorContext *THMapAllocatorContext_new(const char *filename, int shared);
long THMapAllocatorContext_size(THMapAllocatorContext *ctx);
void THMapAllocatorContext_free(THMapAllocatorContext *ctx);

extern THAllocator THMapAllocator;









typedef struct THByteStorage
{
    unsigned char *__data;
    long __size;
    int __refcount;
    char __flag;
    THAllocator *__allocator;
    void *__allocatorContext;
} THByteStorage;

extern unsigned char* THByteStorage_data(const THByteStorage*);
extern long THByteStorage_size(const THByteStorage*);


extern void THByteStorage_set(THByteStorage*, long, unsigned char);
extern unsigned char THByteStorage_get(const THByteStorage*, long);

extern THByteStorage* THByteStorage_new(void);
extern THByteStorage* THByteStorage_newWithSize(long size);
extern THByteStorage* THByteStorage_newWithSize1(unsigned char);
extern THByteStorage* THByteStorage_newWithSize2(unsigned char, unsigned char);
extern THByteStorage* THByteStorage_newWithSize3(unsigned char, unsigned char, unsigned char);
extern THByteStorage* THByteStorage_newWithSize4(unsigned char, unsigned char, unsigned char, unsigned char);
extern THByteStorage* THByteStorage_newWithMapping(const char *filename, long size, int shared);


extern THByteStorage* THByteStorage_newWithData(unsigned char *data, long size);

extern THByteStorage* THByteStorage_newWithAllocator(long size,
                                               THAllocator* allocator,
                                               void *allocatorContext);
extern THByteStorage* THByteStorage_newWithDataAndAllocator(
    unsigned char* data, long size, THAllocator* allocator, void *allocatorContext);


extern void THByteStorage_setFlag(THByteStorage *storage, const char flag);
extern void THByteStorage_clearFlag(THByteStorage *storage, const char flag);
extern void THByteStorage_retain(THByteStorage *storage);


extern void THByteStorage_free(THByteStorage *storage);
extern void THByteStorage_resize(THByteStorage *storage, long size);
extern void THByteStorage_fill(THByteStorage *storage, unsigned char value);
typedef struct THCharStorage
{
    char *__data;
    long __size;
    int __refcount;
    char __flag;
    THAllocator *__allocator;
    void *__allocatorContext;
} THCharStorage;

extern char* THCharStorage_data(const THCharStorage*);
extern long THCharStorage_size(const THCharStorage*);


extern void THCharStorage_set(THCharStorage*, long, char);
extern char THCharStorage_get(const THCharStorage*, long);

extern THCharStorage* THCharStorage_new(void);
extern THCharStorage* THCharStorage_newWithSize(long size);
extern THCharStorage* THCharStorage_newWithSize1(char);
extern THCharStorage* THCharStorage_newWithSize2(char, char);
extern THCharStorage* THCharStorage_newWithSize3(char, char, char);
extern THCharStorage* THCharStorage_newWithSize4(char, char, char, char);
extern THCharStorage* THCharStorage_newWithMapping(const char *filename, long size, int shared);


extern THCharStorage* THCharStorage_newWithData(char *data, long size);

extern THCharStorage* THCharStorage_newWithAllocator(long size,
                                               THAllocator* allocator,
                                               void *allocatorContext);
extern THCharStorage* THCharStorage_newWithDataAndAllocator(
    char* data, long size, THAllocator* allocator, void *allocatorContext);


extern void THCharStorage_setFlag(THCharStorage *storage, const char flag);
extern void THCharStorage_clearFlag(THCharStorage *storage, const char flag);
extern void THCharStorage_retain(THCharStorage *storage);


extern void THCharStorage_free(THCharStorage *storage);
extern void THCharStorage_resize(THCharStorage *storage, long size);
extern void THCharStorage_fill(THCharStorage *storage, char value);
typedef struct THShortStorage
{
    short *__data;
    long __size;
    int __refcount;
    char __flag;
    THAllocator *__allocator;
    void *__allocatorContext;
} THShortStorage;

extern short* THShortStorage_data(const THShortStorage*);
extern long THShortStorage_size(const THShortStorage*);


extern void THShortStorage_set(THShortStorage*, long, short);
extern short THShortStorage_get(const THShortStorage*, long);

extern THShortStorage* THShortStorage_new(void);
extern THShortStorage* THShortStorage_newWithSize(long size);
extern THShortStorage* THShortStorage_newWithSize1(short);
extern THShortStorage* THShortStorage_newWithSize2(short, short);
extern THShortStorage* THShortStorage_newWithSize3(short, short, short);
extern THShortStorage* THShortStorage_newWithSize4(short, short, short, short);
extern THShortStorage* THShortStorage_newWithMapping(const char *filename, long size, int shared);


extern THShortStorage* THShortStorage_newWithData(short *data, long size);

extern THShortStorage* THShortStorage_newWithAllocator(long size,
                                               THAllocator* allocator,
                                               void *allocatorContext);
extern THShortStorage* THShortStorage_newWithDataAndAllocator(
    short* data, long size, THAllocator* allocator, void *allocatorContext);


extern void THShortStorage_setFlag(THShortStorage *storage, const char flag);
extern void THShortStorage_clearFlag(THShortStorage *storage, const char flag);
extern void THShortStorage_retain(THShortStorage *storage);


extern void THShortStorage_free(THShortStorage *storage);
extern void THShortStorage_resize(THShortStorage *storage, long size);
extern void THShortStorage_fill(THShortStorage *storage, short value);
typedef struct THIntStorage
{
    int *__data;
    long __size;
    int __refcount;
    char __flag;
    THAllocator *__allocator;
    void *__allocatorContext;
} THIntStorage;

extern int* THIntStorage_data(const THIntStorage*);
extern long THIntStorage_size(const THIntStorage*);


extern void THIntStorage_set(THIntStorage*, long, int);
extern int THIntStorage_get(const THIntStorage*, long);

extern THIntStorage* THIntStorage_new(void);
extern THIntStorage* THIntStorage_newWithSize(long size);
extern THIntStorage* THIntStorage_newWithSize1(int);
extern THIntStorage* THIntStorage_newWithSize2(int, int);
extern THIntStorage* THIntStorage_newWithSize3(int, int, int);
extern THIntStorage* THIntStorage_newWithSize4(int, int, int, int);
extern THIntStorage* THIntStorage_newWithMapping(const char *filename, long size, int shared);


extern THIntStorage* THIntStorage_newWithData(int *data, long size);

extern THIntStorage* THIntStorage_newWithAllocator(long size,
                                               THAllocator* allocator,
                                               void *allocatorContext);
extern THIntStorage* THIntStorage_newWithDataAndAllocator(
    int* data, long size, THAllocator* allocator, void *allocatorContext);


extern void THIntStorage_setFlag(THIntStorage *storage, const char flag);
extern void THIntStorage_clearFlag(THIntStorage *storage, const char flag);
extern void THIntStorage_retain(THIntStorage *storage);


extern void THIntStorage_free(THIntStorage *storage);
extern void THIntStorage_resize(THIntStorage *storage, long size);
extern void THIntStorage_fill(THIntStorage *storage, int value);
typedef struct THLongStorage
{
    long *__data;
    long __size;
    int __refcount;
    char __flag;
    THAllocator *__allocator;
    void *__allocatorContext;
} THLongStorage;

extern long* THLongStorage_data(const THLongStorage*);
extern long THLongStorage_size(const THLongStorage*);


extern void THLongStorage_set(THLongStorage*, long, long);
extern long THLongStorage_get(const THLongStorage*, long);

extern THLongStorage* THLongStorage_new(void);
extern THLongStorage* THLongStorage_newWithSize(long size);
extern THLongStorage* THLongStorage_newWithSize1(long);
extern THLongStorage* THLongStorage_newWithSize2(long, long);
extern THLongStorage* THLongStorage_newWithSize3(long, long, long);
extern THLongStorage* THLongStorage_newWithSize4(long, long, long, long);
extern THLongStorage* THLongStorage_newWithMapping(const char *filename, long size, int shared);


extern THLongStorage* THLongStorage_newWithData(long *data, long size);

extern THLongStorage* THLongStorage_newWithAllocator(long size,
                                               THAllocator* allocator,
                                               void *allocatorContext);
extern THLongStorage* THLongStorage_newWithDataAndAllocator(
    long* data, long size, THAllocator* allocator, void *allocatorContext);


extern void THLongStorage_setFlag(THLongStorage *storage, const char flag);
extern void THLongStorage_clearFlag(THLongStorage *storage, const char flag);
extern void THLongStorage_retain(THLongStorage *storage);


extern void THLongStorage_free(THLongStorage *storage);
extern void THLongStorage_resize(THLongStorage *storage, long size);
extern void THLongStorage_fill(THLongStorage *storage, long value);
typedef struct THFloatStorage
{
    float *__data;
    long __size;
    int __refcount;
    char __flag;
    THAllocator *__allocator;
    void *__allocatorContext;
} THFloatStorage;

extern float* THFloatStorage_data(const THFloatStorage*);
extern long THFloatStorage_size(const THFloatStorage*);


extern void THFloatStorage_set(THFloatStorage*, long, float);
extern float THFloatStorage_get(const THFloatStorage*, long);

extern THFloatStorage* THFloatStorage_new(void);
extern THFloatStorage* THFloatStorage_newWithSize(long size);
extern THFloatStorage* THFloatStorage_newWithSize1(float);
extern THFloatStorage* THFloatStorage_newWithSize2(float, float);
extern THFloatStorage* THFloatStorage_newWithSize3(float, float, float);
extern THFloatStorage* THFloatStorage_newWithSize4(float, float, float, float);
extern THFloatStorage* THFloatStorage_newWithMapping(const char *filename, long size, int shared);


extern THFloatStorage* THFloatStorage_newWithData(float *data, long size);

extern THFloatStorage* THFloatStorage_newWithAllocator(long size,
                                               THAllocator* allocator,
                                               void *allocatorContext);
extern THFloatStorage* THFloatStorage_newWithDataAndAllocator(
    float* data, long size, THAllocator* allocator, void *allocatorContext);


extern void THFloatStorage_setFlag(THFloatStorage *storage, const char flag);
extern void THFloatStorage_clearFlag(THFloatStorage *storage, const char flag);
extern void THFloatStorage_retain(THFloatStorage *storage);


extern void THFloatStorage_free(THFloatStorage *storage);
extern void THFloatStorage_resize(THFloatStorage *storage, long size);
extern void THFloatStorage_fill(THFloatStorage *storage, float value);
typedef struct THDoubleStorage
{
    double *__data;
    long __size;
    int __refcount;
    char __flag;
    THAllocator *__allocator;
    void *__allocatorContext;
} THDoubleStorage;

extern double* THDoubleStorage_data(const THDoubleStorage*);
extern long THDoubleStorage_size(const THDoubleStorage*);


extern void THDoubleStorage_set(THDoubleStorage*, long, double);
extern double THDoubleStorage_get(const THDoubleStorage*, long);

extern THDoubleStorage* THDoubleStorage_new(void);
extern THDoubleStorage* THDoubleStorage_newWithSize(long size);
extern THDoubleStorage* THDoubleStorage_newWithSize1(double);
extern THDoubleStorage* THDoubleStorage_newWithSize2(double, double);
extern THDoubleStorage* THDoubleStorage_newWithSize3(double, double, double);
extern THDoubleStorage* THDoubleStorage_newWithSize4(double, double, double, double);
extern THDoubleStorage* THDoubleStorage_newWithMapping(const char *filename, long size, int shared);


extern THDoubleStorage* THDoubleStorage_newWithData(double *data, long size);

extern THDoubleStorage* THDoubleStorage_newWithAllocator(long size,
                                               THAllocator* allocator,
                                               void *allocatorContext);
extern THDoubleStorage* THDoubleStorage_newWithDataAndAllocator(
    double* data, long size, THAllocator* allocator, void *allocatorContext);


extern void THDoubleStorage_setFlag(THDoubleStorage *storage, const char flag);
extern void THDoubleStorage_clearFlag(THDoubleStorage *storage, const char flag);
extern void THDoubleStorage_retain(THDoubleStorage *storage);


extern void THDoubleStorage_free(THDoubleStorage *storage);
extern void THDoubleStorage_resize(THDoubleStorage *storage, long size);
extern void THDoubleStorage_fill(THDoubleStorage *storage, double value);








extern void THByteStorage_rawCopy(THByteStorage *storage, unsigned char *src);
extern void THByteStorage_copy(THByteStorage *storage, THByteStorage *src);
extern void THByteStorage_copyByte(THByteStorage *storage, struct THByteStorage *src);
extern void THByteStorage_copyChar(THByteStorage *storage, struct THCharStorage *src);
extern void THByteStorage_copyShort(THByteStorage *storage, struct THShortStorage *src);
extern void THByteStorage_copyInt(THByteStorage *storage, struct THIntStorage *src);
extern void THByteStorage_copyLong(THByteStorage *storage, struct THLongStorage *src);
extern void THByteStorage_copyFloat(THByteStorage *storage, struct THFloatStorage *src);
extern void THByteStorage_copyDouble(THByteStorage *storage, struct THDoubleStorage *src);






extern void THCharStorage_rawCopy(THCharStorage *storage, char *src);
extern void THCharStorage_copy(THCharStorage *storage, THCharStorage *src);
extern void THCharStorage_copyByte(THCharStorage *storage, struct THByteStorage *src);
extern void THCharStorage_copyChar(THCharStorage *storage, struct THCharStorage *src);
extern void THCharStorage_copyShort(THCharStorage *storage, struct THShortStorage *src);
extern void THCharStorage_copyInt(THCharStorage *storage, struct THIntStorage *src);
extern void THCharStorage_copyLong(THCharStorage *storage, struct THLongStorage *src);
extern void THCharStorage_copyFloat(THCharStorage *storage, struct THFloatStorage *src);
extern void THCharStorage_copyDouble(THCharStorage *storage, struct THDoubleStorage *src);






extern void THShortStorage_rawCopy(THShortStorage *storage, short *src);
extern void THShortStorage_copy(THShortStorage *storage, THShortStorage *src);
extern void THShortStorage_copyByte(THShortStorage *storage, struct THByteStorage *src);
extern void THShortStorage_copyChar(THShortStorage *storage, struct THCharStorage *src);
extern void THShortStorage_copyShort(THShortStorage *storage, struct THShortStorage *src);
extern void THShortStorage_copyInt(THShortStorage *storage, struct THIntStorage *src);
extern void THShortStorage_copyLong(THShortStorage *storage, struct THLongStorage *src);
extern void THShortStorage_copyFloat(THShortStorage *storage, struct THFloatStorage *src);
extern void THShortStorage_copyDouble(THShortStorage *storage, struct THDoubleStorage *src);






extern void THIntStorage_rawCopy(THIntStorage *storage, int *src);
extern void THIntStorage_copy(THIntStorage *storage, THIntStorage *src);
extern void THIntStorage_copyByte(THIntStorage *storage, struct THByteStorage *src);
extern void THIntStorage_copyChar(THIntStorage *storage, struct THCharStorage *src);
extern void THIntStorage_copyShort(THIntStorage *storage, struct THShortStorage *src);
extern void THIntStorage_copyInt(THIntStorage *storage, struct THIntStorage *src);
extern void THIntStorage_copyLong(THIntStorage *storage, struct THLongStorage *src);
extern void THIntStorage_copyFloat(THIntStorage *storage, struct THFloatStorage *src);
extern void THIntStorage_copyDouble(THIntStorage *storage, struct THDoubleStorage *src);






extern void THLongStorage_rawCopy(THLongStorage *storage, long *src);
extern void THLongStorage_copy(THLongStorage *storage, THLongStorage *src);
extern void THLongStorage_copyByte(THLongStorage *storage, struct THByteStorage *src);
extern void THLongStorage_copyChar(THLongStorage *storage, struct THCharStorage *src);
extern void THLongStorage_copyShort(THLongStorage *storage, struct THShortStorage *src);
extern void THLongStorage_copyInt(THLongStorage *storage, struct THIntStorage *src);
extern void THLongStorage_copyLong(THLongStorage *storage, struct THLongStorage *src);
extern void THLongStorage_copyFloat(THLongStorage *storage, struct THFloatStorage *src);
extern void THLongStorage_copyDouble(THLongStorage *storage, struct THDoubleStorage *src);






extern void THFloatStorage_rawCopy(THFloatStorage *storage, float *src);
extern void THFloatStorage_copy(THFloatStorage *storage, THFloatStorage *src);
extern void THFloatStorage_copyByte(THFloatStorage *storage, struct THByteStorage *src);
extern void THFloatStorage_copyChar(THFloatStorage *storage, struct THCharStorage *src);
extern void THFloatStorage_copyShort(THFloatStorage *storage, struct THShortStorage *src);
extern void THFloatStorage_copyInt(THFloatStorage *storage, struct THIntStorage *src);
extern void THFloatStorage_copyLong(THFloatStorage *storage, struct THLongStorage *src);
extern void THFloatStorage_copyFloat(THFloatStorage *storage, struct THFloatStorage *src);
extern void THFloatStorage_copyDouble(THFloatStorage *storage, struct THDoubleStorage *src);






extern void THDoubleStorage_rawCopy(THDoubleStorage *storage, double *src);
extern void THDoubleStorage_copy(THDoubleStorage *storage, THDoubleStorage *src);
extern void THDoubleStorage_copyByte(THDoubleStorage *storage, struct THByteStorage *src);
extern void THDoubleStorage_copyChar(THDoubleStorage *storage, struct THCharStorage *src);
extern void THDoubleStorage_copyShort(THDoubleStorage *storage, struct THShortStorage *src);
extern void THDoubleStorage_copyInt(THDoubleStorage *storage, struct THIntStorage *src);
extern void THDoubleStorage_copyLong(THDoubleStorage *storage, struct THLongStorage *src);
extern void THDoubleStorage_copyFloat(THDoubleStorage *storage, struct THFloatStorage *src);
extern void THDoubleStorage_copyDouble(THDoubleStorage *storage, struct THDoubleStorage *src);


















typedef struct THByteTensor
{
    long *__size;
    long *__stride;
    int __nDimension;

    THByteStorage *__storage;
    long __storageOffset;
    int __refcount;

    char __flag;

} THByteTensor;



extern THByteStorage* THByteTensor_storage(const THByteTensor *self);
extern long THByteTensor_storageOffset(const THByteTensor *self);
extern int THByteTensor_nDimension(const THByteTensor *self);
extern long THByteTensor_size(const THByteTensor *self, int dim);
extern long THByteTensor_stride(const THByteTensor *self, int dim);
extern THLongStorage *THByteTensor_newSizeOf(THByteTensor *self);
extern THLongStorage *THByteTensor_newStrideOf(THByteTensor *self);
extern unsigned char *THByteTensor_data(const THByteTensor *self);

extern void THByteTensor_setFlag(THByteTensor *self, const char flag);
extern void THByteTensor_clearFlag(THByteTensor *self, const char flag);



extern THByteTensor *THByteTensor_new(void);
extern THByteTensor *THByteTensor_newWithTensor(THByteTensor *tensor);

extern THByteTensor *THByteTensor_newWithStorage(THByteStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern THByteTensor *THByteTensor_newWithStorage1d(THByteStorage *storage_, long storageOffset_,
                                long size0_, long stride0_);
extern THByteTensor *THByteTensor_newWithStorage2d(THByteStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
extern THByteTensor *THByteTensor_newWithStorage3d(THByteStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
extern THByteTensor *THByteTensor_newWithStorage4d(THByteStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);


extern THByteTensor *THByteTensor_newWithSize(THLongStorage *size_, THLongStorage *stride_);
extern THByteTensor *THByteTensor_newWithSize1d(long size0_);
extern THByteTensor *THByteTensor_newWithSize2d(long size0_, long size1_);
extern THByteTensor *THByteTensor_newWithSize3d(long size0_, long size1_, long size2_);
extern THByteTensor *THByteTensor_newWithSize4d(long size0_, long size1_, long size2_, long size3_);

extern THByteTensor *THByteTensor_newClone(THByteTensor *self);
extern THByteTensor *THByteTensor_newContiguous(THByteTensor *tensor);
extern THByteTensor *THByteTensor_newSelect(THByteTensor *tensor, int dimension_, long sliceIndex_);
extern THByteTensor *THByteTensor_newNarrow(THByteTensor *tensor, int dimension_, long firstIndex_, long size_);
extern THByteTensor *THByteTensor_newTranspose(THByteTensor *tensor, int dimension1_, int dimension2_);
extern THByteTensor *THByteTensor_newUnfold(THByteTensor *tensor, int dimension_, long size_, long step_);

extern void THByteTensor_resize(THByteTensor *tensor, THLongStorage *size, THLongStorage *stride);
extern void THByteTensor_resizeAs(THByteTensor *tensor, THByteTensor *src);
extern void THByteTensor_resize1d(THByteTensor *tensor, long size0_);
extern void THByteTensor_resize2d(THByteTensor *tensor, long size0_, long size1_);
extern void THByteTensor_resize3d(THByteTensor *tensor, long size0_, long size1_, long size2_);
extern void THByteTensor_resize4d(THByteTensor *tensor, long size0_, long size1_, long size2_, long size3_);
extern void THByteTensor_resize5d(THByteTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

extern void THByteTensor_set(THByteTensor *self, THByteTensor *src);
extern void THByteTensor_setStorage(THByteTensor *self, THByteStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern void THByteTensor_setStorage1d(THByteTensor *self, THByteStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_);
extern void THByteTensor_setStorage2d(THByteTensor *self, THByteStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_);
extern void THByteTensor_setStorage3d(THByteTensor *self, THByteStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_);
extern void THByteTensor_setStorage4d(THByteTensor *self, THByteStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_,
                                    long size3_, long stride3_);

extern void THByteTensor_narrow(THByteTensor *self, THByteTensor *src, int dimension_, long firstIndex_, long size_);
extern void THByteTensor_select(THByteTensor *self, THByteTensor *src, int dimension_, long sliceIndex_);
extern void THByteTensor_transpose(THByteTensor *self, THByteTensor *src, int dimension1_, int dimension2_);
extern void THByteTensor_unfold(THByteTensor *self, THByteTensor *src, int dimension_, long size_, long step_);

extern void THByteTensor_squeeze(THByteTensor *self, THByteTensor *src);
extern void THByteTensor_squeeze1d(THByteTensor *self, THByteTensor *src, int dimension_);

extern int THByteTensor_isContiguous(const THByteTensor *self);
extern long THByteTensor_nElement(const THByteTensor *self);

extern void THByteTensor_retain(THByteTensor *self);
extern void THByteTensor_free(THByteTensor *self);
extern void THByteTensor_freeCopyTo(THByteTensor *self, THByteTensor *dst);


extern void THByteTensor_set1d(THByteTensor *tensor, long x0, unsigned char value);
extern void THByteTensor_set2d(THByteTensor *tensor, long x0, long x1, unsigned char value);
extern void THByteTensor_set3d(THByteTensor *tensor, long x0, long x1, long x2, unsigned char value);
extern void THByteTensor_set4d(THByteTensor *tensor, long x0, long x1, long x2, long x3, unsigned char value);

extern unsigned char THByteTensor_get1d(const THByteTensor *tensor, long x0);
extern unsigned char THByteTensor_get2d(const THByteTensor *tensor, long x0, long x1);
extern unsigned char THByteTensor_get3d(const THByteTensor *tensor, long x0, long x1, long x2);
extern unsigned char THByteTensor_get4d(const THByteTensor *tensor, long x0, long x1, long x2, long x3);








typedef struct THCharTensor
{
    long *__size;
    long *__stride;
    int __nDimension;

    THCharStorage *__storage;
    long __storageOffset;
    int __refcount;

    char __flag;

} THCharTensor;



extern THCharStorage* THCharTensor_storage(const THCharTensor *self);
extern long THCharTensor_storageOffset(const THCharTensor *self);
extern int THCharTensor_nDimension(const THCharTensor *self);
extern long THCharTensor_size(const THCharTensor *self, int dim);
extern long THCharTensor_stride(const THCharTensor *self, int dim);
extern THLongStorage *THCharTensor_newSizeOf(THCharTensor *self);
extern THLongStorage *THCharTensor_newStrideOf(THCharTensor *self);
extern char *THCharTensor_data(const THCharTensor *self);

extern void THCharTensor_setFlag(THCharTensor *self, const char flag);
extern void THCharTensor_clearFlag(THCharTensor *self, const char flag);



extern THCharTensor *THCharTensor_new(void);
extern THCharTensor *THCharTensor_newWithTensor(THCharTensor *tensor);

extern THCharTensor *THCharTensor_newWithStorage(THCharStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern THCharTensor *THCharTensor_newWithStorage1d(THCharStorage *storage_, long storageOffset_,
                                long size0_, long stride0_);
extern THCharTensor *THCharTensor_newWithStorage2d(THCharStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
extern THCharTensor *THCharTensor_newWithStorage3d(THCharStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
extern THCharTensor *THCharTensor_newWithStorage4d(THCharStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);


extern THCharTensor *THCharTensor_newWithSize(THLongStorage *size_, THLongStorage *stride_);
extern THCharTensor *THCharTensor_newWithSize1d(long size0_);
extern THCharTensor *THCharTensor_newWithSize2d(long size0_, long size1_);
extern THCharTensor *THCharTensor_newWithSize3d(long size0_, long size1_, long size2_);
extern THCharTensor *THCharTensor_newWithSize4d(long size0_, long size1_, long size2_, long size3_);

extern THCharTensor *THCharTensor_newClone(THCharTensor *self);
extern THCharTensor *THCharTensor_newContiguous(THCharTensor *tensor);
extern THCharTensor *THCharTensor_newSelect(THCharTensor *tensor, int dimension_, long sliceIndex_);
extern THCharTensor *THCharTensor_newNarrow(THCharTensor *tensor, int dimension_, long firstIndex_, long size_);
extern THCharTensor *THCharTensor_newTranspose(THCharTensor *tensor, int dimension1_, int dimension2_);
extern THCharTensor *THCharTensor_newUnfold(THCharTensor *tensor, int dimension_, long size_, long step_);

extern void THCharTensor_resize(THCharTensor *tensor, THLongStorage *size, THLongStorage *stride);
extern void THCharTensor_resizeAs(THCharTensor *tensor, THCharTensor *src);
extern void THCharTensor_resize1d(THCharTensor *tensor, long size0_);
extern void THCharTensor_resize2d(THCharTensor *tensor, long size0_, long size1_);
extern void THCharTensor_resize3d(THCharTensor *tensor, long size0_, long size1_, long size2_);
extern void THCharTensor_resize4d(THCharTensor *tensor, long size0_, long size1_, long size2_, long size3_);
extern void THCharTensor_resize5d(THCharTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

extern void THCharTensor_set(THCharTensor *self, THCharTensor *src);
extern void THCharTensor_setStorage(THCharTensor *self, THCharStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern void THCharTensor_setStorage1d(THCharTensor *self, THCharStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_);
extern void THCharTensor_setStorage2d(THCharTensor *self, THCharStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_);
extern void THCharTensor_setStorage3d(THCharTensor *self, THCharStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_);
extern void THCharTensor_setStorage4d(THCharTensor *self, THCharStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_,
                                    long size3_, long stride3_);

extern void THCharTensor_narrow(THCharTensor *self, THCharTensor *src, int dimension_, long firstIndex_, long size_);
extern void THCharTensor_select(THCharTensor *self, THCharTensor *src, int dimension_, long sliceIndex_);
extern void THCharTensor_transpose(THCharTensor *self, THCharTensor *src, int dimension1_, int dimension2_);
extern void THCharTensor_unfold(THCharTensor *self, THCharTensor *src, int dimension_, long size_, long step_);

extern void THCharTensor_squeeze(THCharTensor *self, THCharTensor *src);
extern void THCharTensor_squeeze1d(THCharTensor *self, THCharTensor *src, int dimension_);

extern int THCharTensor_isContiguous(const THCharTensor *self);
extern long THCharTensor_nElement(const THCharTensor *self);

extern void THCharTensor_retain(THCharTensor *self);
extern void THCharTensor_free(THCharTensor *self);
extern void THCharTensor_freeCopyTo(THCharTensor *self, THCharTensor *dst);


extern void THCharTensor_set1d(THCharTensor *tensor, long x0, char value);
extern void THCharTensor_set2d(THCharTensor *tensor, long x0, long x1, char value);
extern void THCharTensor_set3d(THCharTensor *tensor, long x0, long x1, long x2, char value);
extern void THCharTensor_set4d(THCharTensor *tensor, long x0, long x1, long x2, long x3, char value);

extern char THCharTensor_get1d(const THCharTensor *tensor, long x0);
extern char THCharTensor_get2d(const THCharTensor *tensor, long x0, long x1);
extern char THCharTensor_get3d(const THCharTensor *tensor, long x0, long x1, long x2);
extern char THCharTensor_get4d(const THCharTensor *tensor, long x0, long x1, long x2, long x3);








typedef struct THShortTensor
{
    long *__size;
    long *__stride;
    int __nDimension;

    THShortStorage *__storage;
    long __storageOffset;
    int __refcount;

    char __flag;

} THShortTensor;



extern THShortStorage* THShortTensor_storage(const THShortTensor *self);
extern long THShortTensor_storageOffset(const THShortTensor *self);
extern int THShortTensor_nDimension(const THShortTensor *self);
extern long THShortTensor_size(const THShortTensor *self, int dim);
extern long THShortTensor_stride(const THShortTensor *self, int dim);
extern THLongStorage *THShortTensor_newSizeOf(THShortTensor *self);
extern THLongStorage *THShortTensor_newStrideOf(THShortTensor *self);
extern short *THShortTensor_data(const THShortTensor *self);

extern void THShortTensor_setFlag(THShortTensor *self, const char flag);
extern void THShortTensor_clearFlag(THShortTensor *self, const char flag);



extern THShortTensor *THShortTensor_new(void);
extern THShortTensor *THShortTensor_newWithTensor(THShortTensor *tensor);

extern THShortTensor *THShortTensor_newWithStorage(THShortStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern THShortTensor *THShortTensor_newWithStorage1d(THShortStorage *storage_, long storageOffset_,
                                long size0_, long stride0_);
extern THShortTensor *THShortTensor_newWithStorage2d(THShortStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
extern THShortTensor *THShortTensor_newWithStorage3d(THShortStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
extern THShortTensor *THShortTensor_newWithStorage4d(THShortStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);


extern THShortTensor *THShortTensor_newWithSize(THLongStorage *size_, THLongStorage *stride_);
extern THShortTensor *THShortTensor_newWithSize1d(long size0_);
extern THShortTensor *THShortTensor_newWithSize2d(long size0_, long size1_);
extern THShortTensor *THShortTensor_newWithSize3d(long size0_, long size1_, long size2_);
extern THShortTensor *THShortTensor_newWithSize4d(long size0_, long size1_, long size2_, long size3_);

extern THShortTensor *THShortTensor_newClone(THShortTensor *self);
extern THShortTensor *THShortTensor_newContiguous(THShortTensor *tensor);
extern THShortTensor *THShortTensor_newSelect(THShortTensor *tensor, int dimension_, long sliceIndex_);
extern THShortTensor *THShortTensor_newNarrow(THShortTensor *tensor, int dimension_, long firstIndex_, long size_);
extern THShortTensor *THShortTensor_newTranspose(THShortTensor *tensor, int dimension1_, int dimension2_);
extern THShortTensor *THShortTensor_newUnfold(THShortTensor *tensor, int dimension_, long size_, long step_);

extern void THShortTensor_resize(THShortTensor *tensor, THLongStorage *size, THLongStorage *stride);
extern void THShortTensor_resizeAs(THShortTensor *tensor, THShortTensor *src);
extern void THShortTensor_resize1d(THShortTensor *tensor, long size0_);
extern void THShortTensor_resize2d(THShortTensor *tensor, long size0_, long size1_);
extern void THShortTensor_resize3d(THShortTensor *tensor, long size0_, long size1_, long size2_);
extern void THShortTensor_resize4d(THShortTensor *tensor, long size0_, long size1_, long size2_, long size3_);
extern void THShortTensor_resize5d(THShortTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

extern void THShortTensor_set(THShortTensor *self, THShortTensor *src);
extern void THShortTensor_setStorage(THShortTensor *self, THShortStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern void THShortTensor_setStorage1d(THShortTensor *self, THShortStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_);
extern void THShortTensor_setStorage2d(THShortTensor *self, THShortStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_);
extern void THShortTensor_setStorage3d(THShortTensor *self, THShortStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_);
extern void THShortTensor_setStorage4d(THShortTensor *self, THShortStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_,
                                    long size3_, long stride3_);

extern void THShortTensor_narrow(THShortTensor *self, THShortTensor *src, int dimension_, long firstIndex_, long size_);
extern void THShortTensor_select(THShortTensor *self, THShortTensor *src, int dimension_, long sliceIndex_);
extern void THShortTensor_transpose(THShortTensor *self, THShortTensor *src, int dimension1_, int dimension2_);
extern void THShortTensor_unfold(THShortTensor *self, THShortTensor *src, int dimension_, long size_, long step_);

extern void THShortTensor_squeeze(THShortTensor *self, THShortTensor *src);
extern void THShortTensor_squeeze1d(THShortTensor *self, THShortTensor *src, int dimension_);

extern int THShortTensor_isContiguous(const THShortTensor *self);
extern long THShortTensor_nElement(const THShortTensor *self);

extern void THShortTensor_retain(THShortTensor *self);
extern void THShortTensor_free(THShortTensor *self);
extern void THShortTensor_freeCopyTo(THShortTensor *self, THShortTensor *dst);


extern void THShortTensor_set1d(THShortTensor *tensor, long x0, short value);
extern void THShortTensor_set2d(THShortTensor *tensor, long x0, long x1, short value);
extern void THShortTensor_set3d(THShortTensor *tensor, long x0, long x1, long x2, short value);
extern void THShortTensor_set4d(THShortTensor *tensor, long x0, long x1, long x2, long x3, short value);

extern short THShortTensor_get1d(const THShortTensor *tensor, long x0);
extern short THShortTensor_get2d(const THShortTensor *tensor, long x0, long x1);
extern short THShortTensor_get3d(const THShortTensor *tensor, long x0, long x1, long x2);
extern short THShortTensor_get4d(const THShortTensor *tensor, long x0, long x1, long x2, long x3);








typedef struct THIntTensor
{
    long *__size;
    long *__stride;
    int __nDimension;

    THIntStorage *__storage;
    long __storageOffset;
    int __refcount;

    char __flag;

} THIntTensor;



extern THIntStorage* THIntTensor_storage(const THIntTensor *self);
extern long THIntTensor_storageOffset(const THIntTensor *self);
extern int THIntTensor_nDimension(const THIntTensor *self);
extern long THIntTensor_size(const THIntTensor *self, int dim);
extern long THIntTensor_stride(const THIntTensor *self, int dim);
extern THLongStorage *THIntTensor_newSizeOf(THIntTensor *self);
extern THLongStorage *THIntTensor_newStrideOf(THIntTensor *self);
extern int *THIntTensor_data(const THIntTensor *self);

extern void THIntTensor_setFlag(THIntTensor *self, const char flag);
extern void THIntTensor_clearFlag(THIntTensor *self, const char flag);



extern THIntTensor *THIntTensor_new(void);
extern THIntTensor *THIntTensor_newWithTensor(THIntTensor *tensor);

extern THIntTensor *THIntTensor_newWithStorage(THIntStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern THIntTensor *THIntTensor_newWithStorage1d(THIntStorage *storage_, long storageOffset_,
                                long size0_, long stride0_);
extern THIntTensor *THIntTensor_newWithStorage2d(THIntStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
extern THIntTensor *THIntTensor_newWithStorage3d(THIntStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
extern THIntTensor *THIntTensor_newWithStorage4d(THIntStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);


extern THIntTensor *THIntTensor_newWithSize(THLongStorage *size_, THLongStorage *stride_);
extern THIntTensor *THIntTensor_newWithSize1d(long size0_);
extern THIntTensor *THIntTensor_newWithSize2d(long size0_, long size1_);
extern THIntTensor *THIntTensor_newWithSize3d(long size0_, long size1_, long size2_);
extern THIntTensor *THIntTensor_newWithSize4d(long size0_, long size1_, long size2_, long size3_);

extern THIntTensor *THIntTensor_newClone(THIntTensor *self);
extern THIntTensor *THIntTensor_newContiguous(THIntTensor *tensor);
extern THIntTensor *THIntTensor_newSelect(THIntTensor *tensor, int dimension_, long sliceIndex_);
extern THIntTensor *THIntTensor_newNarrow(THIntTensor *tensor, int dimension_, long firstIndex_, long size_);
extern THIntTensor *THIntTensor_newTranspose(THIntTensor *tensor, int dimension1_, int dimension2_);
extern THIntTensor *THIntTensor_newUnfold(THIntTensor *tensor, int dimension_, long size_, long step_);

extern void THIntTensor_resize(THIntTensor *tensor, THLongStorage *size, THLongStorage *stride);
extern void THIntTensor_resizeAs(THIntTensor *tensor, THIntTensor *src);
extern void THIntTensor_resize1d(THIntTensor *tensor, long size0_);
extern void THIntTensor_resize2d(THIntTensor *tensor, long size0_, long size1_);
extern void THIntTensor_resize3d(THIntTensor *tensor, long size0_, long size1_, long size2_);
extern void THIntTensor_resize4d(THIntTensor *tensor, long size0_, long size1_, long size2_, long size3_);
extern void THIntTensor_resize5d(THIntTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

extern void THIntTensor_set(THIntTensor *self, THIntTensor *src);
extern void THIntTensor_setStorage(THIntTensor *self, THIntStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern void THIntTensor_setStorage1d(THIntTensor *self, THIntStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_);
extern void THIntTensor_setStorage2d(THIntTensor *self, THIntStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_);
extern void THIntTensor_setStorage3d(THIntTensor *self, THIntStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_);
extern void THIntTensor_setStorage4d(THIntTensor *self, THIntStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_,
                                    long size3_, long stride3_);

extern void THIntTensor_narrow(THIntTensor *self, THIntTensor *src, int dimension_, long firstIndex_, long size_);
extern void THIntTensor_select(THIntTensor *self, THIntTensor *src, int dimension_, long sliceIndex_);
extern void THIntTensor_transpose(THIntTensor *self, THIntTensor *src, int dimension1_, int dimension2_);
extern void THIntTensor_unfold(THIntTensor *self, THIntTensor *src, int dimension_, long size_, long step_);

extern void THIntTensor_squeeze(THIntTensor *self, THIntTensor *src);
extern void THIntTensor_squeeze1d(THIntTensor *self, THIntTensor *src, int dimension_);

extern int THIntTensor_isContiguous(const THIntTensor *self);
extern long THIntTensor_nElement(const THIntTensor *self);

extern void THIntTensor_retain(THIntTensor *self);
extern void THIntTensor_free(THIntTensor *self);
extern void THIntTensor_freeCopyTo(THIntTensor *self, THIntTensor *dst);


extern void THIntTensor_set1d(THIntTensor *tensor, long x0, int value);
extern void THIntTensor_set2d(THIntTensor *tensor, long x0, long x1, int value);
extern void THIntTensor_set3d(THIntTensor *tensor, long x0, long x1, long x2, int value);
extern void THIntTensor_set4d(THIntTensor *tensor, long x0, long x1, long x2, long x3, int value);

extern int THIntTensor_get1d(const THIntTensor *tensor, long x0);
extern int THIntTensor_get2d(const THIntTensor *tensor, long x0, long x1);
extern int THIntTensor_get3d(const THIntTensor *tensor, long x0, long x1, long x2);
extern int THIntTensor_get4d(const THIntTensor *tensor, long x0, long x1, long x2, long x3);








typedef struct THLongTensor
{
    long *__size;
    long *__stride;
    int __nDimension;

    THLongStorage *__storage;
    long __storageOffset;
    int __refcount;

    char __flag;

} THLongTensor;



extern THLongStorage* THLongTensor_storage(const THLongTensor *self);
extern long THLongTensor_storageOffset(const THLongTensor *self);
extern int THLongTensor_nDimension(const THLongTensor *self);
extern long THLongTensor_size(const THLongTensor *self, int dim);
extern long THLongTensor_stride(const THLongTensor *self, int dim);
extern THLongStorage *THLongTensor_newSizeOf(THLongTensor *self);
extern THLongStorage *THLongTensor_newStrideOf(THLongTensor *self);
extern long *THLongTensor_data(const THLongTensor *self);

extern void THLongTensor_setFlag(THLongTensor *self, const char flag);
extern void THLongTensor_clearFlag(THLongTensor *self, const char flag);



extern THLongTensor *THLongTensor_new(void);
extern THLongTensor *THLongTensor_newWithTensor(THLongTensor *tensor);

extern THLongTensor *THLongTensor_newWithStorage(THLongStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern THLongTensor *THLongTensor_newWithStorage1d(THLongStorage *storage_, long storageOffset_,
                                long size0_, long stride0_);
extern THLongTensor *THLongTensor_newWithStorage2d(THLongStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
extern THLongTensor *THLongTensor_newWithStorage3d(THLongStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
extern THLongTensor *THLongTensor_newWithStorage4d(THLongStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);


extern THLongTensor *THLongTensor_newWithSize(THLongStorage *size_, THLongStorage *stride_);
extern THLongTensor *THLongTensor_newWithSize1d(long size0_);
extern THLongTensor *THLongTensor_newWithSize2d(long size0_, long size1_);
extern THLongTensor *THLongTensor_newWithSize3d(long size0_, long size1_, long size2_);
extern THLongTensor *THLongTensor_newWithSize4d(long size0_, long size1_, long size2_, long size3_);

extern THLongTensor *THLongTensor_newClone(THLongTensor *self);
extern THLongTensor *THLongTensor_newContiguous(THLongTensor *tensor);
extern THLongTensor *THLongTensor_newSelect(THLongTensor *tensor, int dimension_, long sliceIndex_);
extern THLongTensor *THLongTensor_newNarrow(THLongTensor *tensor, int dimension_, long firstIndex_, long size_);
extern THLongTensor *THLongTensor_newTranspose(THLongTensor *tensor, int dimension1_, int dimension2_);
extern THLongTensor *THLongTensor_newUnfold(THLongTensor *tensor, int dimension_, long size_, long step_);

extern void THLongTensor_resize(THLongTensor *tensor, THLongStorage *size, THLongStorage *stride);
extern void THLongTensor_resizeAs(THLongTensor *tensor, THLongTensor *src);
extern void THLongTensor_resize1d(THLongTensor *tensor, long size0_);
extern void THLongTensor_resize2d(THLongTensor *tensor, long size0_, long size1_);
extern void THLongTensor_resize3d(THLongTensor *tensor, long size0_, long size1_, long size2_);
extern void THLongTensor_resize4d(THLongTensor *tensor, long size0_, long size1_, long size2_, long size3_);
extern void THLongTensor_resize5d(THLongTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

extern void THLongTensor_set(THLongTensor *self, THLongTensor *src);
extern void THLongTensor_setStorage(THLongTensor *self, THLongStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern void THLongTensor_setStorage1d(THLongTensor *self, THLongStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_);
extern void THLongTensor_setStorage2d(THLongTensor *self, THLongStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_);
extern void THLongTensor_setStorage3d(THLongTensor *self, THLongStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_);
extern void THLongTensor_setStorage4d(THLongTensor *self, THLongStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_,
                                    long size3_, long stride3_);

extern void THLongTensor_narrow(THLongTensor *self, THLongTensor *src, int dimension_, long firstIndex_, long size_);
extern void THLongTensor_select(THLongTensor *self, THLongTensor *src, int dimension_, long sliceIndex_);
extern void THLongTensor_transpose(THLongTensor *self, THLongTensor *src, int dimension1_, int dimension2_);
extern void THLongTensor_unfold(THLongTensor *self, THLongTensor *src, int dimension_, long size_, long step_);

extern void THLongTensor_squeeze(THLongTensor *self, THLongTensor *src);
extern void THLongTensor_squeeze1d(THLongTensor *self, THLongTensor *src, int dimension_);

extern int THLongTensor_isContiguous(const THLongTensor *self);
extern long THLongTensor_nElement(const THLongTensor *self);

extern void THLongTensor_retain(THLongTensor *self);
extern void THLongTensor_free(THLongTensor *self);
extern void THLongTensor_freeCopyTo(THLongTensor *self, THLongTensor *dst);


extern void THLongTensor_set1d(THLongTensor *tensor, long x0, long value);
extern void THLongTensor_set2d(THLongTensor *tensor, long x0, long x1, long value);
extern void THLongTensor_set3d(THLongTensor *tensor, long x0, long x1, long x2, long value);
extern void THLongTensor_set4d(THLongTensor *tensor, long x0, long x1, long x2, long x3, long value);

extern long THLongTensor_get1d(const THLongTensor *tensor, long x0);
extern long THLongTensor_get2d(const THLongTensor *tensor, long x0, long x1);
extern long THLongTensor_get3d(const THLongTensor *tensor, long x0, long x1, long x2);
extern long THLongTensor_get4d(const THLongTensor *tensor, long x0, long x1, long x2, long x3);








typedef struct THFloatTensor
{
    long *__size;
    long *__stride;
    int __nDimension;

    THFloatStorage *__storage;
    long __storageOffset;
    int __refcount;

    char __flag;

} THFloatTensor;



extern THFloatStorage* THFloatTensor_storage(const THFloatTensor *self);
extern long THFloatTensor_storageOffset(const THFloatTensor *self);
extern int THFloatTensor_nDimension(const THFloatTensor *self);
extern long THFloatTensor_size(const THFloatTensor *self, int dim);
extern long THFloatTensor_stride(const THFloatTensor *self, int dim);
extern THLongStorage *THFloatTensor_newSizeOf(THFloatTensor *self);
extern THLongStorage *THFloatTensor_newStrideOf(THFloatTensor *self);
extern float *THFloatTensor_data(const THFloatTensor *self);

extern void THFloatTensor_setFlag(THFloatTensor *self, const char flag);
extern void THFloatTensor_clearFlag(THFloatTensor *self, const char flag);



extern THFloatTensor *THFloatTensor_new(void);
extern THFloatTensor *THFloatTensor_newWithTensor(THFloatTensor *tensor);

extern THFloatTensor *THFloatTensor_newWithStorage(THFloatStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern THFloatTensor *THFloatTensor_newWithStorage1d(THFloatStorage *storage_, long storageOffset_,
                                long size0_, long stride0_);
extern THFloatTensor *THFloatTensor_newWithStorage2d(THFloatStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
extern THFloatTensor *THFloatTensor_newWithStorage3d(THFloatStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
extern THFloatTensor *THFloatTensor_newWithStorage4d(THFloatStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);


extern THFloatTensor *THFloatTensor_newWithSize(THLongStorage *size_, THLongStorage *stride_);
extern THFloatTensor *THFloatTensor_newWithSize1d(long size0_);
extern THFloatTensor *THFloatTensor_newWithSize2d(long size0_, long size1_);
extern THFloatTensor *THFloatTensor_newWithSize3d(long size0_, long size1_, long size2_);
extern THFloatTensor *THFloatTensor_newWithSize4d(long size0_, long size1_, long size2_, long size3_);

extern THFloatTensor *THFloatTensor_newClone(THFloatTensor *self);
extern THFloatTensor *THFloatTensor_newContiguous(THFloatTensor *tensor);
extern THFloatTensor *THFloatTensor_newSelect(THFloatTensor *tensor, int dimension_, long sliceIndex_);
extern THFloatTensor *THFloatTensor_newNarrow(THFloatTensor *tensor, int dimension_, long firstIndex_, long size_);
extern THFloatTensor *THFloatTensor_newTranspose(THFloatTensor *tensor, int dimension1_, int dimension2_);
extern THFloatTensor *THFloatTensor_newUnfold(THFloatTensor *tensor, int dimension_, long size_, long step_);

extern void THFloatTensor_resize(THFloatTensor *tensor, THLongStorage *size, THLongStorage *stride);
extern void THFloatTensor_resizeAs(THFloatTensor *tensor, THFloatTensor *src);
extern void THFloatTensor_resize1d(THFloatTensor *tensor, long size0_);
extern void THFloatTensor_resize2d(THFloatTensor *tensor, long size0_, long size1_);
extern void THFloatTensor_resize3d(THFloatTensor *tensor, long size0_, long size1_, long size2_);
extern void THFloatTensor_resize4d(THFloatTensor *tensor, long size0_, long size1_, long size2_, long size3_);
extern void THFloatTensor_resize5d(THFloatTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

extern void THFloatTensor_set(THFloatTensor *self, THFloatTensor *src);
extern void THFloatTensor_setStorage(THFloatTensor *self, THFloatStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern void THFloatTensor_setStorage1d(THFloatTensor *self, THFloatStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_);
extern void THFloatTensor_setStorage2d(THFloatTensor *self, THFloatStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_);
extern void THFloatTensor_setStorage3d(THFloatTensor *self, THFloatStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_);
extern void THFloatTensor_setStorage4d(THFloatTensor *self, THFloatStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_,
                                    long size3_, long stride3_);

extern void THFloatTensor_narrow(THFloatTensor *self, THFloatTensor *src, int dimension_, long firstIndex_, long size_);
extern void THFloatTensor_select(THFloatTensor *self, THFloatTensor *src, int dimension_, long sliceIndex_);
extern void THFloatTensor_transpose(THFloatTensor *self, THFloatTensor *src, int dimension1_, int dimension2_);
extern void THFloatTensor_unfold(THFloatTensor *self, THFloatTensor *src, int dimension_, long size_, long step_);

extern void THFloatTensor_squeeze(THFloatTensor *self, THFloatTensor *src);
extern void THFloatTensor_squeeze1d(THFloatTensor *self, THFloatTensor *src, int dimension_);

extern int THFloatTensor_isContiguous(const THFloatTensor *self);
extern long THFloatTensor_nElement(const THFloatTensor *self);

extern void THFloatTensor_retain(THFloatTensor *self);
extern void THFloatTensor_free(THFloatTensor *self);
extern void THFloatTensor_freeCopyTo(THFloatTensor *self, THFloatTensor *dst);


extern void THFloatTensor_set1d(THFloatTensor *tensor, long x0, float value);
extern void THFloatTensor_set2d(THFloatTensor *tensor, long x0, long x1, float value);
extern void THFloatTensor_set3d(THFloatTensor *tensor, long x0, long x1, long x2, float value);
extern void THFloatTensor_set4d(THFloatTensor *tensor, long x0, long x1, long x2, long x3, float value);

extern float THFloatTensor_get1d(const THFloatTensor *tensor, long x0);
extern float THFloatTensor_get2d(const THFloatTensor *tensor, long x0, long x1);
extern float THFloatTensor_get3d(const THFloatTensor *tensor, long x0, long x1, long x2);
extern float THFloatTensor_get4d(const THFloatTensor *tensor, long x0, long x1, long x2, long x3);








typedef struct THDoubleTensor
{
    long *__size;
    long *__stride;
    int __nDimension;

    THDoubleStorage *__storage;
    long __storageOffset;
    int __refcount;

    char __flag;

} THDoubleTensor;



extern THDoubleStorage* THDoubleTensor_storage(const THDoubleTensor *self);
extern long THDoubleTensor_storageOffset(const THDoubleTensor *self);
extern int THDoubleTensor_nDimension(const THDoubleTensor *self);
extern long THDoubleTensor_size(const THDoubleTensor *self, int dim);
extern long THDoubleTensor_stride(const THDoubleTensor *self, int dim);
extern THLongStorage *THDoubleTensor_newSizeOf(THDoubleTensor *self);
extern THLongStorage *THDoubleTensor_newStrideOf(THDoubleTensor *self);
extern double *THDoubleTensor_data(const THDoubleTensor *self);

extern void THDoubleTensor_setFlag(THDoubleTensor *self, const char flag);
extern void THDoubleTensor_clearFlag(THDoubleTensor *self, const char flag);



extern THDoubleTensor *THDoubleTensor_new(void);
extern THDoubleTensor *THDoubleTensor_newWithTensor(THDoubleTensor *tensor);

extern THDoubleTensor *THDoubleTensor_newWithStorage(THDoubleStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern THDoubleTensor *THDoubleTensor_newWithStorage1d(THDoubleStorage *storage_, long storageOffset_,
                                long size0_, long stride0_);
extern THDoubleTensor *THDoubleTensor_newWithStorage2d(THDoubleStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
extern THDoubleTensor *THDoubleTensor_newWithStorage3d(THDoubleStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
extern THDoubleTensor *THDoubleTensor_newWithStorage4d(THDoubleStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);


extern THDoubleTensor *THDoubleTensor_newWithSize(THLongStorage *size_, THLongStorage *stride_);
extern THDoubleTensor *THDoubleTensor_newWithSize1d(long size0_);
extern THDoubleTensor *THDoubleTensor_newWithSize2d(long size0_, long size1_);
extern THDoubleTensor *THDoubleTensor_newWithSize3d(long size0_, long size1_, long size2_);
extern THDoubleTensor *THDoubleTensor_newWithSize4d(long size0_, long size1_, long size2_, long size3_);

extern THDoubleTensor *THDoubleTensor_newClone(THDoubleTensor *self);
extern THDoubleTensor *THDoubleTensor_newContiguous(THDoubleTensor *tensor);
extern THDoubleTensor *THDoubleTensor_newSelect(THDoubleTensor *tensor, int dimension_, long sliceIndex_);
extern THDoubleTensor *THDoubleTensor_newNarrow(THDoubleTensor *tensor, int dimension_, long firstIndex_, long size_);
extern THDoubleTensor *THDoubleTensor_newTranspose(THDoubleTensor *tensor, int dimension1_, int dimension2_);
extern THDoubleTensor *THDoubleTensor_newUnfold(THDoubleTensor *tensor, int dimension_, long size_, long step_);

extern void THDoubleTensor_resize(THDoubleTensor *tensor, THLongStorage *size, THLongStorage *stride);
extern void THDoubleTensor_resizeAs(THDoubleTensor *tensor, THDoubleTensor *src);
extern void THDoubleTensor_resize1d(THDoubleTensor *tensor, long size0_);
extern void THDoubleTensor_resize2d(THDoubleTensor *tensor, long size0_, long size1_);
extern void THDoubleTensor_resize3d(THDoubleTensor *tensor, long size0_, long size1_, long size2_);
extern void THDoubleTensor_resize4d(THDoubleTensor *tensor, long size0_, long size1_, long size2_, long size3_);
extern void THDoubleTensor_resize5d(THDoubleTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

extern void THDoubleTensor_set(THDoubleTensor *self, THDoubleTensor *src);
extern void THDoubleTensor_setStorage(THDoubleTensor *self, THDoubleStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
extern void THDoubleTensor_setStorage1d(THDoubleTensor *self, THDoubleStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_);
extern void THDoubleTensor_setStorage2d(THDoubleTensor *self, THDoubleStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_);
extern void THDoubleTensor_setStorage3d(THDoubleTensor *self, THDoubleStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_);
extern void THDoubleTensor_setStorage4d(THDoubleTensor *self, THDoubleStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_,
                                    long size3_, long stride3_);

extern void THDoubleTensor_narrow(THDoubleTensor *self, THDoubleTensor *src, int dimension_, long firstIndex_, long size_);
extern void THDoubleTensor_select(THDoubleTensor *self, THDoubleTensor *src, int dimension_, long sliceIndex_);
extern void THDoubleTensor_transpose(THDoubleTensor *self, THDoubleTensor *src, int dimension1_, int dimension2_);
extern void THDoubleTensor_unfold(THDoubleTensor *self, THDoubleTensor *src, int dimension_, long size_, long step_);

extern void THDoubleTensor_squeeze(THDoubleTensor *self, THDoubleTensor *src);
extern void THDoubleTensor_squeeze1d(THDoubleTensor *self, THDoubleTensor *src, int dimension_);

extern int THDoubleTensor_isContiguous(const THDoubleTensor *self);
extern long THDoubleTensor_nElement(const THDoubleTensor *self);

extern void THDoubleTensor_retain(THDoubleTensor *self);
extern void THDoubleTensor_free(THDoubleTensor *self);
extern void THDoubleTensor_freeCopyTo(THDoubleTensor *self, THDoubleTensor *dst);


extern void THDoubleTensor_set1d(THDoubleTensor *tensor, long x0, double value);
extern void THDoubleTensor_set2d(THDoubleTensor *tensor, long x0, long x1, double value);
extern void THDoubleTensor_set3d(THDoubleTensor *tensor, long x0, long x1, long x2, double value);
extern void THDoubleTensor_set4d(THDoubleTensor *tensor, long x0, long x1, long x2, long x3, double value);

extern double THDoubleTensor_get1d(const THDoubleTensor *tensor, long x0);
extern double THDoubleTensor_get2d(const THDoubleTensor *tensor, long x0, long x1);
extern double THDoubleTensor_get3d(const THDoubleTensor *tensor, long x0, long x1, long x2);
extern double THDoubleTensor_get4d(const THDoubleTensor *tensor, long x0, long x1, long x2, long x3);








extern void THByteTensor_copy(THByteTensor *tensor, THByteTensor *src);
extern void THByteTensor_copyByte(THByteTensor *tensor, struct THByteTensor *src);
extern void THByteTensor_copyChar(THByteTensor *tensor, struct THCharTensor *src);
extern void THByteTensor_copyShort(THByteTensor *tensor, struct THShortTensor *src);
extern void THByteTensor_copyInt(THByteTensor *tensor, struct THIntTensor *src);
extern void THByteTensor_copyLong(THByteTensor *tensor, struct THLongTensor *src);
extern void THByteTensor_copyFloat(THByteTensor *tensor, struct THFloatTensor *src);
extern void THByteTensor_copyDouble(THByteTensor *tensor, struct THDoubleTensor *src);






extern void THCharTensor_copy(THCharTensor *tensor, THCharTensor *src);
extern void THCharTensor_copyByte(THCharTensor *tensor, struct THByteTensor *src);
extern void THCharTensor_copyChar(THCharTensor *tensor, struct THCharTensor *src);
extern void THCharTensor_copyShort(THCharTensor *tensor, struct THShortTensor *src);
extern void THCharTensor_copyInt(THCharTensor *tensor, struct THIntTensor *src);
extern void THCharTensor_copyLong(THCharTensor *tensor, struct THLongTensor *src);
extern void THCharTensor_copyFloat(THCharTensor *tensor, struct THFloatTensor *src);
extern void THCharTensor_copyDouble(THCharTensor *tensor, struct THDoubleTensor *src);






extern void THShortTensor_copy(THShortTensor *tensor, THShortTensor *src);
extern void THShortTensor_copyByte(THShortTensor *tensor, struct THByteTensor *src);
extern void THShortTensor_copyChar(THShortTensor *tensor, struct THCharTensor *src);
extern void THShortTensor_copyShort(THShortTensor *tensor, struct THShortTensor *src);
extern void THShortTensor_copyInt(THShortTensor *tensor, struct THIntTensor *src);
extern void THShortTensor_copyLong(THShortTensor *tensor, struct THLongTensor *src);
extern void THShortTensor_copyFloat(THShortTensor *tensor, struct THFloatTensor *src);
extern void THShortTensor_copyDouble(THShortTensor *tensor, struct THDoubleTensor *src);






extern void THIntTensor_copy(THIntTensor *tensor, THIntTensor *src);
extern void THIntTensor_copyByte(THIntTensor *tensor, struct THByteTensor *src);
extern void THIntTensor_copyChar(THIntTensor *tensor, struct THCharTensor *src);
extern void THIntTensor_copyShort(THIntTensor *tensor, struct THShortTensor *src);
extern void THIntTensor_copyInt(THIntTensor *tensor, struct THIntTensor *src);
extern void THIntTensor_copyLong(THIntTensor *tensor, struct THLongTensor *src);
extern void THIntTensor_copyFloat(THIntTensor *tensor, struct THFloatTensor *src);
extern void THIntTensor_copyDouble(THIntTensor *tensor, struct THDoubleTensor *src);






extern void THLongTensor_copy(THLongTensor *tensor, THLongTensor *src);
extern void THLongTensor_copyByte(THLongTensor *tensor, struct THByteTensor *src);
extern void THLongTensor_copyChar(THLongTensor *tensor, struct THCharTensor *src);
extern void THLongTensor_copyShort(THLongTensor *tensor, struct THShortTensor *src);
extern void THLongTensor_copyInt(THLongTensor *tensor, struct THIntTensor *src);
extern void THLongTensor_copyLong(THLongTensor *tensor, struct THLongTensor *src);
extern void THLongTensor_copyFloat(THLongTensor *tensor, struct THFloatTensor *src);
extern void THLongTensor_copyDouble(THLongTensor *tensor, struct THDoubleTensor *src);






extern void THFloatTensor_copy(THFloatTensor *tensor, THFloatTensor *src);
extern void THFloatTensor_copyByte(THFloatTensor *tensor, struct THByteTensor *src);
extern void THFloatTensor_copyChar(THFloatTensor *tensor, struct THCharTensor *src);
extern void THFloatTensor_copyShort(THFloatTensor *tensor, struct THShortTensor *src);
extern void THFloatTensor_copyInt(THFloatTensor *tensor, struct THIntTensor *src);
extern void THFloatTensor_copyLong(THFloatTensor *tensor, struct THLongTensor *src);
extern void THFloatTensor_copyFloat(THFloatTensor *tensor, struct THFloatTensor *src);
extern void THFloatTensor_copyDouble(THFloatTensor *tensor, struct THDoubleTensor *src);






extern void THDoubleTensor_copy(THDoubleTensor *tensor, THDoubleTensor *src);
extern void THDoubleTensor_copyByte(THDoubleTensor *tensor, struct THByteTensor *src);
extern void THDoubleTensor_copyChar(THDoubleTensor *tensor, struct THCharTensor *src);
extern void THDoubleTensor_copyShort(THDoubleTensor *tensor, struct THShortTensor *src);
extern void THDoubleTensor_copyInt(THDoubleTensor *tensor, struct THIntTensor *src);
extern void THDoubleTensor_copyLong(THDoubleTensor *tensor, struct THLongTensor *src);
extern void THDoubleTensor_copyFloat(THDoubleTensor *tensor, struct THFloatTensor *src);
extern void THDoubleTensor_copyDouble(THDoubleTensor *tensor, struct THDoubleTensor *src);









extern void THByteTensor_random(THByteTensor *self, THGenerator *_generator);
extern void THByteTensor_geometric(THByteTensor *self, THGenerator *_generator, double p);
extern void THByteTensor_bernoulli(THByteTensor *self, THGenerator *_generator, double p);




extern void THCharTensor_random(THCharTensor *self, THGenerator *_generator);
extern void THCharTensor_geometric(THCharTensor *self, THGenerator *_generator, double p);
extern void THCharTensor_bernoulli(THCharTensor *self, THGenerator *_generator, double p);




extern void THShortTensor_random(THShortTensor *self, THGenerator *_generator);
extern void THShortTensor_geometric(THShortTensor *self, THGenerator *_generator, double p);
extern void THShortTensor_bernoulli(THShortTensor *self, THGenerator *_generator, double p);




extern void THIntTensor_random(THIntTensor *self, THGenerator *_generator);
extern void THIntTensor_geometric(THIntTensor *self, THGenerator *_generator, double p);
extern void THIntTensor_bernoulli(THIntTensor *self, THGenerator *_generator, double p);




extern void THLongTensor_random(THLongTensor *self, THGenerator *_generator);
extern void THLongTensor_geometric(THLongTensor *self, THGenerator *_generator, double p);
extern void THLongTensor_bernoulli(THLongTensor *self, THGenerator *_generator, double p);
extern void THLongTensor_getRNGState(THGenerator *_generator, THLongTensor *self);
extern void THLongTensor_setRNGState(THGenerator *_generator, THLongTensor *self);




extern void THFloatTensor_random(THFloatTensor *self, THGenerator *_generator);
extern void THFloatTensor_geometric(THFloatTensor *self, THGenerator *_generator, double p);
extern void THFloatTensor_bernoulli(THFloatTensor *self, THGenerator *_generator, double p);


extern void THFloatTensor_uniform(THFloatTensor *self, THGenerator *_generator, double a, double b);
extern void THFloatTensor_normal(THFloatTensor *self, THGenerator *_generator, double mean, double stdv);
extern void THFloatTensor_exponential(THFloatTensor *self, THGenerator *_generator, double lambda);
extern void THFloatTensor_cauchy(THFloatTensor *self, THGenerator *_generator, double median, double sigma);
extern void THFloatTensor_logNormal(THFloatTensor *self, THGenerator *_generator, double mean, double stdv);
extern void THFloatTensor_multinomial(THLongTensor *self, THGenerator *_generator, THFloatTensor *prob_dist, int n_sample, int with_replacement);




extern void THDoubleTensor_random(THDoubleTensor *self, THGenerator *_generator);
extern void THDoubleTensor_geometric(THDoubleTensor *self, THGenerator *_generator, double p);
extern void THDoubleTensor_bernoulli(THDoubleTensor *self, THGenerator *_generator, double p);


extern void THDoubleTensor_uniform(THDoubleTensor *self, THGenerator *_generator, double a, double b);
extern void THDoubleTensor_normal(THDoubleTensor *self, THGenerator *_generator, double mean, double stdv);
extern void THDoubleTensor_exponential(THDoubleTensor *self, THGenerator *_generator, double lambda);
extern void THDoubleTensor_cauchy(THDoubleTensor *self, THGenerator *_generator, double median, double sigma);
extern void THDoubleTensor_logNormal(THDoubleTensor *self, THGenerator *_generator, double mean, double stdv);
extern void THDoubleTensor_multinomial(THLongTensor *self, THGenerator *_generator, THDoubleTensor *prob_dist, int n_sample, int with_replacement);







extern void THByteTensor_fill(THByteTensor *r_, unsigned char value);
extern void THByteTensor_zero(THByteTensor *r_);

extern void THByteTensor_maskedFill(THByteTensor *tensor, THByteTensor *mask, unsigned char value);
extern void THByteTensor_maskedCopy(THByteTensor *tensor, THByteTensor *mask, THByteTensor* src);
extern void THByteTensor_maskedSelect(THByteTensor *tensor, THByteTensor* src, THByteTensor *mask);

extern void THByteTensor_indexSelect(THByteTensor *tensor, THByteTensor *src, int dim, THLongTensor *index);
extern void THByteTensor_indexCopy(THByteTensor *tensor, int dim, THLongTensor *index, THByteTensor *src);
extern void THByteTensor_indexFill(THByteTensor *tensor, int dim, THLongTensor *index, unsigned char val);

extern long THByteTensor_dot(THByteTensor *t, THByteTensor *src);

extern unsigned char THByteTensor_minall(THByteTensor *t);
extern unsigned char THByteTensor_maxall(THByteTensor *t);
extern long THByteTensor_sumall(THByteTensor *t);

extern void THByteTensor_add(THByteTensor *r_, THByteTensor *t, unsigned char value);
extern void THByteTensor_mul(THByteTensor *r_, THByteTensor *t, unsigned char value);
extern void THByteTensor_div(THByteTensor *r_, THByteTensor *t, unsigned char value);

extern void THByteTensor_cadd(THByteTensor *r_, THByteTensor *t, unsigned char value, THByteTensor *src);
extern void THByteTensor_cmul(THByteTensor *r_, THByteTensor *t, THByteTensor *src);
extern void THByteTensor_cdiv(THByteTensor *r_, THByteTensor *t, THByteTensor *src);

extern void THByteTensor_addcmul(THByteTensor *r_, THByteTensor *t, unsigned char value, THByteTensor *src1, THByteTensor *src2);
extern void THByteTensor_addcdiv(THByteTensor *r_, THByteTensor *t, unsigned char value, THByteTensor *src1, THByteTensor *src2);

extern void THByteTensor_addmv(THByteTensor *r_, unsigned char beta, THByteTensor *t, unsigned char alpha, THByteTensor *mat, THByteTensor *vec);
extern void THByteTensor_addmm(THByteTensor *r_, unsigned char beta, THByteTensor *t, unsigned char alpha, THByteTensor *mat1, THByteTensor *mat2);
extern void THByteTensor_addr(THByteTensor *r_, unsigned char beta, THByteTensor *t, unsigned char alpha, THByteTensor *vec1, THByteTensor *vec2);

extern void THByteTensor_match(THByteTensor *r_, THByteTensor *m1, THByteTensor *m2, unsigned char gain);

extern long THByteTensor_numel(THByteTensor *t);
extern void THByteTensor_max(THByteTensor *values_, THLongTensor *indices_, THByteTensor *t, int dimension);
extern void THByteTensor_min(THByteTensor *values_, THLongTensor *indices_, THByteTensor *t, int dimension);
extern void THByteTensor_sum(THByteTensor *r_, THByteTensor *t, int dimension);
extern void THByteTensor_prod(THByteTensor *r_, THByteTensor *t, int dimension);
extern void THByteTensor_cumsum(THByteTensor *r_, THByteTensor *t, int dimension);
extern void THByteTensor_cumprod(THByteTensor *r_, THByteTensor *t, int dimension);
extern void THByteTensor_sign(THByteTensor *r_, THByteTensor *t);
extern long THByteTensor_trace(THByteTensor *t);
extern void THByteTensor_cross(THByteTensor *r_, THByteTensor *a, THByteTensor *b, int dimension);

extern void THByteTensor_zeros(THByteTensor *r_, THLongStorage *size);
extern void THByteTensor_ones(THByteTensor *r_, THLongStorage *size);
extern void THByteTensor_diag(THByteTensor *r_, THByteTensor *t, int k);
extern void THByteTensor_eye(THByteTensor *r_, long n, long m);
extern void THByteTensor_range(THByteTensor *r_, unsigned char xmin, unsigned char xmax, unsigned char step);
extern void THByteTensor_randperm(THByteTensor *r_, THGenerator *_generator, long n);

extern void THByteTensor_reshape(THByteTensor *r_, THByteTensor *t, THLongStorage *size);
extern void THByteTensor_sort(THByteTensor *rt_, THLongTensor *ri_, THByteTensor *t, int dimension, int descendingOrder);
extern void THByteTensor_tril(THByteTensor *r_, THByteTensor *t, long k);
extern void THByteTensor_triu(THByteTensor *r_, THByteTensor *t, long k);
extern void THByteTensor_cat(THByteTensor *r_, THByteTensor *ta, THByteTensor *tb, int dimension);

extern void THByteTensor_ltValue(THByteTensor *r_, THByteTensor* t, unsigned char value);
extern void THByteTensor_leValue(THByteTensor *r_, THByteTensor* t, unsigned char value);
extern void THByteTensor_gtValue(THByteTensor *r_, THByteTensor* t, unsigned char value);
extern void THByteTensor_geValue(THByteTensor *r_, THByteTensor* t, unsigned char value);
extern void THByteTensor_neValue(THByteTensor *r_, THByteTensor* t, unsigned char value);
extern void THByteTensor_eqValue(THByteTensor *r_, THByteTensor* t, unsigned char value);

extern void THByteTensor_ltValueT(THByteTensor *r_, THByteTensor* t, unsigned char value);
extern void THByteTensor_leValueT(THByteTensor *r_, THByteTensor* t, unsigned char value);
extern void THByteTensor_gtValueT(THByteTensor *r_, THByteTensor* t, unsigned char value);
extern void THByteTensor_geValueT(THByteTensor *r_, THByteTensor* t, unsigned char value);
extern void THByteTensor_neValueT(THByteTensor *r_, THByteTensor* t, unsigned char value);
extern void THByteTensor_eqValueT(THByteTensor *r_, THByteTensor* t, unsigned char value);

extern void THByteTensor_ltTensor(THByteTensor *r_, THByteTensor *ta, THByteTensor *tb);
extern void THByteTensor_leTensor(THByteTensor *r_, THByteTensor *ta, THByteTensor *tb);
extern void THByteTensor_gtTensor(THByteTensor *r_, THByteTensor *ta, THByteTensor *tb);
extern void THByteTensor_geTensor(THByteTensor *r_, THByteTensor *ta, THByteTensor *tb);
extern void THByteTensor_neTensor(THByteTensor *r_, THByteTensor *ta, THByteTensor *tb);
extern void THByteTensor_eqTensor(THByteTensor *r_, THByteTensor *ta, THByteTensor *tb);

extern void THByteTensor_ltTensorT(THByteTensor *r_, THByteTensor *ta, THByteTensor *tb);
extern void THByteTensor_leTensorT(THByteTensor *r_, THByteTensor *ta, THByteTensor *tb);
extern void THByteTensor_gtTensorT(THByteTensor *r_, THByteTensor *ta, THByteTensor *tb);
extern void THByteTensor_geTensorT(THByteTensor *r_, THByteTensor *ta, THByteTensor *tb);
extern void THByteTensor_neTensorT(THByteTensor *r_, THByteTensor *ta, THByteTensor *tb);
extern void THByteTensor_eqTensorT(THByteTensor *r_, THByteTensor *ta, THByteTensor *tb);




extern void THCharTensor_fill(THCharTensor *r_, char value);
extern void THCharTensor_zero(THCharTensor *r_);

extern void THCharTensor_maskedFill(THCharTensor *tensor, THByteTensor *mask, char value);
extern void THCharTensor_maskedCopy(THCharTensor *tensor, THByteTensor *mask, THCharTensor* src);
extern void THCharTensor_maskedSelect(THCharTensor *tensor, THCharTensor* src, THByteTensor *mask);

extern void THCharTensor_indexSelect(THCharTensor *tensor, THCharTensor *src, int dim, THLongTensor *index);
extern void THCharTensor_indexCopy(THCharTensor *tensor, int dim, THLongTensor *index, THCharTensor *src);
extern void THCharTensor_indexFill(THCharTensor *tensor, int dim, THLongTensor *index, char val);

extern long THCharTensor_dot(THCharTensor *t, THCharTensor *src);

extern char THCharTensor_minall(THCharTensor *t);
extern char THCharTensor_maxall(THCharTensor *t);
extern long THCharTensor_sumall(THCharTensor *t);

extern void THCharTensor_add(THCharTensor *r_, THCharTensor *t, char value);
extern void THCharTensor_mul(THCharTensor *r_, THCharTensor *t, char value);
extern void THCharTensor_div(THCharTensor *r_, THCharTensor *t, char value);

extern void THCharTensor_cadd(THCharTensor *r_, THCharTensor *t, char value, THCharTensor *src);
extern void THCharTensor_cmul(THCharTensor *r_, THCharTensor *t, THCharTensor *src);
extern void THCharTensor_cdiv(THCharTensor *r_, THCharTensor *t, THCharTensor *src);

extern void THCharTensor_addcmul(THCharTensor *r_, THCharTensor *t, char value, THCharTensor *src1, THCharTensor *src2);
extern void THCharTensor_addcdiv(THCharTensor *r_, THCharTensor *t, char value, THCharTensor *src1, THCharTensor *src2);

extern void THCharTensor_addmv(THCharTensor *r_, char beta, THCharTensor *t, char alpha, THCharTensor *mat, THCharTensor *vec);
extern void THCharTensor_addmm(THCharTensor *r_, char beta, THCharTensor *t, char alpha, THCharTensor *mat1, THCharTensor *mat2);
extern void THCharTensor_addr(THCharTensor *r_, char beta, THCharTensor *t, char alpha, THCharTensor *vec1, THCharTensor *vec2);

extern void THCharTensor_match(THCharTensor *r_, THCharTensor *m1, THCharTensor *m2, char gain);

extern long THCharTensor_numel(THCharTensor *t);
extern void THCharTensor_max(THCharTensor *values_, THLongTensor *indices_, THCharTensor *t, int dimension);
extern void THCharTensor_min(THCharTensor *values_, THLongTensor *indices_, THCharTensor *t, int dimension);
extern void THCharTensor_sum(THCharTensor *r_, THCharTensor *t, int dimension);
extern void THCharTensor_prod(THCharTensor *r_, THCharTensor *t, int dimension);
extern void THCharTensor_cumsum(THCharTensor *r_, THCharTensor *t, int dimension);
extern void THCharTensor_cumprod(THCharTensor *r_, THCharTensor *t, int dimension);
extern void THCharTensor_sign(THCharTensor *r_, THCharTensor *t);
extern long THCharTensor_trace(THCharTensor *t);
extern void THCharTensor_cross(THCharTensor *r_, THCharTensor *a, THCharTensor *b, int dimension);

extern void THCharTensor_zeros(THCharTensor *r_, THLongStorage *size);
extern void THCharTensor_ones(THCharTensor *r_, THLongStorage *size);
extern void THCharTensor_diag(THCharTensor *r_, THCharTensor *t, int k);
extern void THCharTensor_eye(THCharTensor *r_, long n, long m);
extern void THCharTensor_range(THCharTensor *r_, char xmin, char xmax, char step);
extern void THCharTensor_randperm(THCharTensor *r_, THGenerator *_generator, long n);

extern void THCharTensor_reshape(THCharTensor *r_, THCharTensor *t, THLongStorage *size);
extern void THCharTensor_sort(THCharTensor *rt_, THLongTensor *ri_, THCharTensor *t, int dimension, int descendingOrder);
extern void THCharTensor_tril(THCharTensor *r_, THCharTensor *t, long k);
extern void THCharTensor_triu(THCharTensor *r_, THCharTensor *t, long k);
extern void THCharTensor_cat(THCharTensor *r_, THCharTensor *ta, THCharTensor *tb, int dimension);

extern void THCharTensor_ltValue(THByteTensor *r_, THCharTensor* t, char value);
extern void THCharTensor_leValue(THByteTensor *r_, THCharTensor* t, char value);
extern void THCharTensor_gtValue(THByteTensor *r_, THCharTensor* t, char value);
extern void THCharTensor_geValue(THByteTensor *r_, THCharTensor* t, char value);
extern void THCharTensor_neValue(THByteTensor *r_, THCharTensor* t, char value);
extern void THCharTensor_eqValue(THByteTensor *r_, THCharTensor* t, char value);

extern void THCharTensor_ltValueT(THCharTensor *r_, THCharTensor* t, char value);
extern void THCharTensor_leValueT(THCharTensor *r_, THCharTensor* t, char value);
extern void THCharTensor_gtValueT(THCharTensor *r_, THCharTensor* t, char value);
extern void THCharTensor_geValueT(THCharTensor *r_, THCharTensor* t, char value);
extern void THCharTensor_neValueT(THCharTensor *r_, THCharTensor* t, char value);
extern void THCharTensor_eqValueT(THCharTensor *r_, THCharTensor* t, char value);

extern void THCharTensor_ltTensor(THByteTensor *r_, THCharTensor *ta, THCharTensor *tb);
extern void THCharTensor_leTensor(THByteTensor *r_, THCharTensor *ta, THCharTensor *tb);
extern void THCharTensor_gtTensor(THByteTensor *r_, THCharTensor *ta, THCharTensor *tb);
extern void THCharTensor_geTensor(THByteTensor *r_, THCharTensor *ta, THCharTensor *tb);
extern void THCharTensor_neTensor(THByteTensor *r_, THCharTensor *ta, THCharTensor *tb);
extern void THCharTensor_eqTensor(THByteTensor *r_, THCharTensor *ta, THCharTensor *tb);

extern void THCharTensor_ltTensorT(THCharTensor *r_, THCharTensor *ta, THCharTensor *tb);
extern void THCharTensor_leTensorT(THCharTensor *r_, THCharTensor *ta, THCharTensor *tb);
extern void THCharTensor_gtTensorT(THCharTensor *r_, THCharTensor *ta, THCharTensor *tb);
extern void THCharTensor_geTensorT(THCharTensor *r_, THCharTensor *ta, THCharTensor *tb);
extern void THCharTensor_neTensorT(THCharTensor *r_, THCharTensor *ta, THCharTensor *tb);
extern void THCharTensor_eqTensorT(THCharTensor *r_, THCharTensor *ta, THCharTensor *tb);




extern void THShortTensor_fill(THShortTensor *r_, short value);
extern void THShortTensor_zero(THShortTensor *r_);

extern void THShortTensor_maskedFill(THShortTensor *tensor, THByteTensor *mask, short value);
extern void THShortTensor_maskedCopy(THShortTensor *tensor, THByteTensor *mask, THShortTensor* src);
extern void THShortTensor_maskedSelect(THShortTensor *tensor, THShortTensor* src, THByteTensor *mask);

extern void THShortTensor_indexSelect(THShortTensor *tensor, THShortTensor *src, int dim, THLongTensor *index);
extern void THShortTensor_indexCopy(THShortTensor *tensor, int dim, THLongTensor *index, THShortTensor *src);
extern void THShortTensor_indexFill(THShortTensor *tensor, int dim, THLongTensor *index, short val);

extern long THShortTensor_dot(THShortTensor *t, THShortTensor *src);

extern short THShortTensor_minall(THShortTensor *t);
extern short THShortTensor_maxall(THShortTensor *t);
extern long THShortTensor_sumall(THShortTensor *t);

extern void THShortTensor_add(THShortTensor *r_, THShortTensor *t, short value);
extern void THShortTensor_mul(THShortTensor *r_, THShortTensor *t, short value);
extern void THShortTensor_div(THShortTensor *r_, THShortTensor *t, short value);

extern void THShortTensor_cadd(THShortTensor *r_, THShortTensor *t, short value, THShortTensor *src);
extern void THShortTensor_cmul(THShortTensor *r_, THShortTensor *t, THShortTensor *src);
extern void THShortTensor_cdiv(THShortTensor *r_, THShortTensor *t, THShortTensor *src);

extern void THShortTensor_addcmul(THShortTensor *r_, THShortTensor *t, short value, THShortTensor *src1, THShortTensor *src2);
extern void THShortTensor_addcdiv(THShortTensor *r_, THShortTensor *t, short value, THShortTensor *src1, THShortTensor *src2);

extern void THShortTensor_addmv(THShortTensor *r_, short beta, THShortTensor *t, short alpha, THShortTensor *mat, THShortTensor *vec);
extern void THShortTensor_addmm(THShortTensor *r_, short beta, THShortTensor *t, short alpha, THShortTensor *mat1, THShortTensor *mat2);
extern void THShortTensor_addr(THShortTensor *r_, short beta, THShortTensor *t, short alpha, THShortTensor *vec1, THShortTensor *vec2);

extern void THShortTensor_match(THShortTensor *r_, THShortTensor *m1, THShortTensor *m2, short gain);

extern long THShortTensor_numel(THShortTensor *t);
extern void THShortTensor_max(THShortTensor *values_, THLongTensor *indices_, THShortTensor *t, int dimension);
extern void THShortTensor_min(THShortTensor *values_, THLongTensor *indices_, THShortTensor *t, int dimension);
extern void THShortTensor_sum(THShortTensor *r_, THShortTensor *t, int dimension);
extern void THShortTensor_prod(THShortTensor *r_, THShortTensor *t, int dimension);
extern void THShortTensor_cumsum(THShortTensor *r_, THShortTensor *t, int dimension);
extern void THShortTensor_cumprod(THShortTensor *r_, THShortTensor *t, int dimension);
extern void THShortTensor_sign(THShortTensor *r_, THShortTensor *t);
extern long THShortTensor_trace(THShortTensor *t);
extern void THShortTensor_cross(THShortTensor *r_, THShortTensor *a, THShortTensor *b, int dimension);

extern void THShortTensor_zeros(THShortTensor *r_, THLongStorage *size);
extern void THShortTensor_ones(THShortTensor *r_, THLongStorage *size);
extern void THShortTensor_diag(THShortTensor *r_, THShortTensor *t, int k);
extern void THShortTensor_eye(THShortTensor *r_, long n, long m);
extern void THShortTensor_range(THShortTensor *r_, short xmin, short xmax, short step);
extern void THShortTensor_randperm(THShortTensor *r_, THGenerator *_generator, long n);

extern void THShortTensor_reshape(THShortTensor *r_, THShortTensor *t, THLongStorage *size);
extern void THShortTensor_sort(THShortTensor *rt_, THLongTensor *ri_, THShortTensor *t, int dimension, int descendingOrder);
extern void THShortTensor_tril(THShortTensor *r_, THShortTensor *t, long k);
extern void THShortTensor_triu(THShortTensor *r_, THShortTensor *t, long k);
extern void THShortTensor_cat(THShortTensor *r_, THShortTensor *ta, THShortTensor *tb, int dimension);

extern void THShortTensor_ltValue(THByteTensor *r_, THShortTensor* t, short value);
extern void THShortTensor_leValue(THByteTensor *r_, THShortTensor* t, short value);
extern void THShortTensor_gtValue(THByteTensor *r_, THShortTensor* t, short value);
extern void THShortTensor_geValue(THByteTensor *r_, THShortTensor* t, short value);
extern void THShortTensor_neValue(THByteTensor *r_, THShortTensor* t, short value);
extern void THShortTensor_eqValue(THByteTensor *r_, THShortTensor* t, short value);

extern void THShortTensor_ltValueT(THShortTensor *r_, THShortTensor* t, short value);
extern void THShortTensor_leValueT(THShortTensor *r_, THShortTensor* t, short value);
extern void THShortTensor_gtValueT(THShortTensor *r_, THShortTensor* t, short value);
extern void THShortTensor_geValueT(THShortTensor *r_, THShortTensor* t, short value);
extern void THShortTensor_neValueT(THShortTensor *r_, THShortTensor* t, short value);
extern void THShortTensor_eqValueT(THShortTensor *r_, THShortTensor* t, short value);

extern void THShortTensor_ltTensor(THByteTensor *r_, THShortTensor *ta, THShortTensor *tb);
extern void THShortTensor_leTensor(THByteTensor *r_, THShortTensor *ta, THShortTensor *tb);
extern void THShortTensor_gtTensor(THByteTensor *r_, THShortTensor *ta, THShortTensor *tb);
extern void THShortTensor_geTensor(THByteTensor *r_, THShortTensor *ta, THShortTensor *tb);
extern void THShortTensor_neTensor(THByteTensor *r_, THShortTensor *ta, THShortTensor *tb);
extern void THShortTensor_eqTensor(THByteTensor *r_, THShortTensor *ta, THShortTensor *tb);

extern void THShortTensor_ltTensorT(THShortTensor *r_, THShortTensor *ta, THShortTensor *tb);
extern void THShortTensor_leTensorT(THShortTensor *r_, THShortTensor *ta, THShortTensor *tb);
extern void THShortTensor_gtTensorT(THShortTensor *r_, THShortTensor *ta, THShortTensor *tb);
extern void THShortTensor_geTensorT(THShortTensor *r_, THShortTensor *ta, THShortTensor *tb);
extern void THShortTensor_neTensorT(THShortTensor *r_, THShortTensor *ta, THShortTensor *tb);
extern void THShortTensor_eqTensorT(THShortTensor *r_, THShortTensor *ta, THShortTensor *tb);




extern void THIntTensor_fill(THIntTensor *r_, int value);
extern void THIntTensor_zero(THIntTensor *r_);

extern void THIntTensor_maskedFill(THIntTensor *tensor, THByteTensor *mask, int value);
extern void THIntTensor_maskedCopy(THIntTensor *tensor, THByteTensor *mask, THIntTensor* src);
extern void THIntTensor_maskedSelect(THIntTensor *tensor, THIntTensor* src, THByteTensor *mask);

extern void THIntTensor_indexSelect(THIntTensor *tensor, THIntTensor *src, int dim, THLongTensor *index);
extern void THIntTensor_indexCopy(THIntTensor *tensor, int dim, THLongTensor *index, THIntTensor *src);
extern void THIntTensor_indexFill(THIntTensor *tensor, int dim, THLongTensor *index, int val);

extern long THIntTensor_dot(THIntTensor *t, THIntTensor *src);

extern int THIntTensor_minall(THIntTensor *t);
extern int THIntTensor_maxall(THIntTensor *t);
extern long THIntTensor_sumall(THIntTensor *t);

extern void THIntTensor_add(THIntTensor *r_, THIntTensor *t, int value);
extern void THIntTensor_mul(THIntTensor *r_, THIntTensor *t, int value);
extern void THIntTensor_div(THIntTensor *r_, THIntTensor *t, int value);

extern void THIntTensor_cadd(THIntTensor *r_, THIntTensor *t, int value, THIntTensor *src);
extern void THIntTensor_cmul(THIntTensor *r_, THIntTensor *t, THIntTensor *src);
extern void THIntTensor_cdiv(THIntTensor *r_, THIntTensor *t, THIntTensor *src);

extern void THIntTensor_addcmul(THIntTensor *r_, THIntTensor *t, int value, THIntTensor *src1, THIntTensor *src2);
extern void THIntTensor_addcdiv(THIntTensor *r_, THIntTensor *t, int value, THIntTensor *src1, THIntTensor *src2);

extern void THIntTensor_addmv(THIntTensor *r_, int beta, THIntTensor *t, int alpha, THIntTensor *mat, THIntTensor *vec);
extern void THIntTensor_addmm(THIntTensor *r_, int beta, THIntTensor *t, int alpha, THIntTensor *mat1, THIntTensor *mat2);
extern void THIntTensor_addr(THIntTensor *r_, int beta, THIntTensor *t, int alpha, THIntTensor *vec1, THIntTensor *vec2);

extern void THIntTensor_match(THIntTensor *r_, THIntTensor *m1, THIntTensor *m2, int gain);

extern long THIntTensor_numel(THIntTensor *t);
extern void THIntTensor_max(THIntTensor *values_, THLongTensor *indices_, THIntTensor *t, int dimension);
extern void THIntTensor_min(THIntTensor *values_, THLongTensor *indices_, THIntTensor *t, int dimension);
extern void THIntTensor_sum(THIntTensor *r_, THIntTensor *t, int dimension);
extern void THIntTensor_prod(THIntTensor *r_, THIntTensor *t, int dimension);
extern void THIntTensor_cumsum(THIntTensor *r_, THIntTensor *t, int dimension);
extern void THIntTensor_cumprod(THIntTensor *r_, THIntTensor *t, int dimension);
extern void THIntTensor_sign(THIntTensor *r_, THIntTensor *t);
extern long THIntTensor_trace(THIntTensor *t);
extern void THIntTensor_cross(THIntTensor *r_, THIntTensor *a, THIntTensor *b, int dimension);

extern void THIntTensor_zeros(THIntTensor *r_, THLongStorage *size);
extern void THIntTensor_ones(THIntTensor *r_, THLongStorage *size);
extern void THIntTensor_diag(THIntTensor *r_, THIntTensor *t, int k);
extern void THIntTensor_eye(THIntTensor *r_, long n, long m);
extern void THIntTensor_range(THIntTensor *r_, int xmin, int xmax, int step);
extern void THIntTensor_randperm(THIntTensor *r_, THGenerator *_generator, long n);

extern void THIntTensor_reshape(THIntTensor *r_, THIntTensor *t, THLongStorage *size);
extern void THIntTensor_sort(THIntTensor *rt_, THLongTensor *ri_, THIntTensor *t, int dimension, int descendingOrder);
extern void THIntTensor_tril(THIntTensor *r_, THIntTensor *t, long k);
extern void THIntTensor_triu(THIntTensor *r_, THIntTensor *t, long k);
extern void THIntTensor_cat(THIntTensor *r_, THIntTensor *ta, THIntTensor *tb, int dimension);

extern void THIntTensor_ltValue(THByteTensor *r_, THIntTensor* t, int value);
extern void THIntTensor_leValue(THByteTensor *r_, THIntTensor* t, int value);
extern void THIntTensor_gtValue(THByteTensor *r_, THIntTensor* t, int value);
extern void THIntTensor_geValue(THByteTensor *r_, THIntTensor* t, int value);
extern void THIntTensor_neValue(THByteTensor *r_, THIntTensor* t, int value);
extern void THIntTensor_eqValue(THByteTensor *r_, THIntTensor* t, int value);

extern void THIntTensor_ltValueT(THIntTensor *r_, THIntTensor* t, int value);
extern void THIntTensor_leValueT(THIntTensor *r_, THIntTensor* t, int value);
extern void THIntTensor_gtValueT(THIntTensor *r_, THIntTensor* t, int value);
extern void THIntTensor_geValueT(THIntTensor *r_, THIntTensor* t, int value);
extern void THIntTensor_neValueT(THIntTensor *r_, THIntTensor* t, int value);
extern void THIntTensor_eqValueT(THIntTensor *r_, THIntTensor* t, int value);

extern void THIntTensor_ltTensor(THByteTensor *r_, THIntTensor *ta, THIntTensor *tb);
extern void THIntTensor_leTensor(THByteTensor *r_, THIntTensor *ta, THIntTensor *tb);
extern void THIntTensor_gtTensor(THByteTensor *r_, THIntTensor *ta, THIntTensor *tb);
extern void THIntTensor_geTensor(THByteTensor *r_, THIntTensor *ta, THIntTensor *tb);
extern void THIntTensor_neTensor(THByteTensor *r_, THIntTensor *ta, THIntTensor *tb);
extern void THIntTensor_eqTensor(THByteTensor *r_, THIntTensor *ta, THIntTensor *tb);

extern void THIntTensor_ltTensorT(THIntTensor *r_, THIntTensor *ta, THIntTensor *tb);
extern void THIntTensor_leTensorT(THIntTensor *r_, THIntTensor *ta, THIntTensor *tb);
extern void THIntTensor_gtTensorT(THIntTensor *r_, THIntTensor *ta, THIntTensor *tb);
extern void THIntTensor_geTensorT(THIntTensor *r_, THIntTensor *ta, THIntTensor *tb);
extern void THIntTensor_neTensorT(THIntTensor *r_, THIntTensor *ta, THIntTensor *tb);
extern void THIntTensor_eqTensorT(THIntTensor *r_, THIntTensor *ta, THIntTensor *tb);


extern void THIntTensor_abs(THIntTensor *r_, THIntTensor *t);




extern void THLongTensor_fill(THLongTensor *r_, long value);
extern void THLongTensor_zero(THLongTensor *r_);

extern void THLongTensor_maskedFill(THLongTensor *tensor, THByteTensor *mask, long value);
extern void THLongTensor_maskedCopy(THLongTensor *tensor, THByteTensor *mask, THLongTensor* src);
extern void THLongTensor_maskedSelect(THLongTensor *tensor, THLongTensor* src, THByteTensor *mask);

extern void THLongTensor_indexSelect(THLongTensor *tensor, THLongTensor *src, int dim, THLongTensor *index);
extern void THLongTensor_indexCopy(THLongTensor *tensor, int dim, THLongTensor *index, THLongTensor *src);
extern void THLongTensor_indexFill(THLongTensor *tensor, int dim, THLongTensor *index, long val);

extern long THLongTensor_dot(THLongTensor *t, THLongTensor *src);

extern long THLongTensor_minall(THLongTensor *t);
extern long THLongTensor_maxall(THLongTensor *t);
extern long THLongTensor_sumall(THLongTensor *t);

extern void THLongTensor_add(THLongTensor *r_, THLongTensor *t, long value);
extern void THLongTensor_mul(THLongTensor *r_, THLongTensor *t, long value);
extern void THLongTensor_div(THLongTensor *r_, THLongTensor *t, long value);

extern void THLongTensor_cadd(THLongTensor *r_, THLongTensor *t, long value, THLongTensor *src);
extern void THLongTensor_cmul(THLongTensor *r_, THLongTensor *t, THLongTensor *src);
extern void THLongTensor_cdiv(THLongTensor *r_, THLongTensor *t, THLongTensor *src);

extern void THLongTensor_addcmul(THLongTensor *r_, THLongTensor *t, long value, THLongTensor *src1, THLongTensor *src2);
extern void THLongTensor_addcdiv(THLongTensor *r_, THLongTensor *t, long value, THLongTensor *src1, THLongTensor *src2);

extern void THLongTensor_addmv(THLongTensor *r_, long beta, THLongTensor *t, long alpha, THLongTensor *mat, THLongTensor *vec);
extern void THLongTensor_addmm(THLongTensor *r_, long beta, THLongTensor *t, long alpha, THLongTensor *mat1, THLongTensor *mat2);
extern void THLongTensor_addr(THLongTensor *r_, long beta, THLongTensor *t, long alpha, THLongTensor *vec1, THLongTensor *vec2);

extern void THLongTensor_match(THLongTensor *r_, THLongTensor *m1, THLongTensor *m2, long gain);

extern long THLongTensor_numel(THLongTensor *t);
extern void THLongTensor_max(THLongTensor *values_, THLongTensor *indices_, THLongTensor *t, int dimension);
extern void THLongTensor_min(THLongTensor *values_, THLongTensor *indices_, THLongTensor *t, int dimension);
extern void THLongTensor_sum(THLongTensor *r_, THLongTensor *t, int dimension);
extern void THLongTensor_prod(THLongTensor *r_, THLongTensor *t, int dimension);
extern void THLongTensor_cumsum(THLongTensor *r_, THLongTensor *t, int dimension);
extern void THLongTensor_cumprod(THLongTensor *r_, THLongTensor *t, int dimension);
extern void THLongTensor_sign(THLongTensor *r_, THLongTensor *t);
extern long THLongTensor_trace(THLongTensor *t);
extern void THLongTensor_cross(THLongTensor *r_, THLongTensor *a, THLongTensor *b, int dimension);

extern void THLongTensor_zeros(THLongTensor *r_, THLongStorage *size);
extern void THLongTensor_ones(THLongTensor *r_, THLongStorage *size);
extern void THLongTensor_diag(THLongTensor *r_, THLongTensor *t, int k);
extern void THLongTensor_eye(THLongTensor *r_, long n, long m);
extern void THLongTensor_range(THLongTensor *r_, long xmin, long xmax, long step);
extern void THLongTensor_randperm(THLongTensor *r_, THGenerator *_generator, long n);

extern void THLongTensor_reshape(THLongTensor *r_, THLongTensor *t, THLongStorage *size);
extern void THLongTensor_sort(THLongTensor *rt_, THLongTensor *ri_, THLongTensor *t, int dimension, int descendingOrder);
extern void THLongTensor_tril(THLongTensor *r_, THLongTensor *t, long k);
extern void THLongTensor_triu(THLongTensor *r_, THLongTensor *t, long k);
extern void THLongTensor_cat(THLongTensor *r_, THLongTensor *ta, THLongTensor *tb, int dimension);

extern void THLongTensor_ltValue(THByteTensor *r_, THLongTensor* t, long value);
extern void THLongTensor_leValue(THByteTensor *r_, THLongTensor* t, long value);
extern void THLongTensor_gtValue(THByteTensor *r_, THLongTensor* t, long value);
extern void THLongTensor_geValue(THByteTensor *r_, THLongTensor* t, long value);
extern void THLongTensor_neValue(THByteTensor *r_, THLongTensor* t, long value);
extern void THLongTensor_eqValue(THByteTensor *r_, THLongTensor* t, long value);

extern void THLongTensor_ltValueT(THLongTensor *r_, THLongTensor* t, long value);
extern void THLongTensor_leValueT(THLongTensor *r_, THLongTensor* t, long value);
extern void THLongTensor_gtValueT(THLongTensor *r_, THLongTensor* t, long value);
extern void THLongTensor_geValueT(THLongTensor *r_, THLongTensor* t, long value);
extern void THLongTensor_neValueT(THLongTensor *r_, THLongTensor* t, long value);
extern void THLongTensor_eqValueT(THLongTensor *r_, THLongTensor* t, long value);

extern void THLongTensor_ltTensor(THByteTensor *r_, THLongTensor *ta, THLongTensor *tb);
extern void THLongTensor_leTensor(THByteTensor *r_, THLongTensor *ta, THLongTensor *tb);
extern void THLongTensor_gtTensor(THByteTensor *r_, THLongTensor *ta, THLongTensor *tb);
extern void THLongTensor_geTensor(THByteTensor *r_, THLongTensor *ta, THLongTensor *tb);
extern void THLongTensor_neTensor(THByteTensor *r_, THLongTensor *ta, THLongTensor *tb);
extern void THLongTensor_eqTensor(THByteTensor *r_, THLongTensor *ta, THLongTensor *tb);

extern void THLongTensor_ltTensorT(THLongTensor *r_, THLongTensor *ta, THLongTensor *tb);
extern void THLongTensor_leTensorT(THLongTensor *r_, THLongTensor *ta, THLongTensor *tb);
extern void THLongTensor_gtTensorT(THLongTensor *r_, THLongTensor *ta, THLongTensor *tb);
extern void THLongTensor_geTensorT(THLongTensor *r_, THLongTensor *ta, THLongTensor *tb);
extern void THLongTensor_neTensorT(THLongTensor *r_, THLongTensor *ta, THLongTensor *tb);
extern void THLongTensor_eqTensorT(THLongTensor *r_, THLongTensor *ta, THLongTensor *tb);


extern void THLongTensor_abs(THLongTensor *r_, THLongTensor *t);




extern void THFloatTensor_fill(THFloatTensor *r_, float value);
extern void THFloatTensor_zero(THFloatTensor *r_);

extern void THFloatTensor_maskedFill(THFloatTensor *tensor, THByteTensor *mask, float value);
extern void THFloatTensor_maskedCopy(THFloatTensor *tensor, THByteTensor *mask, THFloatTensor* src);
extern void THFloatTensor_maskedSelect(THFloatTensor *tensor, THFloatTensor* src, THByteTensor *mask);

extern void THFloatTensor_indexSelect(THFloatTensor *tensor, THFloatTensor *src, int dim, THLongTensor *index);
extern void THFloatTensor_indexCopy(THFloatTensor *tensor, int dim, THLongTensor *index, THFloatTensor *src);
extern void THFloatTensor_indexFill(THFloatTensor *tensor, int dim, THLongTensor *index, float val);

extern double THFloatTensor_dot(THFloatTensor *t, THFloatTensor *src);

extern float THFloatTensor_minall(THFloatTensor *t);
extern float THFloatTensor_maxall(THFloatTensor *t);
extern double THFloatTensor_sumall(THFloatTensor *t);

extern void THFloatTensor_add(THFloatTensor *r_, THFloatTensor *t, float value);
extern void THFloatTensor_mul(THFloatTensor *r_, THFloatTensor *t, float value);
extern void THFloatTensor_div(THFloatTensor *r_, THFloatTensor *t, float value);

extern void THFloatTensor_cadd(THFloatTensor *r_, THFloatTensor *t, float value, THFloatTensor *src);
extern void THFloatTensor_cmul(THFloatTensor *r_, THFloatTensor *t, THFloatTensor *src);
extern void THFloatTensor_cdiv(THFloatTensor *r_, THFloatTensor *t, THFloatTensor *src);

extern void THFloatTensor_addcmul(THFloatTensor *r_, THFloatTensor *t, float value, THFloatTensor *src1, THFloatTensor *src2);
extern void THFloatTensor_addcdiv(THFloatTensor *r_, THFloatTensor *t, float value, THFloatTensor *src1, THFloatTensor *src2);

extern void THFloatTensor_addmv(THFloatTensor *r_, float beta, THFloatTensor *t, float alpha, THFloatTensor *mat, THFloatTensor *vec);
extern void THFloatTensor_addmm(THFloatTensor *r_, float beta, THFloatTensor *t, float alpha, THFloatTensor *mat1, THFloatTensor *mat2);
extern void THFloatTensor_addr(THFloatTensor *r_, float beta, THFloatTensor *t, float alpha, THFloatTensor *vec1, THFloatTensor *vec2);

extern void THFloatTensor_match(THFloatTensor *r_, THFloatTensor *m1, THFloatTensor *m2, float gain);

extern long THFloatTensor_numel(THFloatTensor *t);
extern void THFloatTensor_max(THFloatTensor *values_, THLongTensor *indices_, THFloatTensor *t, int dimension);
extern void THFloatTensor_min(THFloatTensor *values_, THLongTensor *indices_, THFloatTensor *t, int dimension);
extern void THFloatTensor_sum(THFloatTensor *r_, THFloatTensor *t, int dimension);
extern void THFloatTensor_prod(THFloatTensor *r_, THFloatTensor *t, int dimension);
extern void THFloatTensor_cumsum(THFloatTensor *r_, THFloatTensor *t, int dimension);
extern void THFloatTensor_cumprod(THFloatTensor *r_, THFloatTensor *t, int dimension);
extern void THFloatTensor_sign(THFloatTensor *r_, THFloatTensor *t);
extern double THFloatTensor_trace(THFloatTensor *t);
extern void THFloatTensor_cross(THFloatTensor *r_, THFloatTensor *a, THFloatTensor *b, int dimension);

extern void THFloatTensor_zeros(THFloatTensor *r_, THLongStorage *size);
extern void THFloatTensor_ones(THFloatTensor *r_, THLongStorage *size);
extern void THFloatTensor_diag(THFloatTensor *r_, THFloatTensor *t, int k);
extern void THFloatTensor_eye(THFloatTensor *r_, long n, long m);
extern void THFloatTensor_range(THFloatTensor *r_, float xmin, float xmax, float step);
extern void THFloatTensor_randperm(THFloatTensor *r_, THGenerator *_generator, long n);

extern void THFloatTensor_reshape(THFloatTensor *r_, THFloatTensor *t, THLongStorage *size);
extern void THFloatTensor_sort(THFloatTensor *rt_, THLongTensor *ri_, THFloatTensor *t, int dimension, int descendingOrder);
extern void THFloatTensor_tril(THFloatTensor *r_, THFloatTensor *t, long k);
extern void THFloatTensor_triu(THFloatTensor *r_, THFloatTensor *t, long k);
extern void THFloatTensor_cat(THFloatTensor *r_, THFloatTensor *ta, THFloatTensor *tb, int dimension);

extern void THFloatTensor_ltValue(THByteTensor *r_, THFloatTensor* t, float value);
extern void THFloatTensor_leValue(THByteTensor *r_, THFloatTensor* t, float value);
extern void THFloatTensor_gtValue(THByteTensor *r_, THFloatTensor* t, float value);
extern void THFloatTensor_geValue(THByteTensor *r_, THFloatTensor* t, float value);
extern void THFloatTensor_neValue(THByteTensor *r_, THFloatTensor* t, float value);
extern void THFloatTensor_eqValue(THByteTensor *r_, THFloatTensor* t, float value);

extern void THFloatTensor_ltValueT(THFloatTensor *r_, THFloatTensor* t, float value);
extern void THFloatTensor_leValueT(THFloatTensor *r_, THFloatTensor* t, float value);
extern void THFloatTensor_gtValueT(THFloatTensor *r_, THFloatTensor* t, float value);
extern void THFloatTensor_geValueT(THFloatTensor *r_, THFloatTensor* t, float value);
extern void THFloatTensor_neValueT(THFloatTensor *r_, THFloatTensor* t, float value);
extern void THFloatTensor_eqValueT(THFloatTensor *r_, THFloatTensor* t, float value);

extern void THFloatTensor_ltTensor(THByteTensor *r_, THFloatTensor *ta, THFloatTensor *tb);
extern void THFloatTensor_leTensor(THByteTensor *r_, THFloatTensor *ta, THFloatTensor *tb);
extern void THFloatTensor_gtTensor(THByteTensor *r_, THFloatTensor *ta, THFloatTensor *tb);
extern void THFloatTensor_geTensor(THByteTensor *r_, THFloatTensor *ta, THFloatTensor *tb);
extern void THFloatTensor_neTensor(THByteTensor *r_, THFloatTensor *ta, THFloatTensor *tb);
extern void THFloatTensor_eqTensor(THByteTensor *r_, THFloatTensor *ta, THFloatTensor *tb);

extern void THFloatTensor_ltTensorT(THFloatTensor *r_, THFloatTensor *ta, THFloatTensor *tb);
extern void THFloatTensor_leTensorT(THFloatTensor *r_, THFloatTensor *ta, THFloatTensor *tb);
extern void THFloatTensor_gtTensorT(THFloatTensor *r_, THFloatTensor *ta, THFloatTensor *tb);
extern void THFloatTensor_geTensorT(THFloatTensor *r_, THFloatTensor *ta, THFloatTensor *tb);
extern void THFloatTensor_neTensorT(THFloatTensor *r_, THFloatTensor *ta, THFloatTensor *tb);
extern void THFloatTensor_eqTensorT(THFloatTensor *r_, THFloatTensor *ta, THFloatTensor *tb);







extern void THFloatTensor_log(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_log1p(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_exp(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_cos(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_acos(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_cosh(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_sin(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_asin(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_sinh(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_tan(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_atan(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_atan2(THFloatTensor *r_, THFloatTensor *tx, THFloatTensor *ty);
extern void THFloatTensor_tanh(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_pow(THFloatTensor *r_, THFloatTensor *t, float value);
extern void THFloatTensor_sqrt(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_ceil(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_floor(THFloatTensor *r_, THFloatTensor *t);
extern void THFloatTensor_abs(THFloatTensor *r_, THFloatTensor *t);

extern void THFloatTensor_mean(THFloatTensor *r_, THFloatTensor *t, int dimension);
extern void THFloatTensor_std(THFloatTensor *r_, THFloatTensor *t, int dimension, int flag);
extern void THFloatTensor_var(THFloatTensor *r_, THFloatTensor *t, int dimension, int flag);
extern void THFloatTensor_norm(THFloatTensor *r_, THFloatTensor *t, float value, int dimension);
extern double THFloatTensor_dist(THFloatTensor *a, THFloatTensor *b, float value);
extern void THFloatTensor_histc(THFloatTensor *hist, THFloatTensor *tensor, long nbins, float minvalue, float maxvalue);

extern double THFloatTensor_meanall(THFloatTensor *self);
extern double THFloatTensor_varall(THFloatTensor *self);
extern double THFloatTensor_stdall(THFloatTensor *self);
extern double THFloatTensor_normall(THFloatTensor *t, float value);

extern void THFloatTensor_linspace(THFloatTensor *r_, float a, float b, long n);
extern void THFloatTensor_logspace(THFloatTensor *r_, float a, float b, long n);
extern void THFloatTensor_rand(THFloatTensor *r_, THGenerator *_generator, THLongStorage *size);
extern void THFloatTensor_randn(THFloatTensor *r_, THGenerator *_generator, THLongStorage *size);




extern void THDoubleTensor_fill(THDoubleTensor *r_, double value);
extern void THDoubleTensor_zero(THDoubleTensor *r_);

extern void THDoubleTensor_maskedFill(THDoubleTensor *tensor, THByteTensor *mask, double value);
extern void THDoubleTensor_maskedCopy(THDoubleTensor *tensor, THByteTensor *mask, THDoubleTensor* src);
extern void THDoubleTensor_maskedSelect(THDoubleTensor *tensor, THDoubleTensor* src, THByteTensor *mask);

extern void THDoubleTensor_indexSelect(THDoubleTensor *tensor, THDoubleTensor *src, int dim, THLongTensor *index);
extern void THDoubleTensor_indexCopy(THDoubleTensor *tensor, int dim, THLongTensor *index, THDoubleTensor *src);
extern void THDoubleTensor_indexFill(THDoubleTensor *tensor, int dim, THLongTensor *index, double val);

extern double THDoubleTensor_dot(THDoubleTensor *t, THDoubleTensor *src);

extern double THDoubleTensor_minall(THDoubleTensor *t);
extern double THDoubleTensor_maxall(THDoubleTensor *t);
extern double THDoubleTensor_sumall(THDoubleTensor *t);

extern void THDoubleTensor_add(THDoubleTensor *r_, THDoubleTensor *t, double value);
extern void THDoubleTensor_mul(THDoubleTensor *r_, THDoubleTensor *t, double value);
extern void THDoubleTensor_div(THDoubleTensor *r_, THDoubleTensor *t, double value);

extern void THDoubleTensor_cadd(THDoubleTensor *r_, THDoubleTensor *t, double value, THDoubleTensor *src);
extern void THDoubleTensor_cmul(THDoubleTensor *r_, THDoubleTensor *t, THDoubleTensor *src);
extern void THDoubleTensor_cdiv(THDoubleTensor *r_, THDoubleTensor *t, THDoubleTensor *src);

extern void THDoubleTensor_addcmul(THDoubleTensor *r_, THDoubleTensor *t, double value, THDoubleTensor *src1, THDoubleTensor *src2);
extern void THDoubleTensor_addcdiv(THDoubleTensor *r_, THDoubleTensor *t, double value, THDoubleTensor *src1, THDoubleTensor *src2);

extern void THDoubleTensor_addmv(THDoubleTensor *r_, double beta, THDoubleTensor *t, double alpha, THDoubleTensor *mat, THDoubleTensor *vec);
extern void THDoubleTensor_addmm(THDoubleTensor *r_, double beta, THDoubleTensor *t, double alpha, THDoubleTensor *mat1, THDoubleTensor *mat2);
extern void THDoubleTensor_addr(THDoubleTensor *r_, double beta, THDoubleTensor *t, double alpha, THDoubleTensor *vec1, THDoubleTensor *vec2);

extern void THDoubleTensor_match(THDoubleTensor *r_, THDoubleTensor *m1, THDoubleTensor *m2, double gain);

extern long THDoubleTensor_numel(THDoubleTensor *t);
extern void THDoubleTensor_max(THDoubleTensor *values_, THLongTensor *indices_, THDoubleTensor *t, int dimension);
extern void THDoubleTensor_min(THDoubleTensor *values_, THLongTensor *indices_, THDoubleTensor *t, int dimension);
extern void THDoubleTensor_sum(THDoubleTensor *r_, THDoubleTensor *t, int dimension);
extern void THDoubleTensor_prod(THDoubleTensor *r_, THDoubleTensor *t, int dimension);
extern void THDoubleTensor_cumsum(THDoubleTensor *r_, THDoubleTensor *t, int dimension);
extern void THDoubleTensor_cumprod(THDoubleTensor *r_, THDoubleTensor *t, int dimension);
extern void THDoubleTensor_sign(THDoubleTensor *r_, THDoubleTensor *t);
extern double THDoubleTensor_trace(THDoubleTensor *t);
extern void THDoubleTensor_cross(THDoubleTensor *r_, THDoubleTensor *a, THDoubleTensor *b, int dimension);

extern void THDoubleTensor_zeros(THDoubleTensor *r_, THLongStorage *size);
extern void THDoubleTensor_ones(THDoubleTensor *r_, THLongStorage *size);
extern void THDoubleTensor_diag(THDoubleTensor *r_, THDoubleTensor *t, int k);
extern void THDoubleTensor_eye(THDoubleTensor *r_, long n, long m);
extern void THDoubleTensor_range(THDoubleTensor *r_, double xmin, double xmax, double step);
extern void THDoubleTensor_randperm(THDoubleTensor *r_, THGenerator *_generator, long n);

extern void THDoubleTensor_reshape(THDoubleTensor *r_, THDoubleTensor *t, THLongStorage *size);
extern void THDoubleTensor_sort(THDoubleTensor *rt_, THLongTensor *ri_, THDoubleTensor *t, int dimension, int descendingOrder);
extern void THDoubleTensor_tril(THDoubleTensor *r_, THDoubleTensor *t, long k);
extern void THDoubleTensor_triu(THDoubleTensor *r_, THDoubleTensor *t, long k);
extern void THDoubleTensor_cat(THDoubleTensor *r_, THDoubleTensor *ta, THDoubleTensor *tb, int dimension);

extern void THDoubleTensor_ltValue(THByteTensor *r_, THDoubleTensor* t, double value);
extern void THDoubleTensor_leValue(THByteTensor *r_, THDoubleTensor* t, double value);
extern void THDoubleTensor_gtValue(THByteTensor *r_, THDoubleTensor* t, double value);
extern void THDoubleTensor_geValue(THByteTensor *r_, THDoubleTensor* t, double value);
extern void THDoubleTensor_neValue(THByteTensor *r_, THDoubleTensor* t, double value);
extern void THDoubleTensor_eqValue(THByteTensor *r_, THDoubleTensor* t, double value);

extern void THDoubleTensor_ltValueT(THDoubleTensor *r_, THDoubleTensor* t, double value);
extern void THDoubleTensor_leValueT(THDoubleTensor *r_, THDoubleTensor* t, double value);
extern void THDoubleTensor_gtValueT(THDoubleTensor *r_, THDoubleTensor* t, double value);
extern void THDoubleTensor_geValueT(THDoubleTensor *r_, THDoubleTensor* t, double value);
extern void THDoubleTensor_neValueT(THDoubleTensor *r_, THDoubleTensor* t, double value);
extern void THDoubleTensor_eqValueT(THDoubleTensor *r_, THDoubleTensor* t, double value);

extern void THDoubleTensor_ltTensor(THByteTensor *r_, THDoubleTensor *ta, THDoubleTensor *tb);
extern void THDoubleTensor_leTensor(THByteTensor *r_, THDoubleTensor *ta, THDoubleTensor *tb);
extern void THDoubleTensor_gtTensor(THByteTensor *r_, THDoubleTensor *ta, THDoubleTensor *tb);
extern void THDoubleTensor_geTensor(THByteTensor *r_, THDoubleTensor *ta, THDoubleTensor *tb);
extern void THDoubleTensor_neTensor(THByteTensor *r_, THDoubleTensor *ta, THDoubleTensor *tb);
extern void THDoubleTensor_eqTensor(THByteTensor *r_, THDoubleTensor *ta, THDoubleTensor *tb);

extern void THDoubleTensor_ltTensorT(THDoubleTensor *r_, THDoubleTensor *ta, THDoubleTensor *tb);
extern void THDoubleTensor_leTensorT(THDoubleTensor *r_, THDoubleTensor *ta, THDoubleTensor *tb);
extern void THDoubleTensor_gtTensorT(THDoubleTensor *r_, THDoubleTensor *ta, THDoubleTensor *tb);
extern void THDoubleTensor_geTensorT(THDoubleTensor *r_, THDoubleTensor *ta, THDoubleTensor *tb);
extern void THDoubleTensor_neTensorT(THDoubleTensor *r_, THDoubleTensor *ta, THDoubleTensor *tb);
extern void THDoubleTensor_eqTensorT(THDoubleTensor *r_, THDoubleTensor *ta, THDoubleTensor *tb);







extern void THDoubleTensor_log(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_log1p(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_exp(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_cos(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_acos(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_cosh(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_sin(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_asin(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_sinh(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_tan(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_atan(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_atan2(THDoubleTensor *r_, THDoubleTensor *tx, THDoubleTensor *ty);
extern void THDoubleTensor_tanh(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_pow(THDoubleTensor *r_, THDoubleTensor *t, double value);
extern void THDoubleTensor_sqrt(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_ceil(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_floor(THDoubleTensor *r_, THDoubleTensor *t);
extern void THDoubleTensor_abs(THDoubleTensor *r_, THDoubleTensor *t);

extern void THDoubleTensor_mean(THDoubleTensor *r_, THDoubleTensor *t, int dimension);
extern void THDoubleTensor_std(THDoubleTensor *r_, THDoubleTensor *t, int dimension, int flag);
extern void THDoubleTensor_var(THDoubleTensor *r_, THDoubleTensor *t, int dimension, int flag);
extern void THDoubleTensor_norm(THDoubleTensor *r_, THDoubleTensor *t, double value, int dimension);
extern double THDoubleTensor_dist(THDoubleTensor *a, THDoubleTensor *b, double value);
extern void THDoubleTensor_histc(THDoubleTensor *hist, THDoubleTensor *tensor, long nbins, double minvalue, double maxvalue);

extern double THDoubleTensor_meanall(THDoubleTensor *self);
extern double THDoubleTensor_varall(THDoubleTensor *self);
extern double THDoubleTensor_stdall(THDoubleTensor *self);
extern double THDoubleTensor_normall(THDoubleTensor *t, double value);

extern void THDoubleTensor_linspace(THDoubleTensor *r_, double a, double b, long n);
extern void THDoubleTensor_logspace(THDoubleTensor *r_, double a, double b, long n);
extern void THDoubleTensor_rand(THDoubleTensor *r_, THGenerator *_generator, THLongStorage *size);
extern void THDoubleTensor_randn(THDoubleTensor *r_, THGenerator *_generator, THLongStorage *size);








extern void THByteTensor_validXCorr2Dptr(unsigned char *r_,
                                    unsigned char alpha,
                                    unsigned char *t_, long ir, long ic,
                                    unsigned char *k_, long kr, long kc,
                                    long sr, long sc);

extern void THByteTensor_validConv2Dptr(unsigned char *r_,
                                   unsigned char alpha,
                                   unsigned char *t_, long ir, long ic,
                                   unsigned char *k_, long kr, long kc,
                                   long sr, long sc);

extern void THByteTensor_fullXCorr2Dptr(unsigned char *r_,
                                   unsigned char alpha,
                                   unsigned char *t_, long ir, long ic,
                                   unsigned char *k_, long kr, long kc,
                                   long sr, long sc);

extern void THByteTensor_fullConv2Dptr(unsigned char *r_,
                                  unsigned char alpha,
                                  unsigned char *t_, long ir, long ic,
                                  unsigned char *k_, long kr, long kc,
                                  long sr, long sc);

extern void THByteTensor_validXCorr2DRevptr(unsigned char *r_,
                                       unsigned char alpha,
                                       unsigned char *t_, long ir, long ic,
                                       unsigned char *k_, long kr, long kc,
                                       long sr, long sc);

extern void THByteTensor_conv2DRevger(THByteTensor *r_, unsigned char beta, unsigned char alpha, THByteTensor *t_, THByteTensor *k_, long srow, long scol);
extern void THByteTensor_conv2DRevgerm(THByteTensor *r_, unsigned char beta, unsigned char alpha, THByteTensor *t_, THByteTensor *k_, long srow, long scol);
extern void THByteTensor_conv2Dger(THByteTensor *r_, unsigned char beta, unsigned char alpha, THByteTensor *t_, THByteTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THByteTensor_conv2Dmv(THByteTensor *r_, unsigned char beta, unsigned char alpha, THByteTensor *t_, THByteTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THByteTensor_conv2Dmm(THByteTensor *r_, unsigned char beta, unsigned char alpha, THByteTensor *t_, THByteTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THByteTensor_conv2Dmul(THByteTensor *r_, unsigned char beta, unsigned char alpha, THByteTensor *t_, THByteTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THByteTensor_conv2Dcmul(THByteTensor *r_, unsigned char beta, unsigned char alpha, THByteTensor *t_, THByteTensor *k_, long srow, long scol, const char *vf, const char *xc);

extern void THByteTensor_validXCorr3Dptr(unsigned char *r_,
                                    unsigned char alpha,
                                    unsigned char *t_, long it, long ir, long ic,
                                    unsigned char *k_, long kt, long kr, long kc,
                                    long st, long sr, long sc);

extern void THByteTensor_validConv3Dptr(unsigned char *r_,
                                   unsigned char alpha,
                                   unsigned char *t_, long it, long ir, long ic,
                                   unsigned char *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THByteTensor_fullXCorr3Dptr(unsigned char *r_,
                                   unsigned char alpha,
                                   unsigned char *t_, long it, long ir, long ic,
                                   unsigned char *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THByteTensor_fullConv3Dptr(unsigned char *r_,
                                  unsigned char alpha,
                                  unsigned char *t_, long it, long ir, long ic,
                                  unsigned char *k_, long kt, long kr, long kc,
                                  long st, long sr, long sc);

extern void THByteTensor_validXCorr3DRevptr(unsigned char *r_,
                                       unsigned char alpha,
                                       unsigned char *t_, long it, long ir, long ic,
                                       unsigned char *k_, long kt, long kr, long kc,
                                       long st, long sr, long sc);

extern void THByteTensor_conv3DRevger(THByteTensor *r_, unsigned char beta, unsigned char alpha, THByteTensor *t_, THByteTensor *k_, long sdepth, long srow, long scol);
extern void THByteTensor_conv3Dger(THByteTensor *r_, unsigned char beta, unsigned char alpha, THByteTensor *t_, THByteTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THByteTensor_conv3Dmv(THByteTensor *r_, unsigned char beta, unsigned char alpha, THByteTensor *t_, THByteTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THByteTensor_conv3Dmul(THByteTensor *r_, unsigned char beta, unsigned char alpha, THByteTensor *t_, THByteTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THByteTensor_conv3Dcmul(THByteTensor *r_, unsigned char beta, unsigned char alpha, THByteTensor *t_, THByteTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);





extern void THCharTensor_validXCorr2Dptr(char *r_,
                                    char alpha,
                                    char *t_, long ir, long ic,
                                    char *k_, long kr, long kc,
                                    long sr, long sc);

extern void THCharTensor_validConv2Dptr(char *r_,
                                   char alpha,
                                   char *t_, long ir, long ic,
                                   char *k_, long kr, long kc,
                                   long sr, long sc);

extern void THCharTensor_fullXCorr2Dptr(char *r_,
                                   char alpha,
                                   char *t_, long ir, long ic,
                                   char *k_, long kr, long kc,
                                   long sr, long sc);

extern void THCharTensor_fullConv2Dptr(char *r_,
                                  char alpha,
                                  char *t_, long ir, long ic,
                                  char *k_, long kr, long kc,
                                  long sr, long sc);

extern void THCharTensor_validXCorr2DRevptr(char *r_,
                                       char alpha,
                                       char *t_, long ir, long ic,
                                       char *k_, long kr, long kc,
                                       long sr, long sc);

extern void THCharTensor_conv2DRevger(THCharTensor *r_, char beta, char alpha, THCharTensor *t_, THCharTensor *k_, long srow, long scol);
extern void THCharTensor_conv2DRevgerm(THCharTensor *r_, char beta, char alpha, THCharTensor *t_, THCharTensor *k_, long srow, long scol);
extern void THCharTensor_conv2Dger(THCharTensor *r_, char beta, char alpha, THCharTensor *t_, THCharTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THCharTensor_conv2Dmv(THCharTensor *r_, char beta, char alpha, THCharTensor *t_, THCharTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THCharTensor_conv2Dmm(THCharTensor *r_, char beta, char alpha, THCharTensor *t_, THCharTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THCharTensor_conv2Dmul(THCharTensor *r_, char beta, char alpha, THCharTensor *t_, THCharTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THCharTensor_conv2Dcmul(THCharTensor *r_, char beta, char alpha, THCharTensor *t_, THCharTensor *k_, long srow, long scol, const char *vf, const char *xc);

extern void THCharTensor_validXCorr3Dptr(char *r_,
                                    char alpha,
                                    char *t_, long it, long ir, long ic,
                                    char *k_, long kt, long kr, long kc,
                                    long st, long sr, long sc);

extern void THCharTensor_validConv3Dptr(char *r_,
                                   char alpha,
                                   char *t_, long it, long ir, long ic,
                                   char *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THCharTensor_fullXCorr3Dptr(char *r_,
                                   char alpha,
                                   char *t_, long it, long ir, long ic,
                                   char *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THCharTensor_fullConv3Dptr(char *r_,
                                  char alpha,
                                  char *t_, long it, long ir, long ic,
                                  char *k_, long kt, long kr, long kc,
                                  long st, long sr, long sc);

extern void THCharTensor_validXCorr3DRevptr(char *r_,
                                       char alpha,
                                       char *t_, long it, long ir, long ic,
                                       char *k_, long kt, long kr, long kc,
                                       long st, long sr, long sc);

extern void THCharTensor_conv3DRevger(THCharTensor *r_, char beta, char alpha, THCharTensor *t_, THCharTensor *k_, long sdepth, long srow, long scol);
extern void THCharTensor_conv3Dger(THCharTensor *r_, char beta, char alpha, THCharTensor *t_, THCharTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THCharTensor_conv3Dmv(THCharTensor *r_, char beta, char alpha, THCharTensor *t_, THCharTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THCharTensor_conv3Dmul(THCharTensor *r_, char beta, char alpha, THCharTensor *t_, THCharTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THCharTensor_conv3Dcmul(THCharTensor *r_, char beta, char alpha, THCharTensor *t_, THCharTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);





extern void THShortTensor_validXCorr2Dptr(short *r_,
                                    short alpha,
                                    short *t_, long ir, long ic,
                                    short *k_, long kr, long kc,
                                    long sr, long sc);

extern void THShortTensor_validConv2Dptr(short *r_,
                                   short alpha,
                                   short *t_, long ir, long ic,
                                   short *k_, long kr, long kc,
                                   long sr, long sc);

extern void THShortTensor_fullXCorr2Dptr(short *r_,
                                   short alpha,
                                   short *t_, long ir, long ic,
                                   short *k_, long kr, long kc,
                                   long sr, long sc);

extern void THShortTensor_fullConv2Dptr(short *r_,
                                  short alpha,
                                  short *t_, long ir, long ic,
                                  short *k_, long kr, long kc,
                                  long sr, long sc);

extern void THShortTensor_validXCorr2DRevptr(short *r_,
                                       short alpha,
                                       short *t_, long ir, long ic,
                                       short *k_, long kr, long kc,
                                       long sr, long sc);

extern void THShortTensor_conv2DRevger(THShortTensor *r_, short beta, short alpha, THShortTensor *t_, THShortTensor *k_, long srow, long scol);
extern void THShortTensor_conv2DRevgerm(THShortTensor *r_, short beta, short alpha, THShortTensor *t_, THShortTensor *k_, long srow, long scol);
extern void THShortTensor_conv2Dger(THShortTensor *r_, short beta, short alpha, THShortTensor *t_, THShortTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THShortTensor_conv2Dmv(THShortTensor *r_, short beta, short alpha, THShortTensor *t_, THShortTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THShortTensor_conv2Dmm(THShortTensor *r_, short beta, short alpha, THShortTensor *t_, THShortTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THShortTensor_conv2Dmul(THShortTensor *r_, short beta, short alpha, THShortTensor *t_, THShortTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THShortTensor_conv2Dcmul(THShortTensor *r_, short beta, short alpha, THShortTensor *t_, THShortTensor *k_, long srow, long scol, const char *vf, const char *xc);

extern void THShortTensor_validXCorr3Dptr(short *r_,
                                    short alpha,
                                    short *t_, long it, long ir, long ic,
                                    short *k_, long kt, long kr, long kc,
                                    long st, long sr, long sc);

extern void THShortTensor_validConv3Dptr(short *r_,
                                   short alpha,
                                   short *t_, long it, long ir, long ic,
                                   short *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THShortTensor_fullXCorr3Dptr(short *r_,
                                   short alpha,
                                   short *t_, long it, long ir, long ic,
                                   short *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THShortTensor_fullConv3Dptr(short *r_,
                                  short alpha,
                                  short *t_, long it, long ir, long ic,
                                  short *k_, long kt, long kr, long kc,
                                  long st, long sr, long sc);

extern void THShortTensor_validXCorr3DRevptr(short *r_,
                                       short alpha,
                                       short *t_, long it, long ir, long ic,
                                       short *k_, long kt, long kr, long kc,
                                       long st, long sr, long sc);

extern void THShortTensor_conv3DRevger(THShortTensor *r_, short beta, short alpha, THShortTensor *t_, THShortTensor *k_, long sdepth, long srow, long scol);
extern void THShortTensor_conv3Dger(THShortTensor *r_, short beta, short alpha, THShortTensor *t_, THShortTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THShortTensor_conv3Dmv(THShortTensor *r_, short beta, short alpha, THShortTensor *t_, THShortTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THShortTensor_conv3Dmul(THShortTensor *r_, short beta, short alpha, THShortTensor *t_, THShortTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THShortTensor_conv3Dcmul(THShortTensor *r_, short beta, short alpha, THShortTensor *t_, THShortTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);





extern void THIntTensor_validXCorr2Dptr(int *r_,
                                    int alpha,
                                    int *t_, long ir, long ic,
                                    int *k_, long kr, long kc,
                                    long sr, long sc);

extern void THIntTensor_validConv2Dptr(int *r_,
                                   int alpha,
                                   int *t_, long ir, long ic,
                                   int *k_, long kr, long kc,
                                   long sr, long sc);

extern void THIntTensor_fullXCorr2Dptr(int *r_,
                                   int alpha,
                                   int *t_, long ir, long ic,
                                   int *k_, long kr, long kc,
                                   long sr, long sc);

extern void THIntTensor_fullConv2Dptr(int *r_,
                                  int alpha,
                                  int *t_, long ir, long ic,
                                  int *k_, long kr, long kc,
                                  long sr, long sc);

extern void THIntTensor_validXCorr2DRevptr(int *r_,
                                       int alpha,
                                       int *t_, long ir, long ic,
                                       int *k_, long kr, long kc,
                                       long sr, long sc);

extern void THIntTensor_conv2DRevger(THIntTensor *r_, int beta, int alpha, THIntTensor *t_, THIntTensor *k_, long srow, long scol);
extern void THIntTensor_conv2DRevgerm(THIntTensor *r_, int beta, int alpha, THIntTensor *t_, THIntTensor *k_, long srow, long scol);
extern void THIntTensor_conv2Dger(THIntTensor *r_, int beta, int alpha, THIntTensor *t_, THIntTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THIntTensor_conv2Dmv(THIntTensor *r_, int beta, int alpha, THIntTensor *t_, THIntTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THIntTensor_conv2Dmm(THIntTensor *r_, int beta, int alpha, THIntTensor *t_, THIntTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THIntTensor_conv2Dmul(THIntTensor *r_, int beta, int alpha, THIntTensor *t_, THIntTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THIntTensor_conv2Dcmul(THIntTensor *r_, int beta, int alpha, THIntTensor *t_, THIntTensor *k_, long srow, long scol, const char *vf, const char *xc);

extern void THIntTensor_validXCorr3Dptr(int *r_,
                                    int alpha,
                                    int *t_, long it, long ir, long ic,
                                    int *k_, long kt, long kr, long kc,
                                    long st, long sr, long sc);

extern void THIntTensor_validConv3Dptr(int *r_,
                                   int alpha,
                                   int *t_, long it, long ir, long ic,
                                   int *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THIntTensor_fullXCorr3Dptr(int *r_,
                                   int alpha,
                                   int *t_, long it, long ir, long ic,
                                   int *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THIntTensor_fullConv3Dptr(int *r_,
                                  int alpha,
                                  int *t_, long it, long ir, long ic,
                                  int *k_, long kt, long kr, long kc,
                                  long st, long sr, long sc);

extern void THIntTensor_validXCorr3DRevptr(int *r_,
                                       int alpha,
                                       int *t_, long it, long ir, long ic,
                                       int *k_, long kt, long kr, long kc,
                                       long st, long sr, long sc);

extern void THIntTensor_conv3DRevger(THIntTensor *r_, int beta, int alpha, THIntTensor *t_, THIntTensor *k_, long sdepth, long srow, long scol);
extern void THIntTensor_conv3Dger(THIntTensor *r_, int beta, int alpha, THIntTensor *t_, THIntTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THIntTensor_conv3Dmv(THIntTensor *r_, int beta, int alpha, THIntTensor *t_, THIntTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THIntTensor_conv3Dmul(THIntTensor *r_, int beta, int alpha, THIntTensor *t_, THIntTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THIntTensor_conv3Dcmul(THIntTensor *r_, int beta, int alpha, THIntTensor *t_, THIntTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);





extern void THLongTensor_validXCorr2Dptr(long *r_,
                                    long alpha,
                                    long *t_, long ir, long ic,
                                    long *k_, long kr, long kc,
                                    long sr, long sc);

extern void THLongTensor_validConv2Dptr(long *r_,
                                   long alpha,
                                   long *t_, long ir, long ic,
                                   long *k_, long kr, long kc,
                                   long sr, long sc);

extern void THLongTensor_fullXCorr2Dptr(long *r_,
                                   long alpha,
                                   long *t_, long ir, long ic,
                                   long *k_, long kr, long kc,
                                   long sr, long sc);

extern void THLongTensor_fullConv2Dptr(long *r_,
                                  long alpha,
                                  long *t_, long ir, long ic,
                                  long *k_, long kr, long kc,
                                  long sr, long sc);

extern void THLongTensor_validXCorr2DRevptr(long *r_,
                                       long alpha,
                                       long *t_, long ir, long ic,
                                       long *k_, long kr, long kc,
                                       long sr, long sc);

extern void THLongTensor_conv2DRevger(THLongTensor *r_, long beta, long alpha, THLongTensor *t_, THLongTensor *k_, long srow, long scol);
extern void THLongTensor_conv2DRevgerm(THLongTensor *r_, long beta, long alpha, THLongTensor *t_, THLongTensor *k_, long srow, long scol);
extern void THLongTensor_conv2Dger(THLongTensor *r_, long beta, long alpha, THLongTensor *t_, THLongTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THLongTensor_conv2Dmv(THLongTensor *r_, long beta, long alpha, THLongTensor *t_, THLongTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THLongTensor_conv2Dmm(THLongTensor *r_, long beta, long alpha, THLongTensor *t_, THLongTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THLongTensor_conv2Dmul(THLongTensor *r_, long beta, long alpha, THLongTensor *t_, THLongTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THLongTensor_conv2Dcmul(THLongTensor *r_, long beta, long alpha, THLongTensor *t_, THLongTensor *k_, long srow, long scol, const char *vf, const char *xc);

extern void THLongTensor_validXCorr3Dptr(long *r_,
                                    long alpha,
                                    long *t_, long it, long ir, long ic,
                                    long *k_, long kt, long kr, long kc,
                                    long st, long sr, long sc);

extern void THLongTensor_validConv3Dptr(long *r_,
                                   long alpha,
                                   long *t_, long it, long ir, long ic,
                                   long *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THLongTensor_fullXCorr3Dptr(long *r_,
                                   long alpha,
                                   long *t_, long it, long ir, long ic,
                                   long *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THLongTensor_fullConv3Dptr(long *r_,
                                  long alpha,
                                  long *t_, long it, long ir, long ic,
                                  long *k_, long kt, long kr, long kc,
                                  long st, long sr, long sc);

extern void THLongTensor_validXCorr3DRevptr(long *r_,
                                       long alpha,
                                       long *t_, long it, long ir, long ic,
                                       long *k_, long kt, long kr, long kc,
                                       long st, long sr, long sc);

extern void THLongTensor_conv3DRevger(THLongTensor *r_, long beta, long alpha, THLongTensor *t_, THLongTensor *k_, long sdepth, long srow, long scol);
extern void THLongTensor_conv3Dger(THLongTensor *r_, long beta, long alpha, THLongTensor *t_, THLongTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THLongTensor_conv3Dmv(THLongTensor *r_, long beta, long alpha, THLongTensor *t_, THLongTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THLongTensor_conv3Dmul(THLongTensor *r_, long beta, long alpha, THLongTensor *t_, THLongTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THLongTensor_conv3Dcmul(THLongTensor *r_, long beta, long alpha, THLongTensor *t_, THLongTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);





extern void THFloatTensor_validXCorr2Dptr(float *r_,
                                    float alpha,
                                    float *t_, long ir, long ic,
                                    float *k_, long kr, long kc,
                                    long sr, long sc);

extern void THFloatTensor_validConv2Dptr(float *r_,
                                   float alpha,
                                   float *t_, long ir, long ic,
                                   float *k_, long kr, long kc,
                                   long sr, long sc);

extern void THFloatTensor_fullXCorr2Dptr(float *r_,
                                   float alpha,
                                   float *t_, long ir, long ic,
                                   float *k_, long kr, long kc,
                                   long sr, long sc);

extern void THFloatTensor_fullConv2Dptr(float *r_,
                                  float alpha,
                                  float *t_, long ir, long ic,
                                  float *k_, long kr, long kc,
                                  long sr, long sc);

extern void THFloatTensor_validXCorr2DRevptr(float *r_,
                                       float alpha,
                                       float *t_, long ir, long ic,
                                       float *k_, long kr, long kc,
                                       long sr, long sc);

extern void THFloatTensor_conv2DRevger(THFloatTensor *r_, float beta, float alpha, THFloatTensor *t_, THFloatTensor *k_, long srow, long scol);
extern void THFloatTensor_conv2DRevgerm(THFloatTensor *r_, float beta, float alpha, THFloatTensor *t_, THFloatTensor *k_, long srow, long scol);
extern void THFloatTensor_conv2Dger(THFloatTensor *r_, float beta, float alpha, THFloatTensor *t_, THFloatTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THFloatTensor_conv2Dmv(THFloatTensor *r_, float beta, float alpha, THFloatTensor *t_, THFloatTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THFloatTensor_conv2Dmm(THFloatTensor *r_, float beta, float alpha, THFloatTensor *t_, THFloatTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THFloatTensor_conv2Dmul(THFloatTensor *r_, float beta, float alpha, THFloatTensor *t_, THFloatTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THFloatTensor_conv2Dcmul(THFloatTensor *r_, float beta, float alpha, THFloatTensor *t_, THFloatTensor *k_, long srow, long scol, const char *vf, const char *xc);

extern void THFloatTensor_validXCorr3Dptr(float *r_,
                                    float alpha,
                                    float *t_, long it, long ir, long ic,
                                    float *k_, long kt, long kr, long kc,
                                    long st, long sr, long sc);

extern void THFloatTensor_validConv3Dptr(float *r_,
                                   float alpha,
                                   float *t_, long it, long ir, long ic,
                                   float *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THFloatTensor_fullXCorr3Dptr(float *r_,
                                   float alpha,
                                   float *t_, long it, long ir, long ic,
                                   float *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THFloatTensor_fullConv3Dptr(float *r_,
                                  float alpha,
                                  float *t_, long it, long ir, long ic,
                                  float *k_, long kt, long kr, long kc,
                                  long st, long sr, long sc);

extern void THFloatTensor_validXCorr3DRevptr(float *r_,
                                       float alpha,
                                       float *t_, long it, long ir, long ic,
                                       float *k_, long kt, long kr, long kc,
                                       long st, long sr, long sc);

extern void THFloatTensor_conv3DRevger(THFloatTensor *r_, float beta, float alpha, THFloatTensor *t_, THFloatTensor *k_, long sdepth, long srow, long scol);
extern void THFloatTensor_conv3Dger(THFloatTensor *r_, float beta, float alpha, THFloatTensor *t_, THFloatTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THFloatTensor_conv3Dmv(THFloatTensor *r_, float beta, float alpha, THFloatTensor *t_, THFloatTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THFloatTensor_conv3Dmul(THFloatTensor *r_, float beta, float alpha, THFloatTensor *t_, THFloatTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THFloatTensor_conv3Dcmul(THFloatTensor *r_, float beta, float alpha, THFloatTensor *t_, THFloatTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);





extern void THDoubleTensor_validXCorr2Dptr(double *r_,
                                    double alpha,
                                    double *t_, long ir, long ic,
                                    double *k_, long kr, long kc,
                                    long sr, long sc);

extern void THDoubleTensor_validConv2Dptr(double *r_,
                                   double alpha,
                                   double *t_, long ir, long ic,
                                   double *k_, long kr, long kc,
                                   long sr, long sc);

extern void THDoubleTensor_fullXCorr2Dptr(double *r_,
                                   double alpha,
                                   double *t_, long ir, long ic,
                                   double *k_, long kr, long kc,
                                   long sr, long sc);

extern void THDoubleTensor_fullConv2Dptr(double *r_,
                                  double alpha,
                                  double *t_, long ir, long ic,
                                  double *k_, long kr, long kc,
                                  long sr, long sc);

extern void THDoubleTensor_validXCorr2DRevptr(double *r_,
                                       double alpha,
                                       double *t_, long ir, long ic,
                                       double *k_, long kr, long kc,
                                       long sr, long sc);

extern void THDoubleTensor_conv2DRevger(THDoubleTensor *r_, double beta, double alpha, THDoubleTensor *t_, THDoubleTensor *k_, long srow, long scol);
extern void THDoubleTensor_conv2DRevgerm(THDoubleTensor *r_, double beta, double alpha, THDoubleTensor *t_, THDoubleTensor *k_, long srow, long scol);
extern void THDoubleTensor_conv2Dger(THDoubleTensor *r_, double beta, double alpha, THDoubleTensor *t_, THDoubleTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THDoubleTensor_conv2Dmv(THDoubleTensor *r_, double beta, double alpha, THDoubleTensor *t_, THDoubleTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THDoubleTensor_conv2Dmm(THDoubleTensor *r_, double beta, double alpha, THDoubleTensor *t_, THDoubleTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THDoubleTensor_conv2Dmul(THDoubleTensor *r_, double beta, double alpha, THDoubleTensor *t_, THDoubleTensor *k_, long srow, long scol, const char *vf, const char *xc);
extern void THDoubleTensor_conv2Dcmul(THDoubleTensor *r_, double beta, double alpha, THDoubleTensor *t_, THDoubleTensor *k_, long srow, long scol, const char *vf, const char *xc);

extern void THDoubleTensor_validXCorr3Dptr(double *r_,
                                    double alpha,
                                    double *t_, long it, long ir, long ic,
                                    double *k_, long kt, long kr, long kc,
                                    long st, long sr, long sc);

extern void THDoubleTensor_validConv3Dptr(double *r_,
                                   double alpha,
                                   double *t_, long it, long ir, long ic,
                                   double *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THDoubleTensor_fullXCorr3Dptr(double *r_,
                                   double alpha,
                                   double *t_, long it, long ir, long ic,
                                   double *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

extern void THDoubleTensor_fullConv3Dptr(double *r_,
                                  double alpha,
                                  double *t_, long it, long ir, long ic,
                                  double *k_, long kt, long kr, long kc,
                                  long st, long sr, long sc);

extern void THDoubleTensor_validXCorr3DRevptr(double *r_,
                                       double alpha,
                                       double *t_, long it, long ir, long ic,
                                       double *k_, long kt, long kr, long kc,
                                       long st, long sr, long sc);

extern void THDoubleTensor_conv3DRevger(THDoubleTensor *r_, double beta, double alpha, THDoubleTensor *t_, THDoubleTensor *k_, long sdepth, long srow, long scol);
extern void THDoubleTensor_conv3Dger(THDoubleTensor *r_, double beta, double alpha, THDoubleTensor *t_, THDoubleTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THDoubleTensor_conv3Dmv(THDoubleTensor *r_, double beta, double alpha, THDoubleTensor *t_, THDoubleTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THDoubleTensor_conv3Dmul(THDoubleTensor *r_, double beta, double alpha, THDoubleTensor *t_, THDoubleTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
extern void THDoubleTensor_conv3Dcmul(THDoubleTensor *r_, double beta, double alpha, THDoubleTensor *t_, THDoubleTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);






extern void THFloatTensor_gesv(THFloatTensor *rb_, THFloatTensor *ra_, THFloatTensor *b_, THFloatTensor *a_);
extern void THFloatTensor_gels(THFloatTensor *rb_, THFloatTensor *ra_, THFloatTensor *b_, THFloatTensor *a_);
extern void THFloatTensor_syev(THFloatTensor *re_, THFloatTensor *rv_, THFloatTensor *a_, const char *jobz, const char *uplo);
extern void THFloatTensor_geev(THFloatTensor *re_, THFloatTensor *rv_, THFloatTensor *a_, const char *jobvr);
extern void THFloatTensor_gesvd(THFloatTensor *ru_, THFloatTensor *rs_, THFloatTensor *rv_, THFloatTensor *a, const char *jobu);
extern void THFloatTensor_gesvd2(THFloatTensor *ru_, THFloatTensor *rs_, THFloatTensor *rv_, THFloatTensor *ra_, THFloatTensor *a, const char *jobu);
extern void THFloatTensor_getri(THFloatTensor *ra_, THFloatTensor *a);
extern void THFloatTensor_potri(THFloatTensor *ra_, THFloatTensor *a);
extern void THFloatTensor_potrf(THFloatTensor *ra_, THFloatTensor *a);




extern void THDoubleTensor_gesv(THDoubleTensor *rb_, THDoubleTensor *ra_, THDoubleTensor *b_, THDoubleTensor *a_);
extern void THDoubleTensor_gels(THDoubleTensor *rb_, THDoubleTensor *ra_, THDoubleTensor *b_, THDoubleTensor *a_);
extern void THDoubleTensor_syev(THDoubleTensor *re_, THDoubleTensor *rv_, THDoubleTensor *a_, const char *jobz, const char *uplo);
extern void THDoubleTensor_geev(THDoubleTensor *re_, THDoubleTensor *rv_, THDoubleTensor *a_, const char *jobvr);
extern void THDoubleTensor_gesvd(THDoubleTensor *ru_, THDoubleTensor *rs_, THDoubleTensor *rv_, THDoubleTensor *a, const char *jobu);
extern void THDoubleTensor_gesvd2(THDoubleTensor *ru_, THDoubleTensor *rs_, THDoubleTensor *rv_, THDoubleTensor *ra_, THDoubleTensor *a, const char *jobu);
extern void THDoubleTensor_getri(THDoubleTensor *ra_, THDoubleTensor *a);
extern void THDoubleTensor_potri(THDoubleTensor *ra_, THDoubleTensor *a);
extern void THDoubleTensor_potrf(THDoubleTensor *ra_, THDoubleTensor *a);







typedef struct THFile__ THFile;

extern int THFile_isOpened(THFile *self);
extern int THFile_isQuiet(THFile *self);
extern int THFile_isReadable(THFile *self);
extern int THFile_isWritable(THFile *self);
extern int THFile_isBinary(THFile *self);
extern int THFile_isAutoSpacing(THFile *self);
extern int THFile_hasError(THFile *self);

extern void THFile_binary(THFile *self);
extern void THFile_ascii(THFile *self);
extern void THFile_autoSpacing(THFile *self);
extern void THFile_noAutoSpacing(THFile *self);
extern void THFile_quiet(THFile *self);
extern void THFile_pedantic(THFile *self);
extern void THFile_clearError(THFile *self);


extern unsigned char THFile_readByteScalar(THFile *self);
extern char THFile_readCharScalar(THFile *self);
extern short THFile_readShortScalar(THFile *self);
extern int THFile_readIntScalar(THFile *self);
extern long THFile_readLongScalar(THFile *self);
extern float THFile_readFloatScalar(THFile *self);
extern double THFile_readDoubleScalar(THFile *self);

extern void THFile_writeByteScalar(THFile *self, unsigned char scalar);
extern void THFile_writeCharScalar(THFile *self, char scalar);
extern void THFile_writeShortScalar(THFile *self, short scalar);
extern void THFile_writeIntScalar(THFile *self, int scalar);
extern void THFile_writeLongScalar(THFile *self, long scalar);
extern void THFile_writeFloatScalar(THFile *self, float scalar);
extern void THFile_writeDoubleScalar(THFile *self, double scalar);


extern long THFile_readByte(THFile *self, THByteStorage *storage);
extern long THFile_readChar(THFile *self, THCharStorage *storage);
extern long THFile_readShort(THFile *self, THShortStorage *storage);
extern long THFile_readInt(THFile *self, THIntStorage *storage);
extern long THFile_readLong(THFile *self, THLongStorage *storage);
extern long THFile_readFloat(THFile *self, THFloatStorage *storage);
extern long THFile_readDouble(THFile *self, THDoubleStorage *storage);

extern long THFile_writeByte(THFile *self, THByteStorage *storage);
extern long THFile_writeChar(THFile *self, THCharStorage *storage);
extern long THFile_writeShort(THFile *self, THShortStorage *storage);
extern long THFile_writeInt(THFile *self, THIntStorage *storage);
extern long THFile_writeLong(THFile *self, THLongStorage *storage);
extern long THFile_writeFloat(THFile *self, THFloatStorage *storage);
extern long THFile_writeDouble(THFile *self, THDoubleStorage *storage);


extern long THFile_readByteRaw(THFile *self, unsigned char *data, long n);
extern long THFile_readCharRaw(THFile *self, char *data, long n);
extern long THFile_readShortRaw(THFile *self, short *data, long n);
extern long THFile_readIntRaw(THFile *self, int *data, long n);
extern long THFile_readLongRaw(THFile *self, long *data, long n);
extern long THFile_readFloatRaw(THFile *self, float *data, long n);
extern long THFile_readDoubleRaw(THFile *self, double *data, long n);
extern long THFile_readStringRaw(THFile *self, const char *format, char **str_);

extern long THFile_writeByteRaw(THFile *self, unsigned char *data, long n);
extern long THFile_writeCharRaw(THFile *self, char *data, long n);
extern long THFile_writeShortRaw(THFile *self, short *data, long n);
extern long THFile_writeIntRaw(THFile *self, int *data, long n);
extern long THFile_writeLongRaw(THFile *self, long *data, long n);
extern long THFile_writeFloatRaw(THFile *self, float *data, long n);
extern long THFile_writeDoubleRaw(THFile *self, double *data, long n);
extern long THFile_writeStringRaw(THFile *self, const char *str, long size);

extern void THFile_synchronize(THFile *self);
extern void THFile_seek(THFile *self, long position);
extern void THFile_seekEnd(THFile *self);
extern long THFile_position(THFile *self);
extern void THFile_close(THFile *self);
extern void THFile_free(THFile *self);





extern THFile *THDiskFile_new(const char *name, const char *mode, int isQuiet);
extern THFile *THPipeFile_new(const char *name, const char *mode, int isQuiet);

extern const char *THDiskFile_name(THFile *self);

extern int THDiskFile_isLittleEndianCPU(void);
extern int THDiskFile_isBigEndianCPU(void);
extern void THDiskFile_nativeEndianEncoding(THFile *self);
extern void THDiskFile_littleEndianEncoding(THFile *self);
extern void THDiskFile_bigEndianEncoding(THFile *self);






extern THFile *THMemoryFile_newWithStorage(THCharStorage *storage, const char *mode);
extern THFile *THMemoryFile_new(const char *mode);

extern THCharStorage *THMemoryFile_storage(THFile *self);


]]

local path = package.searchpath('libTH', package.cpath)
if not path and jit.os == 'OSX' then
   path = package.searchpath('libTH', package.cpath:gsub('%.so', '.dylib'))

end
assert(path, 'TH library not found')

return ffi.load(path)
