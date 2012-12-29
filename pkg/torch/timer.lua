local torch = require 'torch'
local ffi = require 'ffi'

if jit.os == 'OSX' then
   ffi.cdef([[
     typedef long time_t;

     typedef struct timeval {
      time_t tv_sec;
      int tv_usec;
     };

    int gettimeofday(struct timeval* t, void* tzp);
  ]])
else
   ffi.cdef([[
     typedef long time_t;

     typedef struct timeval {
      time_t tv_sec;
      time_t tv_usec;
     };

    int gettimeofday(struct timeval* t, void* tzp);
  ]])
end

ffi.cdef([[
struct rusage {
             struct timeval ru_utime; /* user time used */
             struct timeval ru_stime; /* system time used */
             long ru_maxrss;          /* integral max resident set size */
             long ru_ixrss;           /* integral shared text memory size */
             long ru_idrss;           /* integral unshared data size */
             long ru_isrss;           /* integral unshared stack size */
             long ru_minflt;          /* page reclaims */
             long ru_majflt;          /* page faults */
             long ru_nswap;           /* swaps */
             long ru_inblock;         /* block input operations */
             long ru_oublock;         /* block output operations */
             long ru_msgsnd;          /* messages sent */
             long ru_msgrcv;          /* messages received */
             long ru_nsignals;        /* signals received */
             long ru_nvcsw;           /* voluntary context switches */
             long ru_nivcsw;          /* involuntary context switches */
};

int getrusage(int who, struct rusage *r_usage);

]])

local Timer = torch.class('torch.Timer')

Timer.RUSAGE_SELF = 0
Timer.RUSAGE_CHILDREN = -1

function Timer.real()
   local time = ffi.new("struct timeval")
   ffi.C.gettimeofday(time, nil)
   return (tonumber(time.tv_sec) + tonumber(time.tv_usec)/1000000.0)
end

function Timer.user()
   local time = ffi.new("struct rusage")
   ffi.C.getrusage(Timer.RUSAGE_SELF, time)
   return (tonumber(time.ru_utime.tv_sec) + tonumber(time.ru_utime.tv_usec)/1000000.0)
end

function Timer.sys()
   local time = ffi.new("struct rusage")
   ffi.C.getrusage(Timer.RUSAGE_SELF, time)
   return (tonumber(time.ru_stime.tv_sec) + tonumber(time.ru_stime.tv_usec)/1000000.0)
end

function Timer.new()
   local self = Timer.__init()
   self:reset()
   return self
end

function Timer:reset()
   self.__isRunning = true
   self.__totalrealtime = 0
   self.__totalusertime = 0
   self.__totalsystime = 0
   self.__startrealtime = Timer.real()
   self.__startusertime = Timer.user()
   self.__startsystime = Timer.sys()   
   return self
end

function Timer:stop()
   if self.__isRunning then
      self.__totalrealtime = self.__totalrealtime + Timer.real() - self.__startrealtime
      self.__totalusertime = self.__totalusertime + Timer.user() - self.__startusertime
      self.__totalsystime = self.__totalsystime + Timer.sys() - self.__startsystime
      self.__isRunning = false
   end
end

function Timer:resume()
   if not self.__isRunning then
      self.__startrealtime = Timer.real()
      self.__startusertime = Timer.user()
      self.__startsystime = Timer.sys()
      self.__isRunning = true
   end
end

function Timer:time()
  return {
     real = self.__isRunning and (self.__totalrealtime + Timer.real() - self.__startrealtime) or self.__totalrealtime,
     user = self.__isRunning and (self.__totalusertime + Timer.user() - self.__startusertime) or self.__totalusertime,
     sys  = self.__isRunning and (self.__totalsystime + Timer.sys() - self.__startsystime) or self.__totalsystime
  }
end

torch.Timer = torch.constructor(Timer)
