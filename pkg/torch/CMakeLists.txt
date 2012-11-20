SET(src)

SET(luasrc init.lua argcheck.lua argtypes.lua Timer.lua Storage.lua Tensor.lua apply.lua dimapply.lua iterator.lua maths.lua)
  
ADD_TORCH_PACKAGE(torch "${src}" "${luasrc}")
ADD_LIBRARY(maths SHARED maths.c)

INSTALL(TARGETS maths
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/torch")
