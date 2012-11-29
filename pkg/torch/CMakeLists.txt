CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(luasrc init.lua argcheck.lua argtypes.lua Timer.lua Storage.lua
Tensor.lua apply.lua dimapply.lua iterator.lua maths.lua print.lua)
  
ADD_LIBRARY(maths SHARED maths.c)

# Note:
# Might be replaced by ${LUA_DIR} and ${LIBDIR} if using luarocks

INSTALL(TARGETS maths
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/torch")

INSTALL(FILES ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/torch")
