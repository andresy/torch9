CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

INCLUDE(TorchTemplate)

SET(luasrc init.lua argcheck.lua argtypes.lua Timer.lua Storage.lua
Tensor.lua apply.lua dimapply.lua iterator.lua print.lua dispatch.lua namedispatch.lua)

ADD_TORCH_TEMPLATE("maths.c" mathcfiles)
ADD_TORCH_TEMPLATE("blas.c" mathcfiles)
ADD_TORCH_TEMPLATE("maths.lua" luasrc "torch")

ADD_LIBRARY(maths SHARED ${mathcfiles})

FIND_PACKAGE(BLAS)
IF(BLAS_FOUND)
  ADD_DEFINITIONS(-DUSE_BLAS)
  TARGET_LINK_LIBRARIES(maths ${BLAS_LIBRARIES})
ENDIF(BLAS_FOUND)

# Note:
# Might be replaced by ${LUA_DIR} and ${LIBDIR} if using luarocks

INSTALL(TARGETS maths
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/torch")

INSTALL(FILES ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/torch")
