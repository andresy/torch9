CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

INCLUDE(TorchTemplate)

SET(luasrc env.lua init.lua register.lua timer.lua apply.lua dimapply.lua
display.lua random.lua file.lua diskfile.lua memoryfile.lua
serialization.lua clib.lua)

SET(mathcfiles random.c)

ADD_TORCH_TEMPLATE("maths.c" mathcfiles)
ADD_TORCH_TEMPLATE("blas.c" mathcfiles)

ADD_TORCH_TEMPLATE("storage.lua" luasrc "torch")
ADD_TORCH_TEMPLATE("tensor.lua" luasrc "torch")
ADD_TORCH_TEMPLATE("maths.lua" luasrc "torch")

ADD_LIBRARY(torch MODULE ${mathcfiles})

FIND_PACKAGE(BLAS)
IF(BLAS_FOUND)
  ADD_DEFINITIONS(-DUSE_BLAS)
  TARGET_LINK_LIBRARIES(torch ${BLAS_LIBRARIES})
ENDIF(BLAS_FOUND)

INSTALL(TARGETS torch
  DESTINATION "${LUA_CPATH_DIR}")

INSTALL(FILES ${luasrc}
  DESTINATION "${LUA_PATH_DIR}")
