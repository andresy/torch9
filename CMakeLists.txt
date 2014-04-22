CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

INCLUDE(TorchTemplate)

SET(luasrc env.lua init.lua register.lua registernumbers.lua timer.lua apply.lua dimapply.lua
display.lua random.lua file.lua diskfile.lua memoryfile.lua
serialization.lua TH.lua)

ADD_TORCH_TEMPLATE("storage.lua" luasrc "torch")
ADD_TORCH_TEMPLATE("tensor.lua" luasrc "torch")
ADD_TORCH_TEMPLATE("maths.lua" luasrc "torch")
ADD_TORCH_TEMPLATE("lapack.lua" luasrc "torch")
ADD_TORCH_TEMPLATE("conv.lua" luasrc "torch")

INSTALL(FILES ${luasrc}
  DESTINATION "${LUA_PATH_DIR}")

# TH ###

FILE(RELATIVE_PATH TH_INSTALL_BIN_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LUA_CPATH_DIR}")
FILE(RELATIVE_PATH TH_INSTALL_LIB_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LUA_CPATH_DIR}")
SET(TH_INSTALL_INCLUDE_SUBDIR "include")
SET(TH_INSTALL_CMAKE_SUBDIR "cmake/TH")

ADD_SUBDIRECTORY(TH)
