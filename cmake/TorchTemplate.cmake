MACRO(ADD_TORCH_TEMPLATE filename)
  GET_FILENAME_COMPONENT(_ext  ${filename} EXT)
  GET_FILENAME_COMPONENT(_file ${filename} NAME_WE)
  IF(NOT ${ARGV2} STREQUAL "")
    LIST(APPEND tpl_${ARGV2}_${filename}_files "${CMAKE_CURRENT_BINARY_DIR}/${filename}")
  ENDIF()
  LIST(APPEND tpl_${ARGV2}_${filename}_files "${CMAKE_CURRENT_BINARY_DIR}/${_file}_byte${_ext}")
  LIST(APPEND tpl_${ARGV2}_${filename}_files "${CMAKE_CURRENT_BINARY_DIR}/${_file}_char${_ext}")
  LIST(APPEND tpl_${ARGV2}_${filename}_files "${CMAKE_CURRENT_BINARY_DIR}/${_file}_short${_ext}")
  LIST(APPEND tpl_${ARGV2}_${filename}_files "${CMAKE_CURRENT_BINARY_DIR}/${_file}_int${_ext}")
  LIST(APPEND tpl_${ARGV2}_${filename}_files "${CMAKE_CURRENT_BINARY_DIR}/${_file}_long${_ext}")
  LIST(APPEND tpl_${ARGV2}_${filename}_files "${CMAKE_CURRENT_BINARY_DIR}/${_file}_float${_ext}")
  LIST(APPEND tpl_${ARGV2}_${filename}_files "${CMAKE_CURRENT_BINARY_DIR}/${_file}_double${_ext}")

  ADD_CUSTOM_COMMAND(
    OUTPUT ${tpl_${ARGV2}_${filename}_files}
    COMMAND ${Torch_SOURCE_LUA} ARGS
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/template.lua"
    "${CMAKE_CURRENT_SOURCE_DIR}/${filename}"
    "${CMAKE_CURRENT_BINARY_DIR}/${filename}"
    "${ARGV2}"
    DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/${filename}"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/template.lua"
    ${Torch_SOURCE_LUA})

  SET_PROPERTY(SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${_file}_byte${_ext}" PROPERTY COMPILE_DEFINITIONS REAL_IS_BYTE byte=unsigned\ char)
  SET_PROPERTY(SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${_file}_char${_ext}" PROPERTY COMPILE_DEFINITIONS REAL_IS_CHAR byte=unsigned\ char)
  SET_PROPERTY(SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${_file}_short${_ext}" PROPERTY COMPILE_DEFINITIONS REAL_IS_SHORT byte=unsigned\ char)
  SET_PROPERTY(SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${_file}_int${_ext}" PROPERTY COMPILE_DEFINITIONS REAL_IS_INT byte=unsigned\ char)
  SET_PROPERTY(SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${_file}_long${_ext}" PROPERTY COMPILE_DEFINITIONS REAL_IS_LONG byte=unsigned\ char)
  SET_PROPERTY(SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${_file}_float${_ext}" PROPERTY COMPILE_DEFINITIONS REAL_IS_FLOAT byte=unsigned\ char)
  SET_PROPERTY(SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${_file}_double${_ext}" PROPERTY COMPILE_DEFINITIONS REAL_IS_DOUBLE byte=unsigned\ char)

  IF(${ARGC} GREATER 1)
    LIST(APPEND ${ARGV1} ${tpl_${ARGV2}_${filename}_files})
  ENDIF()

  IF(NOT ${ARGV2} STREQUAL "")
    ADD_CUSTOM_TARGET(tpl_tpl_${ARGV2}_${_file} ALL DEPENDS ${tpl_${ARGV2}_${filename}_files})
  ELSE()
    ADD_CUSTOM_TARGET(tpl_${_file} ALL DEPENDS ${tpl_${ARGV2}_${filename}_files})
  ENDIF()

ENDMACRO()
