package = "torch"
version = "9.scm-1"

source = {
   url = "git://github.com/andresy/torch9.git",
}

description = {
   summary = "Torch9",
   detailed = [[
         Torch9 provides a Matlab-like environment for state-of-the-art machine
         learning algorithms. It provides a very efficient implementation, thanks 
         to Luajit and few C code lines for critical inner loops.
   ]],
   homepage = "http://www.torch.ch",
   license = "BSD"
}

dependencies = {
   "lua >= 5.1",
   "argcheck >= 1",
   "class >= 1"
}

build = {
   type = "command",
   build_command = "cmake -E make_directory build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DLUA_PATH_DIR=$(LUADIR)/torch -DLUA_CPATH_DIR=$(LIBDIR) -DLUA_EXECUTABLE=$(LUA) .. && $(MAKE)",
   install_command = "cd build && $(MAKE) install"
}
