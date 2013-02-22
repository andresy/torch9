Torch9 Core Library.
===============

Torch9 provides a Matlab-like environment for state-of-the-art machine
learning algorithms. It is easy to use and provides a very efficient
implementation, thanks to an easy and fast scripting language (Luajit) and
few C code lines for critical inner loops.

This package provides the core of the Torch9 distribution.

Note: Torch9 is still beta.

Installation
------------

### Requirements

*   C compiler
*   [luajit](http://www.luajit.org)
*   [cmake](http://www.cmake.org)
*   git
*   [luarocks](http://www.luarocks.org)

### Getting the last version from the git:

```sh
# first get the argcheck dependency
luarocks build https://raw.github.com/andresy/argcheck/master/rocks/argcheck-scm-1.rockspec

# now get torch
luarocks build https://github.com/andresy/torch9/blob/master/rocks/torch-9.scm-1.rockspec
```

Running
=======
```sh
$ luajit -ltorch
Torch 9.0 -- Copyright (C) 2001-2013 Idiap, NEC Labs, NYU. http://www.torch.ch/
LuaJIT 2.0.0 -- Copyright (C) 2005-2012 Mike Pall. http://luajit.org/
JIT: ON CMOV SSE2 SSE3 SSE4.1 fold cse dce fwd dse narrow loop abc sink fuse
> 
```

Documentation
=============

Coming soon.


