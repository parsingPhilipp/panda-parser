from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
   Extension("decomposition",       ["decomposition.pyx"]),
   Extension("parser.viterbi.viterbi",      ["parser/viterbi/viterbi.pyx"]),
   # Extension("parser.cfg_parser.cfg", ["parser/cfg_parser/cfg.pyx"], language='c++'),
   Extension("grammar.lcfrs",  ["grammar/lcfrs.pyx"]),
#   Extension("util.enumerator", ["util/enumerator.pyx"]),
   Extension("parser.cpp_cfg_parser.parser_wrapper", sources=["parser/cpp_cfg_parser/parser_wrapper.pyx", "parser/cpp_cfg_parser/cfg.cpp", "parser/cpp_cfg_parser/parser.cpp"], language='c++', extra_compile_args=["-std=c++11"], extra_link_args=["-std=c++11"]),
   Extension("parser.sDCP_parser.sdcp_parser_wrapper", sources=["parser/sDCP_parser/sdcp_parser_wrapper.pyx"], language='c++', extra_compile_args=["-std=c++14", "-Wall", "-gdwarf-3" , "-O3"], extra_link_args=["-std=c++14",  "-O3", "-gdwarf-3"]),
    Extension("parser.sDCP_parser.trace_manager", sources=["parser/sDCP_parser/trace_manager.pyx"], language='c++', extra_compile_args=["-std=c++14", "-Wall", "-gdwarf-3" , "-O3", "-msse2", "-ffast-math", "-ftree-vectorizer-verbose=2", "-fdump-tree-optimized", "-ftree-vectorize", "-lpthread", "-fopenmp"
        , "-rdynamic"], extra_link_args=["-std=c++14", "-fdump-tree-optimized", "-ftree-vectorizer-verbose=2",  "-O3", "-ftree-vectorize", "-gdwarf-3", "-lpthread", "-fopenmp"], include_dirs=["/usr/include/eigen3"]),
   Extension("parser.LCFRS.LCFRS_Parser_Wrapper", sources=["parser/LCFRS/LCFRS_Parser_Wrapper.pyx"], language='c++', extra_compile_args=["-std=c++14"], extra_link_args=["-std=c++14"])
]

setup(
  name = 'Hybrid Grammar Implementation',
  ext_modules = cythonize(ext_modules), requires=['Cython']
)
