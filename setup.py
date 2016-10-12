from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
   Extension("decomposition",       ["decomposition.pyx"]),
   Extension("parser.viterbi.viterbi",      ["parser/viterbi/viterbi.pyx"]),
   # Extension("parser.cfg_parser.cfg", ["parser/cfg_parser/cfg.pyx"], language='c++'),
   Extension("grammar.LCFRS.lcfrs",  ["grammar/LCFRS/lcfrs.pyx"]),
   Extension("util.enumerator", ["util/enumerator.pyx"]),
   Extension("parser.cpp_cfg_parser.parser_wrapper", sources=["parser/cpp_cfg_parser/parser_wrapper.pyx", "parser/cpp_cfg_parser/cfg.cpp", "parser/cpp_cfg_parser/parser.cpp"], language='c++')
]

setup(
  name = 'Hybrid Grammar Implementation',
  ext_modules = cythonize(ext_modules), requires=['Cython']
)