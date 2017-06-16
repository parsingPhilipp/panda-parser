from __future__ import print_function
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from git import Repo
from Cython.Build import cythonize
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst')) as f:
    long_description = f.read()

sterm_parser_repo = "git@gitlab.tcs.inf.tu-dresden.de:kilian/sterm-parser.git"
dep_name = "sterm-parser"
cython_dependency_src_path = path.join(here, "build", dep_name)
the_branch = 'origin/MT_Hypergraph'
the_commit = 'f043d0c8ee148aa16bc69e950e132ec8fc45bfdc'
sterm_include = [cython_dependency_src_path]
# change if eigen is installed in the user-local directory
# $COMPUTE_ROOT/usr/include/eigen3,
compute_root = ""
eigen_include = [compute_root + "/usr/include/eigen3", compute_root + "/usr/include"]
add_include = [compute_root + "/usr/local/include"]

class CustomBuildExtCommand(build_ext):
    """Customized setuptools install command - checks out repo with c++ parsing and training backend."""
    def run(self):
        print("Checking out commit " + the_commit + " of " + dep_name + ".")
        if not path.isdir(cython_dependency_src_path):
            repo = Repo.clone_from(sterm_parser_repo, cython_dependency_src_path)
        else:
            repo = Repo(cython_dependency_src_path)

        repo.remote('origin').fetch()
        # goal_branch = repo.commit(the_commit)
        repo.git.checkout(the_commit)
        # repo.head.reference = goal_branch
        # assert repo.head.is_detached
        # reset the index and working tree to match the pointed-to commit
        # repo.head.reset(index=True, working_tree=True)

        build_ext.run(self)


ext_modules=[
    Extension("decomposition",       ["decomposition.pyx"]),
    Extension("parser.viterbi.viterbi",      ["parser/viterbi/viterbi.pyx"]),
    Extension("parser.fst.lazy_composition", ["parser/fst/lazy_composition.pyx"], language='c++', extra_compile_args=['-std=c++14', '-lfst', '-ldl'], extra_link_args=['-lfst', '-ldl'], include_dirs=add_include),
    Extension("util.enumerator", sources=["util/enumerator.pyx"], language='c++'),
    Extension("grammar.lcfrs",  ["grammar/lcfrs.pyx"]),
    Extension("parser.cpp_cfg_parser.parser_wrapper", sources=["parser/cpp_cfg_parser/parser_wrapper.pyx", "parser/cpp_cfg_parser/cfg.cpp", "parser/cpp_cfg_parser/parser.cpp"], language='c++', extra_compile_args=["-std=c++11"], extra_link_args=["-std=c++11"]),
    Extension("parser.sDCP_parser.sdcp_parser_wrapper", sources=["parser/sDCP_parser/sdcp_parser_wrapper.pyx"], language='c++', extra_compile_args=["-std=c++14", "-Wall", "-gdwarf-3" , "-O3"], extra_link_args=["-std=c++14",  "-O3", "-gdwarf-3"], include_dirs=sterm_include),
    Extension("parser.LCFRS.LCFRS_Parser_Wrapper", sources=["parser/LCFRS/LCFRS_Parser_Wrapper.pyx"], language='c++',
              extra_compile_args=["-std=c++14"], extra_link_args=["-std=c++14"], include_dirs=sterm_include),
    Extension("parser.trace_manager.trace_manager", sources=["parser/trace_manager/trace_manager.pyx"], language='c++', extra_compile_args=["-std=c++14", "-Wall", "-gdwarf-3" , "-O3"], include_dirs=eigen_include+sterm_include),
    Extension("parser.supervised_trainer.trainer", sources=["parser/supervised_trainer/trainer.pyx"], language='c++', extra_compile_args=["-std=c++14", "-Wall", "-gdwarf-3", "-O3"], include_dirs=eigen_include+sterm_include),
    Extension("parser.LCFRS.LCFRS_trace_manager", sources=["parser/LCFRS/LCFRS_trace_manager.pyx"], language='c++',
              extra_compile_args=["-std=c++14", "-Wall", "-gdwarf-3", "-O3"], include_dirs=eigen_include+sterm_include),
    Extension("parser.sDCP_parser.sdcp_trace_manager", sources=["parser/sDCP_parser/sdcp_trace_manager.pyx"], language='c++',
              extra_compile_args=["-std=c++14", "-Wall", "-gdwarf-3", "-O3"], include_dirs=eigen_include+sterm_include),
    Extension("parser.trace_manager.sm_trainer_util", sources=["parser/trace_manager/sm_trainer_util.pyx"], language='c++', extra_compile_args=["-std=c++14", "-Wall", "-gdwarf-3" , "-O3", "-msse2", "-ftree-vectorizer-verbose=2", "-fdump-tree-optimized", "-ftree-vectorize", "-lpthread", "-fopenmp"
        , "-rdynamic"], extra_link_args=["-std=c++14", "-fdump-tree-optimized", "-ftree-vectorizer-verbose=2",  "-O3", "-ftree-vectorize", "-gdwarf-3", "-lpthread", "-fopenmp"], include_dirs=eigen_include+sterm_include),
    Extension("parser.coarse_to_fine_parser.ranker", sources=["parser/coarse_to_fine_parser/ranker.pyx"], language='c++', extra_compile_args=["-std=c++14", "-Wall", "-gdwarf-3" , "-O3", "-msse2", "-ftree-vectorizer-verbose=2", "-fdump-tree-optimized", "-ftree-vectorize", "-lpthread", "-fopenmp"
        , "-rdynamic"], extra_link_args=["-std=c++14", "-fdump-tree-optimized", "-ftree-vectorizer-verbose=2",  "-O3", "-ftree-vectorize", "-gdwarf-3", "-lpthread", "-fopenmp"], include_dirs=eigen_include+sterm_include),
    Extension("parser.trace_manager.score_validator", sources=["parser/trace_manager/score_validator.pyx"], language='c++', extra_compile_args=["-std=c++14", "-Wall", "-gdwarf-3" , "-O3", "-msse2", "-ftree-vectorizer-verbose=2", "-fdump-tree-optimized", "-ftree-vectorize", "-lpthread", "-fopenmp"
        , "-rdynamic"], extra_link_args=["-std=c++14", "-fdump-tree-optimized", "-ftree-vectorizer-verbose=2",  "-O3", "-ftree-vectorize", "-gdwarf-3", "-lpthread", "-fopenmp"], include_dirs=eigen_include+sterm_include),
    Extension("parser.trace_manager.sm_trainer", sources=["parser/trace_manager/sm_trainer.pyx"], language='c++', extra_compile_args=["-std=c++14", "-Wall", "-gdwarf-3" , "-O3", "-msse2", "-ftree-vectorizer-verbose=2", "-fdump-tree-optimized", "-ftree-vectorize", "-lpthread", "-fopenmp"
        , "-rdynamic"], extra_link_args=["-std=c++14", "-fdump-tree-optimized", "-ftree-vectorizer-verbose=2",  "-O3", "-ftree-vectorize", "-gdwarf-3", "-lpthread", "-fopenmp"], include_dirs=eigen_include+sterm_include)
]

setup(
    cmdclass={
        'build_ext': CustomBuildExtCommand,
    },
    name='hyberparse',
    version='0.2.1',
    description='Implementation of LCFRS/sDCP hybrid grammars',
    url='https://gitlab.tcs.inf.tu-dresden.de/hybrid-grammars/lcfrs-sdcp-hybrid-grammars',
    author='Kilian Gebhardt',
    author_email='kilian.gebhardt@tu-dresden.de',

    license=None,

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: NLP Researchers',
        'Topic :: Syntactic Parsing :: Grammar-based parsing formalisms',

        # Pick your license as you wish (should match "license" above)
        # 'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='parsing parse LCFRS hybrid grammar',

    packages=[],
    # install_requires=[],

    ext_modules=cythonize(ext_modules),
    requires=['Cython']
)
