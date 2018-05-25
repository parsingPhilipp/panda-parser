import codecs
import sys
import tempfile

TRAINING = "TRAIN"
VALIDATION = "VALIDATION"
TESTING = "TEST"
TESTING_INPUT = "TEST_INPUT"
RESULT = "RESULT"


class Resource(object):
    def __init__(self, path, start=1, end=None):
        self.start = start
        self.end = end
        self.path = path

    def init(self):
        pass

    def finalize(self):
        pass


class CorpusFile(Resource):
    def __init__(self, path=None, start=None, end=None, limit=None, length_limit=None, header=None, exclude=None,
                 directory=None, logger=None, filter=None, type=None):
        super(CorpusFile, self).__init__(path, start, end)
        self.limit = limit
        self.length_limit = length_limit
        self.file = None
        self.header = header
        self.directory = directory
        self.logger = logger if logger is not None else sys.stdout
        if exclude is None:
            self.exclude = []
        else:
            self.exclude = exclude
        self.filter = filter
        self.type = type

    def init(self):
        if self.path is None:
            _, self.path = tempfile.mkstemp(dir=self.directory)

        self.file = codecs.open(self.path, mode='w', encoding="utf-8")
        if self.header is not None:
            self.file.write(self.header)
        print('Opened', self.path, file=self.logger)

    def finalize(self):
        self.file.close()

    def write(self, content):
        self.file.write(content)

    def __str__(self):
        attributes = [('path', self.path), ('length limit', self.length_limit), ('start', self.start),
                      ('end', self.end), ('limit', self.limit), ('exclude', self.exclude)]
        return '{' + ', '.join(map(lambda x: x[0] + ' : ' + str(x[1]), attributes)) + '}'


class Logger(object):
    def __init__(self, path=None):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        if path is None:
            self.log, path = tempfile.mkstemp()
        else:
            self.log = open(path, "a")

    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.stdout.flush()


class ScorerResource(Resource):
    def __init__(self, path=None, start=None, end=None):
        super(ScorerResource, self).__init__(path, start, end)

    def score(self, system, gold, secondaries=None):
        assert False

    def failure(self, gold):
        assert False


__all_ = ["TRAINING", "VALIDATION", "TESTING", "TESTING_INPUT", "RESULT", "Resource", "CorpusResource", "Logger",
          "ScorerResource"]
