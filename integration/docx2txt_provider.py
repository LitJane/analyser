import os
import sys
import warnings

import docx2txt
from overrides import overrides

from integration.doc_providers import DirDocProvider


class GDriveTestDocProvider(DirDocProvider):

  @overrides
  def getdocuments(self, query):
    return self.read_documents(query)

  def read_doc(self, fn):

    text = ''
    try:
      text = docx2txt.process(fn)
    except Exception:
      info_ = f"Cannot read file {fn} with docx2txt, error: {sys.exc_info()}"
      warnings.warn(info_, RuntimeWarning)

      try:
        os.system('antiword -w 0 "' + fn + '" > "' + fn + '.txt"')
        with open(fn + '.txt') as f:
          text = f.read()
      except Exception:
        exc_info_ = f"Cannot read file {fn} with antiword, error: {sys.exc_info()}"
        warnings.warn(exc_info_, RuntimeWarning)

    return text

  def read_documents(self, filename_prefix):
    filenames = self.list_filenames(filename_prefix)

    for file in filenames:
      try:
        text = self.read_doc(file)
        yield file, text

      except Exception:
        info_ = f"Cannot read file {file}, error: {sys.exc_info()}"
        warnings.warn(info_, RuntimeWarning)
